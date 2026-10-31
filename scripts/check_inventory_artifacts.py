"""Check actual wheel/sdist data and source bytes; no network access."""
import argparse
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import tarfile
import zipfile


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stable_dist', type=Path)
    parser.add_argument('hndl_dist', type=Path)
    parser.add_argument('--stable-site', type=Path, required=True)
    parser.add_argument('--hndl-site', type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    checks = {}
    for folder, site, distribution, package, source, data in (
        (args.stable_dist, args.stable_site, 'music3_particle_sliders-0.2.0', 'music3', repo / 'music3',
         ['response-core.lock.json', 'data/sample-cards.json']),
        (args.hndl_dist, args.hndl_site, 'music3_particle_sliders_hndl-0.1.0', 'music3_hndl',
         repo / 'cohorts/hndl/src/music3_hndl', ['core.lock.json', 'critic.hndl']),
    ):
        wheel = folder / (distribution + '-py3-none-any.whl')
        archive = folder / (distribution + '.tar.gz')
        with zipfile.ZipFile(wheel) as built:
            assert built.testzip() is None
            expected = [p for p in source.rglob('*') if p.is_file()
                        and '__pycache__' not in p.parts and p.suffix in ('.py', '.json', '.hndl')]
            for path in expected:
                name = package + '/' + path.relative_to(source).as_posix()
                assert built.read(name) == path.read_bytes(), name
                assert (site / name).read_bytes() == path.read_bytes(), name
            for name in data:
                assert built.read(package + '/' + name) == (source / name).read_bytes()
            metadata_name = next(n for n in built.namelist() if n.endswith('.dist-info/METADATA'))
            metadata = Parser().parsestr(built.read(metadata_name).decode())
            assert any(' @ git+' in dependency for dependency in metadata.get_all('Requires-Dist', []))
        with tarfile.open(archive) as built:
            for name in data:
                suffix = '/' + ('src/' if package == 'music3_hndl' else '') + package + '/' + name
                matches = [member for member in built.getmembers() if member.name.endswith(suffix)]
                assert len(matches) == 1, suffix
                assert built.extractfile(matches[0]).read() == (source / name).read_bytes()
        checks[distribution] = dict(wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
            sdist_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            package_data=data, source_files_checked=len(expected), installed_files_checked=len(expected),
            direct_vcs_dependency=True,
            pypi_upload_supported=False)
    assert (repo / 'music3/data/sample-cards.json').read_bytes() == (repo / 'configs/music3/sample-cards.json').read_bytes()
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    main()
