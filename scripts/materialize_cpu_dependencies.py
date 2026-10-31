"""Owner-machine-only replay of read-only installed CPU dependencies.

Run with the named source environment's Python. No .pth or editable installation
is copied, and core/PG/HNDL/HyperGAN/product packages are installed separately.
Dependency bytes are reused without pins/hashes; this is not clean-install or
portable dependency-resolution evidence.
"""
import argparse
from importlib import metadata
import json
from pathlib import Path

from packaging.requirements import Requirement


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('packages', nargs='+')
    args = parser.parse_args()
    destination = args.destination.resolve()
    pending, versions = list(args.packages), {}
    while pending:
        name = pending.pop()
        key = name.lower().replace('_', '-')
        if key in versions:
            continue
        if key == 'setuptools':
            # Torch declares setuptools as a runtime dependency. The caller
            # installs the patched wheel locally; never symlink an install
            # target that pip will later replace.
            versions[key] = dict(provided_by='explicit local patched-wheel install')
            continue
        dist = metadata.distribution(name)
        versions[key] = dict(version=dist.version, source=str(dist.locate_file('')))
        for relative in dist.files or ():
            parts = Path(str(relative)).parts
            if not parts or parts[0] in ('.', '..') or parts[0].endswith('.pth'):
                continue
            target = destination / parts[0]
            source = Path(dist.locate_file(parts[0])).resolve()
            if not target.exists() and source.exists():
                target.symlink_to(source, target_is_directory=source.is_dir())
        for dependency in dist.requires or ():
            requirement = Requirement(dependency)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    print(json.dumps(versions, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
