"""Verify the separate reviewed response proposal without changing old locks."""
import hashlib
from importlib import metadata
from importlib import util
import json
from pathlib import Path
import sys

CORE_COMMIT = "21761ca8fd059439b58aa56a69bf8d1647f50e31"
CORE_VERSION = "0.8.2"
PG_VERSION = "0.9.1"
LOCK = Path(__file__).with_name("response-core.lock.json")


def verify_response_core():
    """Check installed distribution ownership and the owner's canonical map.

    No source-tree fallback, environment mutation, package install or download.
    The exact local wheel and companion archive remain required review inputs.
    """
    lock = json.loads(LOCK.read_text())
    if (lock["core"]["commit"] != CORE_COMMIT or lock["core"]["version"] != CORE_VERSION
            or lock["particlegan"]["version"] != PG_VERSION):
        raise RuntimeError("Music3 response proposal requires its canonical core0.8.2/PG0.9.1 lock")
    dist = metadata.distribution("particle-sliders-core")
    if dist.version != CORE_VERSION or metadata.version("particlegan") != PG_VERSION:
        raise RuntimeError("use the exact reviewed core0.8.2/PG0.9.1 response environment; old CLI has its own lock")
    registered = {str(path) for path in dist.files or ()}
    expected = lock["source_sha256"]
    root = Path(dist.locate_file("")).resolve()
    paths = {}
    for name, digest in expected.items():
        path = Path(dist.locate_file(name)).resolve()
        if name not in registered or not path.is_relative_to(root) or not path.is_file():
            raise RuntimeError("response source must belong to the installed wheel: " + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("installed response source differs from canonical lock: " + name)
        paths[name] = path
    for name in registered:
        if (name.startswith(("particle_sliders/", "concept_slider_core/"))
                and "__pycache__" not in Path(name).parts and name not in expected):
            raise RuntimeError("unexpected source in the reviewed core distribution: " + name)
    direct = dist.read_text("direct_url.json")
    if direct:
        provenance = json.loads(direct)
        if "dir_info" in provenance:
            raise RuntimeError("source-tree/editable core installs are unsupported")
        if "vcs_info" in provenance:
            vcs = provenance["vcs_info"]
            url = provenance.get("url", "").rstrip("/").removesuffix(".git")
            if (url != lock["core"]["repository"] or vcs.get("vcs") != "git"
                    or vcs.get("commit_id") != CORE_COMMIT or vcs.get("requested_revision") != CORE_COMMIT):
                raise RuntimeError("core VCS provenance differs from the response proposal")
        elif "archive_info" in provenance:
            archive = provenance["archive_info"]
            hashes = archive.get("hashes", {})
            digest = hashes.get("sha256", archive.get("hash", "").removeprefix("sha256="))
            if digest != lock["core"]["wheel_sha256"]:
                raise RuntimeError("core wheel archive differs from the response proposal")
        else:
            raise RuntimeError("unrecognized response core installation provenance")
    # Reject a previously imported foreign module even when metadata is correct.
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] not in ("particle_sliders", "concept_slider_core"):
            continue
        relative = name.replace(".", "/")
        target = paths.get(relative + ".py", paths.get(relative + "/__init__.py"))
        origin = getattr(module, "__file__", None)
        if target is None or not isinstance(origin, str) or Path(origin).resolve() != target:
            raise RuntimeError("foreign response core module was already imported: " + name)
        if hasattr(module, "__path__") and tuple(Path(path).resolve() for path in module.__path__) != (target.parent,):
            raise RuntimeError("foreign response package search path: " + name)
    spec = util.find_spec("particle_sliders")
    if (spec is None or not isinstance(spec.origin, str)
            or Path(spec.origin).resolve() != paths["particle_sliders/__init__.py"]
            or tuple(Path(path).resolve() for path in spec.submodule_search_locations or ())
               != (paths["particle_sliders/__init__.py"].parent,)):
        raise RuntimeError("Python would import core from outside the reviewed installed distribution")
    return lock
