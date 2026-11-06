"""Pin and verify the AgentX installation used by external validation.

The development and external paths use the same AgentX interfaces. External
validation additionally verifies the installed distribution's origin and
content before it measures anything, keeping evaluator code outside the
candidate's control.

## What can actually go wrong, in order of likelihood

**1. `PYTHONPATH` can shadow the installed package.** A source checkout on the
path may otherwise replace the verified installation silently.

**2. A version collision that is not a coincidence.** The pinned distribution
calls itself ``aiperf 0.12.0``, and there is an unrelated package on PyPI with
that exact name and that exact version. A version check alone therefore proves
nothing: `direct_url.json` -- which records the git remote and commit uv
installed from -- is what distinguishes them.

**3. Somebody edits the installed copy.** The environment is a directory on a
shared machine. A content digest over the installed package is the only thing
that notices.

So this module checks all three, refuses rather than warns, and says which one
failed.

## Why a refusal and not a warning

A warning on a measurement path is read once, by the person who already knew.
The failure mode here is silent by construction -- a shadowed import produces
plausible numbers -- so the only useful response is to stop. `require()` raises
:class:`AgentXNotPinned` with the specific mismatch, and every final-validation
entry point calls it before it takes a device.

## Keeping the manifest honest

    # after a deliberate re-install of the pin
    $SERVERBENCH_PY -m final_validate.agentx_pin --write

The manifest is committed. Regenerating it is how a pin CHANGE is recorded, and
changing a pin invalidates every measurement taken under the old one -- the same
rule the three baseline images live under.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import sysconfig
from pathlib import Path

#: `SemiAnalysisAI/agentx-harness` at the commit this suite measures with.
#: Recorded in the installed distribution's `direct_url.json`, which is the only
#: field that distinguishes this `aiperf 0.12.0` from the unrelated PyPI package
#: of the same name and version.
PINNED_COMMIT = "56a0cf70f4c0359454ee4bd15a17770b541a3e3e"
PINNED_VERSION = "0.12.0"
PINNED_REMOTE = "https://github.com/SemiAnalysisAI/agentx-harness.git"

#: Where the digest lives. Committed: it is how a pin change is made visible.
MANIFEST = (Path(__file__).resolve().parents[2]
            / "artifacts" / "agentx" / "aiperf-pin.json")

#: Files whose contents are hashed. `.py` only: the digest exists to catch an
#: edit to the measurement logic, and hashing caches would make it depend on
#: whichever interpreter imported the package last.
HASHED_SUFFIXES = (".py",)


class AgentXNotPinned(RuntimeError):
    """The AgentX about to measure something is not the pinned one."""


def _purelib() -> Path:
    return Path(sysconfig.get_paths()["purelib"]).resolve()


def _shadowing_entries() -> list[str]:
    """`PYTHONPATH` entries that hold an importable ``aiperf``.

    Checked as a STRING LIST rather than by comparing the resolved import,
    because a shadow that is byte-identical to the install resolves to a
    different path and behaves identically -- so the resolved import cannot
    distinguish "correct" from "about to break on the next edit".
    """
    out = []
    for entry in (os.environ.get("PYTHONPATH") or "").split(os.pathsep):
        if not entry.strip():
            continue
        here = Path(entry)
        if (here / "aiperf" / "__init__.py").is_file():
            out.append(str(here))
    return out


def digest_of(package_dir: Path) -> tuple[str, int]:
    """A stable digest over the installed package's source, and its file count.

    Paths are hashed alongside contents and in sorted order, so a file that is
    deleted or moved changes the digest -- a content-only hash of a set of files
    is invariant under a rename, and a renamed module is a different library.
    """
    files = sorted(p for p in package_dir.rglob("*")
                   if p.is_file() and p.suffix in HASHED_SUFFIXES
                   and "__pycache__" not in p.parts)
    running = hashlib.sha256()
    for path in files:
        running.update(str(path.relative_to(package_dir)).encode("utf-8"))
        running.update(b"\0")
        running.update(path.read_bytes())
        running.update(b"\0")
    return running.hexdigest(), len(files)


def describe() -> dict:
    """Everything checkable about the AgentX this interpreter would import.

    Never raises for a mismatch -- it reports. `require()` is what refuses, so a
    report can be written into a run's evidence even when the run is being
    stopped because of what it says.
    """
    import importlib.metadata as md

    body: dict = {
        "expected_commit": PINNED_COMMIT,
        "expected_version": PINNED_VERSION,
        "expected_remote": PINNED_REMOTE,
        "interpreter": sys.executable,
        "purelib": str(_purelib()),
        "pythonpath_shadows": _shadowing_entries(),
    }
    try:
        import aiperf
    except Exception as exc:                        # noqa: BLE001
        body["import_error"] = f"{type(exc).__name__}: {exc}"
        return body

    module_file = Path(aiperf.__file__).resolve()
    package_dir = module_file.parent
    body["module_file"] = str(module_file)
    body["inside_purelib"] = str(package_dir).startswith(str(_purelib()))
    try:
        body["version"] = md.version("aiperf")
    except Exception as exc:                        # noqa: BLE001
        body["version_error"] = f"{type(exc).__name__}: {exc}"

    direct_url = None
    for info in _purelib().glob("aiperf-*.dist-info/direct_url.json"):
        direct_url = json.loads(info.read_text("utf-8"))
        break
    body["direct_url"] = direct_url
    body["commit"] = ((direct_url or {}).get("vcs_info") or {}).get("commit_id")
    body["remote"] = (direct_url or {}).get("url")

    body["digest"], body["file_count"] = digest_of(package_dir)
    return body


def load_manifest() -> dict | None:
    if not MANIFEST.is_file():
        return None
    return json.loads(MANIFEST.read_text("utf-8"))


def require() -> dict:
    """Refuse unless the AgentX about to be imported is the pinned one.

    Returns the description, so a caller can write it into the run's evidence:
    a measurement that cannot say which library produced it is a measurement
    nobody can reproduce.
    """
    got = describe()
    problems: list[str] = []

    if "import_error" in got:
        problems.append(
            f"aiperf does not import: {got['import_error']}. The final "
            f"validator measures through AgentX and has no fallback client")
    else:
        if got["pythonpath_shadows"]:
            problems.append(
                f"PYTHONPATH carries an importable `aiperf` at "
                f"{got['pythonpath_shadows']}. That is the vendored tree the "
                f"coding agent can read, and it SHADOWS the installed build "
                f"without any error -- while the copies are byte-identical "
                f"the substitution is invisible until one is edited. Unset "
                f"PYTHONPATH for the final-validation path")
        if not got.get("inside_purelib"):
            problems.append(
                f"aiperf resolved to {got.get('module_file')}, which is outside "
                f"this interpreter's site-packages ({got['purelib']}). The "
                f"final validator measures only with the installed pin")
        if got.get("version") != PINNED_VERSION:
            problems.append(
                f"aiperf version is {got.get('version')!r}, pinned "
                f"{PINNED_VERSION!r}")
        if got.get("commit") != PINNED_COMMIT:
            problems.append(
                f"installed from commit {got.get('commit')!r}, pinned "
                f"{PINNED_COMMIT!r}. Note that a version check alone would NOT "
                f"have caught this: an unrelated PyPI package is also called "
                f"aiperf 0.12.0, and `direct_url.json` is the only field that "
                f"tells them apart")
        if got.get("remote") != PINNED_REMOTE:
            problems.append(
                f"installed from {got.get('remote')!r}, pinned "
                f"{PINNED_REMOTE!r}")

        manifest = load_manifest()
        if manifest is None:
            problems.append(
                f"no digest manifest at {MANIFEST}. Write one with "
                f"`python -m final_validate.agentx_pin --write` after "
                f"verifying the install, or the 'unchanged' half of this check "
                f"is not being made at all")
        elif manifest.get("digest") != got.get("digest"):
            problems.append(
                f"the installed aiperf source digest is {got.get('digest')} "
                f"({got.get('file_count')} files) and the manifest records "
                f"{manifest.get('digest')} ({manifest.get('file_count')} "
                f"files). Somebody edited the installed copy, or the pin moved "
                f"without the manifest being rewritten")

    if problems:
        raise AgentXNotPinned(
            "the AgentX about to measure is not the pinned one:\n  - "
            + "\n  - ".join(problems))
    return got


def _write() -> int:
    got = describe()
    if "import_error" in got:
        sys.stderr.write(f"cannot describe aiperf: {got['import_error']}\n")
        return 3
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "schema": "serverbench.agentx_pin/1",
        "remote": got.get("remote"),
        "commit": got.get("commit"),
        "version": got.get("version"),
        "digest": got.get("digest"),
        "file_count": got.get("file_count"),
        "hashed_suffixes": list(HASHED_SUFFIXES),
        "note": (
            "The AgentX the FINAL VALIDATOR measures with. Committed so a pin "
            "change is a visible edit to this file rather than a quiet "
            "difference between two hosts. Changing the pin invalidates every "
            "measurement taken under the old one -- the same rule the three "
            "baseline container images live under. Regenerate only after a "
            "deliberate re-install, and re-measure everything afterwards."),
        "why_digest_and_not_version": (
            "an unrelated PyPI distribution is also called aiperf 0.12.0, so a "
            "version check proves nothing. `direct_url.json` identifies the "
            "git commit; this digest is what notices an edit to the installed "
            "files afterwards."),
    }
    MANIFEST.write_text(json.dumps(body, indent=1) + "\n", "utf-8")
    print(f"wrote {MANIFEST}")
    print(f"  commit {body['commit']}")
    print(f"  digest {body['digest']} over {body['file_count']} .py files")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--write" in args:
        return _write()
    got = describe()
    print(json.dumps(got, indent=1))
    try:
        require()
    except AgentXNotPinned as exc:
        sys.stderr.write(f"\nREFUSED: {exc}\n")
        return 3
    print("\nPINNED: this is the AgentX the final validator measures with.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
