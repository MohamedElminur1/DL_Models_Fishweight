"""Freeze guard (ANALYSIS_PLAN.md section 12).

`assert_test_allowed()` must be called before any code fits a final model, predicts an outer
test fold or computes a metric from such predictions. It passes only when

(a) the annotated git tag `analysis-plan-frozen` exists and its commit SHA is recorded in
    decisions.tsv, and
(b) ANALYSIS_PLAN.md, src/, configs/, data/processed/ and uv.lock in the working tree are
    identical to that tag (no difference to the tagged files and no untracked file in those
    paths), or identical to a tag `analysis-fix-N` whose ANALYSIS_PLAN.md equals the frozen
    one.

It fails closed: a missing git, tag, file or decisions entry all raise GuardError. The tag
name and the guarded paths are fixed here on purpose; they are not read from a config file.
Files that git ignores (such as __pycache__) do not count as untracked.

`assert_files_committed` is the second check of the analysis: evaluate.py uses it to refuse
to score anything unless the primary-model selection files are committed and unmodified.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TAG = "analysis-plan-frozen"
FIX_TAG_PATTERN = re.compile(r"^analysis-fix-[0-9]+$")
PLAN_FILE = "ANALYSIS_PLAN.md"
GUARDED_PATHS = (PLAN_FILE, "src", "configs", "data/processed", "uv.lock")
DECISIONS_FILE = "decisions.tsv"


class GuardError(RuntimeError):
    """Outer test data may not be used: the analysis is not frozen, or it changed after the freeze."""


def _git(args: list[str]) -> subprocess.CompletedProcess:
    """Run git in REPO_ROOT, ignoring GIT_* environment variables that could redirect it."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, env=env, capture_output=True, check=False
        )
    except OSError as exc:
        raise GuardError(f"git could not be run, so the freeze cannot be checked: {exc}") from exc


def _tag_commit(tag: str) -> str | None:
    """The commit a tag points to, or None when the tag does not exist."""
    result = _git(["rev-parse", "--verify", "--quiet", f"refs/tags/{tag}^{{commit}}"])
    return result.stdout.decode().strip() if result.returncode == 0 else None


def _file_at(commit: str, path: str) -> bytes:
    """The bytes of `path` as stored at `commit`; GuardError when git cannot show it."""
    result = _git(["show", f"{commit}:{path}"])
    if result.returncode != 0:
        raise GuardError(f"{path} cannot be read at {commit[:12]}: {result.stderr.decode().strip()}")
    return result.stdout


def _differences(commit: str) -> list[str]:
    """Guarded paths that differ from `commit` or are untracked; an empty list means identical."""
    changed = _git(["diff", "--name-only", "--no-ext-diff", commit, "--", *GUARDED_PATHS])
    if changed.returncode != 0:
        raise GuardError(f"git diff failed: {changed.stderr.decode().strip()}")
    untracked = _git(["ls-files", "--others", "--exclude-standard", "--", *GUARDED_PATHS])
    if untracked.returncode != 0:
        raise GuardError(f"git ls-files failed: {untracked.stderr.decode().strip()}")
    names = [line for line in changed.stdout.decode().splitlines() if line]
    names += [f"{line} (untracked)" for line in untracked.stdout.decode().splitlines() if line]
    return names


def _fix_tags() -> list[str]:
    """Names of the tags analysis-fix-N, in increasing N."""
    result = _git(["tag", "--list", "analysis-fix-*"])
    names = [n for n in result.stdout.decode().split() if FIX_TAG_PATTERN.match(n)]
    return sorted(names, key=lambda n: int(n.rsplit("-", 1)[1]))


def assert_test_allowed() -> dict[str, str]:
    """Raise GuardError unless the analysis is frozen and unchanged; otherwise return provenance.

    Returns a dict with the frozen tag, its commit, the SHA-256 of the frozen plan, and the
    tag the working tree matches (`matched_tag`, `matched_commit`: the frozen tag itself, or
    an analysis-fix-N tag). Call it first in every code path that fits a final model or
    predicts an outer test fold, and before computing any metric on test predictions.
    """
    kind = _git(["cat-file", "-t", f"refs/tags/{TAG}"])
    if kind.returncode != 0:
        raise GuardError(
            f"git tag '{TAG}' does not exist: the analysis plan is not frozen, so outer test "
            "folds may not be fitted or predicted (plan section 12)"
        )
    if kind.stdout.decode().strip() != "tag":
        raise GuardError(f"'{TAG}' is a lightweight tag; the freeze needs an annotated tag (git tag -a)")
    commit = _tag_commit(TAG)
    if commit is None:
        raise GuardError(f"git tag '{TAG}' does not point to a commit")

    decisions = REPO_ROOT / DECISIONS_FILE
    if not decisions.is_file():
        raise GuardError(f"{DECISIONS_FILE} is missing, so the freeze commit is not recorded")
    if commit not in decisions.read_text(encoding="utf-8"):
        raise GuardError(f"the commit of tag '{TAG}' ({commit[:12]}) is not recorded in {DECISIONS_FILE}")

    frozen_plan = _file_at(commit, PLAN_FILE)
    if not (REPO_ROOT / PLAN_FILE).is_file():
        raise GuardError(f"{PLAN_FILE} is missing from the working tree")
    provenance = {"tag": TAG, "commit": commit, "plan_sha256": hashlib.sha256(frozen_plan).hexdigest()}

    differences = _differences(commit)
    if not differences:
        return {**provenance, "matched_tag": TAG, "matched_commit": commit}
    for name in _fix_tags():
        fix_commit = _tag_commit(name)
        if fix_commit is None or _file_at(fix_commit, PLAN_FILE) != frozen_plan:
            continue
        if not _differences(fix_commit):
            return {**provenance, "matched_tag": name, "matched_commit": fix_commit}
    raise GuardError(
        f"the working tree differs from tag '{TAG}' in: {', '.join(differences)}; it also matches "
        f"no analysis-fix-N tag with the frozen {PLAN_FILE} (plan section 12)"
    )


def assert_files_committed(paths: list[str]) -> None:
    """Raise GuardError unless every path is tracked by git and unmodified since HEAD.

    "Unmodified" covers the working tree and the index: a file that is new, staged but not
    committed, edited or deleted is refused. Paths are relative to the repository root.
    """
    for path in paths:
        tracked = _git(["ls-files", "--error-unmatch", "--", path])
        if tracked.returncode != 0:
            raise GuardError(f"{path} is not tracked by git: commit it before this step")
        in_head = _git(["cat-file", "-e", f"HEAD:{path}"])
        if in_head.returncode != 0:
            raise GuardError(f"{path} is not in the last commit: commit it before this step")
        changed = _git(["diff", "--quiet", "HEAD", "--", path])
        if changed.returncode != 0:
            raise GuardError(f"{path} differs from the last commit: commit or restore it before this step")
