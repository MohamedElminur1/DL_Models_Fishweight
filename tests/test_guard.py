"""Unit tests for src/guard.py on throwaway git repositories (never the real repository)."""

import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from src import guard

PLAN = "plan v1\n"


def git(repo: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.org",
         "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *args],
        cwd=repo, env=env, check=True, capture_output=True, text=True,
    )
    return done.stdout.strip()


def commit_all(repo: Path, message: str) -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", message)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A repo with a committed plan, src/, configs/, data/processed/ and uv.lock, no tag yet,
    and the guard pointed at it."""
    git(tmp_path, "init", "-q")
    (tmp_path / "ANALYSIS_PLAN.md").write_text(PLAN)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "model.py").write_text("X = 1\n")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "phase4.yaml").write_text("a: 1\n")
    (tmp_path / "data" / "processed").mkdir(parents=True)
    (tmp_path / "data" / "processed" / "t1_table.csv").write_text("pond,w\n1,2.9\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "decisions.tsv").write_text("ts\tdecision\n")
    (tmp_path / ".gitignore").write_text("__pycache__/\n")
    commit_all(tmp_path, "scaffold")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    return tmp_path


def freeze(repo: Path, record: bool = True) -> str:
    """Annotated tag on HEAD and, unless `record` is false, its commit SHA in decisions.tsv."""
    git(repo, "tag", "-a", guard.TAG, "-m", "freeze")
    sha = git(repo, "rev-parse", f"{guard.TAG}^{{commit}}")
    if record:
        with (repo / "decisions.tsv").open("a") as handle:
            handle.write(f"2026-10-14T12:00:00Z\tfroze the plan at {sha}\n")
    return sha


# ------------------------------------------------------------------------------ the tag

def test_no_tag_raises(repo):
    with pytest.raises(guard.GuardError, match="does not exist"):
        guard.assert_test_allowed()


def test_lightweight_tag_raises(repo):
    git(repo, "tag", guard.TAG)
    with pytest.raises(guard.GuardError, match="lightweight"):
        guard.assert_test_allowed()


def test_tag_whose_commit_is_not_recorded_in_decisions_raises(repo):
    freeze(repo, record=False)
    with pytest.raises(guard.GuardError, match="not recorded"):
        guard.assert_test_allowed()


def test_a_short_sha_in_decisions_is_not_enough(repo):
    sha = freeze(repo, record=False)
    (repo / "decisions.tsv").write_text(f"froze at {sha[:12]}\n")
    with pytest.raises(guard.GuardError, match="not recorded"):
        guard.assert_test_allowed()


def test_missing_decisions_file_raises(repo):
    freeze(repo)
    (repo / "decisions.tsv").unlink()
    with pytest.raises(guard.GuardError, match="decisions.tsv"):
        guard.assert_test_allowed()


def test_frozen_unchanged_tree_passes_and_reports_provenance(repo):
    sha = freeze(repo)
    info = guard.assert_test_allowed()
    assert info["tag"] == guard.TAG and info["commit"] == sha and len(sha) == 40
    assert info["plan_sha256"] == hashlib.sha256(PLAN.encode()).hexdigest()
    assert info["matched_tag"] == guard.TAG and info["matched_commit"] == sha


def test_changes_outside_the_guarded_paths_do_not_matter(repo):
    freeze(repo)
    (repo / "results").mkdir()
    (repo / "results" / "predictions.csv").write_text("a,b\n")  # untracked, not guarded
    (repo / "decisions.tsv").write_text((repo / "decisions.tsv").read_text() + "more\n")
    (repo / "src" / "__pycache__").mkdir()
    (repo / "src" / "__pycache__" / "model.pyc").write_bytes(b"\0")  # ignored by git
    assert guard.assert_test_allowed()["matched_tag"] == guard.TAG


# --------------------------------------------------------------- the working tree vs the tag

def test_plan_edited_after_the_tag_raises(repo):
    freeze(repo)
    (repo / "ANALYSIS_PLAN.md").write_text("plan v1 edited\n")
    with pytest.raises(guard.GuardError, match="ANALYSIS_PLAN.md"):
        guard.assert_test_allowed()


def test_source_file_edited_after_the_tag_raises(repo):
    freeze(repo)
    (repo / "src" / "model.py").write_text("X = 2\n")
    with pytest.raises(guard.GuardError, match="src/model.py"):
        guard.assert_test_allowed()


def test_config_edited_after_the_tag_raises(repo):
    freeze(repo)
    (repo / "configs" / "phase4.yaml").write_text("a: 2\n")
    with pytest.raises(guard.GuardError, match="configs/phase4.yaml"):
        guard.assert_test_allowed()


def test_untracked_file_in_src_raises(repo):
    freeze(repo)
    (repo / "src" / "extra.py").write_text("Y = 1\n")
    with pytest.raises(guard.GuardError, match=r"src/extra.py \(untracked\)"):
        guard.assert_test_allowed()


def test_untracked_file_in_configs_raises(repo):
    freeze(repo)
    (repo / "configs" / "extra.yaml").write_text("b: 1\n")
    with pytest.raises(guard.GuardError, match="untracked"):
        guard.assert_test_allowed()


def test_staged_new_file_in_src_raises(repo):
    freeze(repo)
    (repo / "src" / "extra.py").write_text("Y = 1\n")
    git(repo, "add", "src/extra.py")
    with pytest.raises(guard.GuardError, match="extra.py"):
        guard.assert_test_allowed()


def test_deleted_source_file_raises(repo):
    freeze(repo)
    (repo / "src" / "model.py").unlink()
    with pytest.raises(guard.GuardError, match="src/model.py"):
        guard.assert_test_allowed()


def test_processed_data_edited_after_the_tag_raises(repo):
    freeze(repo)
    (repo / "data" / "processed" / "t1_table.csv").write_text("pond,w\n1,3.0\n")
    with pytest.raises(guard.GuardError, match="data/processed/t1_table.csv"):
        guard.assert_test_allowed()


def test_untracked_file_in_processed_data_raises(repo):
    freeze(repo)
    (repo / "data" / "processed" / "t2_table.csv").write_text("pond,y\n1,0.1\n")
    with pytest.raises(guard.GuardError, match=r"data/processed/t2_table.csv \(untracked\)"):
        guard.assert_test_allowed()


def test_deleted_processed_data_file_raises(repo):
    freeze(repo)
    (repo / "data" / "processed" / "t1_table.csv").unlink()
    with pytest.raises(guard.GuardError, match="data/processed/t1_table.csv"):
        guard.assert_test_allowed()


def test_lock_file_edited_after_the_tag_raises(repo):
    freeze(repo)
    (repo / "uv.lock").write_text("version = 2\n")
    with pytest.raises(guard.GuardError, match="uv.lock"):
        guard.assert_test_allowed()


def test_lock_file_added_after_the_tag_raises(repo):
    git(repo, "rm", "-q", "uv.lock")
    git(repo, "commit", "-q", "-m", "no lock yet")
    freeze(repo)
    (repo / "uv.lock").write_text("version = 1\n")
    with pytest.raises(guard.GuardError, match=r"uv.lock \(untracked\)"):
        guard.assert_test_allowed()


def test_other_data_directories_are_not_guarded(repo):
    freeze(repo)
    (repo / "data" / "interim").mkdir()
    (repo / "data" / "interim" / "qc_rows.csv").write_text("a\n")
    assert guard.assert_test_allowed()["matched_tag"] == guard.TAG


def test_the_fix_tag_must_also_match_the_processed_data(repo):
    freeze(repo)
    (repo / "data" / "processed" / "t1_table.csv").write_text("pond,w\n1,3.0\n")
    commit_all(repo, "data fix")
    git(repo, "tag", "-a", "analysis-fix-1", "-m", "fix 1")
    assert guard.assert_test_allowed()["matched_tag"] == "analysis-fix-1"
    (repo / "uv.lock").write_text("version = 3\n")  # changed again after the fix tag
    with pytest.raises(guard.GuardError, match="uv.lock"):
        guard.assert_test_allowed()


def test_missing_plan_file_raises(repo):
    freeze(repo)
    (repo / "ANALYSIS_PLAN.md").unlink()
    with pytest.raises(guard.GuardError, match="missing"):
        guard.assert_test_allowed()


def test_restoring_the_files_byte_for_byte_passes_again(repo):
    freeze(repo)
    path = repo / "src" / "model.py"
    path.write_text("X = 2\n")
    with pytest.raises(guard.GuardError):
        guard.assert_test_allowed()
    path.write_text("X = 1\n")
    assert guard.assert_test_allowed()["tag"] == guard.TAG


def test_the_tree_is_compared_with_the_tag_not_with_head(repo):
    freeze(repo)
    (repo / "src" / "model.py").write_text("X = 2\n")
    commit_all(repo, "edit after the freeze")  # HEAD now equals the working tree
    with pytest.raises(guard.GuardError, match="src/model.py"):
        guard.assert_test_allowed()


def test_a_trailing_newline_counts_as_a_change(repo):
    freeze(repo)
    (repo / "ANALYSIS_PLAN.md").write_text(PLAN + "\n")
    with pytest.raises(guard.GuardError):
        guard.assert_test_allowed()


# ----------------------------------------------------------------------------- fix tags

def test_a_fix_tag_with_the_frozen_plan_allows_the_fixed_tree(repo):
    freeze(repo)
    (repo / "src" / "model.py").write_text("X = 2\n")
    commit_all(repo, "fix")
    git(repo, "tag", "-a", "analysis-fix-1", "-m", "fix 1")
    info = guard.assert_test_allowed()
    assert info["tag"] == guard.TAG and info["matched_tag"] == "analysis-fix-1"
    assert info["matched_commit"] == git(repo, "rev-parse", "analysis-fix-1^{commit}")


def test_a_tree_changed_after_the_fix_tag_raises(repo):
    freeze(repo)
    (repo / "src" / "model.py").write_text("X = 2\n")
    commit_all(repo, "fix")
    git(repo, "tag", "-a", "analysis-fix-1", "-m", "fix 1")
    (repo / "src" / "model.py").write_text("X = 3\n")
    with pytest.raises(guard.GuardError, match="no analysis-fix-N tag"):
        guard.assert_test_allowed()


def test_a_fix_tag_whose_plan_differs_from_the_frozen_plan_does_not_count(repo):
    freeze(repo)
    (repo / "ANALYSIS_PLAN.md").write_text("plan v2\n")
    commit_all(repo, "plan changed")
    git(repo, "tag", "-a", "analysis-fix-1", "-m", "plan changed under a fix tag")
    with pytest.raises(guard.GuardError):
        guard.assert_test_allowed()


def test_a_tag_that_is_not_named_analysis_fix_n_does_not_count(repo):
    freeze(repo)
    (repo / "src" / "model.py").write_text("X = 2\n")
    commit_all(repo, "fix")
    git(repo, "tag", "-a", "analysis-fix-final", "-m", "wrong name")
    git(repo, "tag", "-a", "my-analysis-fix-1", "-m", "wrong name")
    with pytest.raises(guard.GuardError):
        guard.assert_test_allowed()


def test_an_untracked_file_defeats_the_fix_tag_too(repo):
    freeze(repo)
    (repo / "src" / "model.py").write_text("X = 2\n")
    commit_all(repo, "fix")
    git(repo, "tag", "-a", "analysis-fix-1", "-m", "fix 1")
    (repo / "src" / "extra.py").write_text("Z = 1\n")
    with pytest.raises(guard.GuardError):
        guard.assert_test_allowed()


def test_the_second_fix_tag_is_found_after_the_first(repo):
    freeze(repo)
    for n in (1, 2):
        (repo / "src" / "model.py").write_text(f"X = {n + 1}\n")
        commit_all(repo, f"fix {n}")
        git(repo, "tag", "-a", f"analysis-fix-{n}", "-m", f"fix {n}")
    assert guard.assert_test_allowed()["matched_tag"] == "analysis-fix-2"


# ------------------------------------------------------------------------------- git itself

def test_directory_that_is_not_a_repository_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    with pytest.raises(guard.GuardError):
        guard.assert_test_allowed()


def test_git_environment_variables_cannot_redirect_the_check(repo, tmp_path_factory, monkeypatch):
    other = tmp_path_factory.mktemp("other")
    git(other, "init", "-q")
    (other / "ANALYSIS_PLAN.md").write_text(PLAN)
    commit_all(other, "plan")
    git(other, "tag", "-a", guard.TAG, "-m", "freeze")  # the tag exists only in the other repository
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    with pytest.raises(guard.GuardError, match="does not exist"):
        guard.assert_test_allowed()


# ------------------------------------------------------------ files that must be committed

SELECTION = "results/selection/primary_T1.json"


@pytest.fixture
def selection(repo):
    (repo / "results" / "selection").mkdir(parents=True)
    (repo / SELECTION).write_text('{"baseline": "gc"}\n')
    return repo


def test_a_committed_unmodified_file_passes(selection):
    commit_all(selection, "selection")
    guard.assert_files_committed([SELECTION])


def test_an_untracked_file_is_refused(selection):
    with pytest.raises(guard.GuardError, match="not tracked"):
        guard.assert_files_committed([SELECTION])


def test_a_staged_but_uncommitted_file_is_refused(selection):
    git(selection, "add", SELECTION)
    with pytest.raises(guard.GuardError, match="not in the last commit"):
        guard.assert_files_committed([SELECTION])


def test_an_edited_file_is_refused(selection):
    commit_all(selection, "selection")
    (selection / SELECTION).write_text('{"baseline": "ridge_time"}\n')
    with pytest.raises(guard.GuardError, match="differs from the last commit"):
        guard.assert_files_committed([SELECTION])


def test_a_staged_edit_is_refused(selection):
    commit_all(selection, "selection")
    (selection / SELECTION).write_text('{"baseline": "ridge_time"}\n')
    git(selection, "add", SELECTION)
    with pytest.raises(guard.GuardError, match="differs from the last commit"):
        guard.assert_files_committed([SELECTION])


def test_a_deleted_file_is_refused(selection):
    commit_all(selection, "selection")
    (selection / SELECTION).unlink()
    with pytest.raises(guard.GuardError, match="differs from the last commit"):
        guard.assert_files_committed([SELECTION])


def test_every_listed_file_is_checked(selection):
    commit_all(selection, "selection")
    with pytest.raises(guard.GuardError, match="primary_T2.json"):
        guard.assert_files_committed([SELECTION, "results/selection/primary_T2.json"])
