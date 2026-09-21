from pathlib import Path

import pytest

from amuse.datamodel import Particles
from amuse.units import units

from dcaf.dcaf import DcafSystem


class DummyFramework:
    def __init__(self):
        stars = Particles(2)
        stars.mass = [1.0, 1.0] | units.MSun
        stars.x = [0.0, 1.0] | units.parsec
        stars.y = [0.0, 0.0] | units.parsec
        stars.z = [0.0, 0.0] | units.parsec
        self.target_stars = stars
        self.background_gas = None


def make_snapshot(folder: Path, index: int = 0):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"stars_{index:03d}.amuse").write_text("")


def build_system(output_folder: Path, resume: bool):
    return DcafSystem(
        framework=DummyFramework(),
        output_folder=str(output_folder),
        log_level="info",
        resume=resume,
    )


def test_resume_uses_latest_snapshot_folder_and_creates_next_segment(tmp_path):
    base = tmp_path / "dcaf_output"
    make_snapshot(base)

    system = build_system(base, resume=True)

    assert system.resume_source_folder == str(base)
    assert system.output_folder == str(tmp_path / "dcaf_output_1")


def test_resume_reuses_latest_empty_folder_and_archives_log(tmp_path):
    base = tmp_path / "dcaf_output"
    latest = tmp_path / "dcaf_output_1"
    make_snapshot(base)
    latest.mkdir()
    (latest / "dcaf.log").write_text("failed attempt")
    (latest / "dcaf.log.failed").write_text("older failure")

    system = build_system(base, resume=True)

    assert system.resume_source_folder == str(base)
    assert system.output_folder == str(latest)
    assert (latest / "dcaf.log.failed").read_text() == "failed attempt"
    assert (latest / "dcaf.log").exists()


def test_resume_fails_with_two_consecutive_empty_folders(tmp_path):
    base = tmp_path / "dcaf_output"
    empty_1 = tmp_path / "dcaf_output_1"
    base.mkdir()
    empty_1.mkdir()

    with pytest.raises(FileNotFoundError, match="two consecutive empty output folders"):
        build_system(base, resume=True)


def test_resume_fails_with_non_continuous_output_folders(tmp_path):
    base = tmp_path / "dcaf_output"
    make_snapshot(base)
    (tmp_path / "dcaf_output_2").mkdir()

    with pytest.raises(FileNotFoundError, match="Non-continuous output folders found"):
        build_system(base, resume=True)
