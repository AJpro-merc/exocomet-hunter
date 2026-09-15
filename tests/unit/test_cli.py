# Copyright (c) 2026 Atharva Joshi
# SPDX-License-Identifier: BSD-3-Clause

"""The console script must parse and answer for --help without touching MAST.

``exocomet`` is the first thing a user of the installed package runs, so these
tests exist to make sure it never regresses into the state it was found in:
declared in ``pyproject.toml`` but absent from the source tree. Nothing here
downloads anything -- the checks stop at argument parsing, and the one test that
calls into a command handler stubs the archive out.
"""

from __future__ import annotations

import pytest

from exocomet import cli


class TestHelp:
    def test_top_level_help_exits_zero(self, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--help"])
        assert exc.value.code == 0
        assert "run" in capsys.readouterr().out

    @pytest.mark.parametrize("command", ["run", "validate", "inject"])
    def test_subcommand_help_exits_zero(self, command, capsys) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main([command, "--help"])
        assert exc.value.code == 0
        assert capsys.readouterr().out

    def test_version_exits_zero(self) -> None:
        with pytest.raises(SystemExit) as exc:
            cli.main(["--version"])
        assert exc.value.code == 0

    def test_no_subcommand_prints_help_and_fails(self, capsys) -> None:
        assert cli.main([]) == 2
        assert "usage:" in capsys.readouterr().out


class TestParsing:
    def test_run_binds_target_mission_and_handler(self) -> None:
        args = cli.build_parser().parse_args(["run", "KIC 3542116", "--mission", "TESS"])
        assert args.target == "KIC 3542116"
        assert args.mission == "TESS"
        assert args.func is cli.cmd_run

    def test_run_defaults_to_kepler(self) -> None:
        args = cli.build_parser().parse_args(["run", "KIC 3542116"])
        assert args.mission == "Kepler"

    def test_run_rejects_an_unknown_mission(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run", "KIC 1", "--mission", "Hubble"])

    def test_run_requires_a_target(self) -> None:
        with pytest.raises(SystemExit):
            cli.build_parser().parse_args(["run"])

    def test_validate_defaults_to_the_repository_target_list(self) -> None:
        args = cli.build_parser().parse_args(["validate"])
        assert args.targets == cli.DEFAULT_VALIDATION_TARGETS
        assert args.func is cli.cmd_validate

    def test_inject_binds_n_and_handler(self) -> None:
        args = cli.build_parser().parse_args(["inject", "--n", "7"])
        assert args.n == 7
        assert args.func is cli.cmd_inject

    def test_inject_accepts_a_depth_sweep(self) -> None:
        args = cli.build_parser().parse_args(["inject", "--depths", "1e-4", "1e-3"])
        assert args.depths == [1.0e-4, 1.0e-3]


class TestFailureHandling:
    def test_a_failing_command_returns_one_rather_than_raising(self, capsys) -> None:
        """An unavailable target is a non-zero exit, not a traceback."""
        assert cli.main(["validate", "--targets", "no/such/file.yaml"]) == 1
        assert "error:" in capsys.readouterr().err

    def test_run_reports_an_unreachable_target_as_a_failure(self, monkeypatch, capsys) -> None:
        def refuse(*_args, **_kwargs):
            raise FileNotFoundError("no Kepler light curves found for KIC 1")

        monkeypatch.setattr(cli, "_prepare_light_curve", refuse)
        assert cli.main(["run", "KIC 1"]) == 1
        assert "no Kepler light curves" in capsys.readouterr().err


class TestInjectOffline:
    def test_inject_runs_end_to_end_on_a_synthetic_host(self, capsys) -> None:
        """The calibration path works with no --target, hence no network."""
        code = cli.main(["inject", "--n", "1", "--depths", "5e-3", "--n-points", "800"])
        assert code == 0
        out = capsys.readouterr().out
        assert "synthetic host" in out
        assert "detection floor" in out
