"""Doctor row for the managed local engines (Desktop-oriented copy)."""

from __future__ import annotations

import yaml


def _home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr("cli._hermes_home", home)
    import hermes_constants

    hermes_constants._default_hermes_root_memo = None
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: home)
    (home / "config.yaml").write_text(
        yaml.dump({"local_runtime": {"enabled": True, "engine": "vllm"}, "model": {}}),
        encoding="utf-8")
    return home


def test_doctor_local_row_reports_engine(tmp_path, monkeypatch, capsys):
    _home(tmp_path, monkeypatch)
    from hermes_cli.doctor_local import _check_managed_local_engine

    finding = _check_managed_local_engine(False)
    out = capsys.readouterr().out
    assert "engine: vllm" in out
    assert finding.issues  # venv missing while engine is vllm is the install contract
    assert any("Local Models" in issue for issue in finding.issues)
    assert not any("hermes local" in issue for issue in finding.issues)
