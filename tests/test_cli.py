from __future__ import annotations

import argparse
import pathlib
import sys
import tempfile

import pytest

from pawse.cli import _codec, main


def invoke(*args: str, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    monkeypatch.setattr(sys, "argv", ["pawse", *args])
    return main()


class TestEncode:
    def test_sos(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pawse", "encode", "SOS"])
        main()
        assert capsys.readouterr().out.strip() == "... --- ..."

    def test_case_insensitive(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pawse", "encode", "sos"])
        main()
        lower = capsys.readouterr().out.strip()
        monkeypatch.setattr(sys, "argv", ["pawse", "encode", "SOS"])
        main()
        upper = capsys.readouterr().out.strip()
        assert lower == upper

    def test_wpm_does_not_change_symbols(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pawse", "encode", "HELLO", "--wpm", "15"])
        main()
        out1 = capsys.readouterr().out.strip()
        monkeypatch.setattr(sys, "argv", ["pawse", "encode", "HELLO", "--wpm", "30"])
        main()
        out2 = capsys.readouterr().out.strip()
        assert out1 == out2

    def test_empty_text(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["pawse", "encode", ""])
        main()
        assert capsys.readouterr().out.strip() == ""


class TestDecode:
    def test_roundtrip(self, monkeypatch, capsys):
        with tempfile.TemporaryDirectory() as td:
            wav = pathlib.Path(td) / "out.wav"
            monkeypatch.setattr(sys, "argv", ["pawse", "to-wav", "SOS", str(wav)])
            main()
            monkeypatch.setattr(
                sys, "argv", ["pawse", "decode", str(wav.with_suffix(".wav"))]
            )
            main()
        assert capsys.readouterr().out.strip() == "SOS"

    def test_multi_word(self, monkeypatch, capsys):
        with tempfile.TemporaryDirectory() as td:
            wav = pathlib.Path(td) / "out.wav"
            monkeypatch.setattr(sys, "argv", ["pawse", "to-wav", "HI HI", str(wav)])
            main()
            monkeypatch.setattr(
                sys, "argv", ["pawse", "decode", str(wav.with_suffix(".wav"))]
            )
            main()
        assert capsys.readouterr().out.strip() == "HI HI"

    def test_custom_wpm(self, monkeypatch, capsys):
        with tempfile.TemporaryDirectory() as td:
            wav = pathlib.Path(td) / "out.wav"
            monkeypatch.setattr(
                sys, "argv", ["pawse", "to-wav", "CQ", str(wav), "--wpm", "15"]
            )
            main()
            monkeypatch.setattr(
                sys,
                "argv",
                ["pawse", "decode", str(wav.with_suffix(".wav")), "--wpm", "15"],
            )
            main()
        assert capsys.readouterr().out.strip() == "CQ"


class TestToWav:
    def test_creates_file(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "test"
            monkeypatch.setattr(sys, "argv", ["pawse", "to-wav", "SOS", str(out)])
            main()
            assert out.with_suffix(".wav").exists()

    def test_appends_wav_suffix(self, monkeypatch):
        with tempfile.TemporaryDirectory() as td:
            out = pathlib.Path(td) / "nosuffix"
            monkeypatch.setattr(sys, "argv", ["pawse", "to-wav", "E", str(out)])
            main()
            assert out.with_suffix(".wav").exists()


class TestPlay:
    def test_play_no_sounddevice(self, monkeypatch):
        monkeypatch.setattr("pawse.core.sd", None)
        monkeypatch.setattr(sys, "argv", ["pawse", "play", "SOS"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == "sounddevice not installed"


class TestCodecArgs:
    def test_defaults(self):
        args = argparse.Namespace(
            wpm=25.0, frequency=750.0, farnsworth_wpm=None, volume=0.9
        )
        c = _codec(args)
        assert c.wpm == 25.0
        assert c.frequency == 750.0
        assert c.farnsworth_wpm is None
        assert c.volume == 0.9

    def test_custom_args(self):
        args = argparse.Namespace(
            wpm=15.0, frequency=600.0, farnsworth_wpm=8.0, volume=0.5
        )
        c = _codec(args)
        assert c.wpm == 15.0
        assert c.frequency == 600.0
        assert c.farnsworth_wpm == 8.0
        assert c.volume == 0.5

    def test_missing_subcommand(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["pawse"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code != 0
