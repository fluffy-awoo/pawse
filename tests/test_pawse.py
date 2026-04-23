from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from pawse import Codec
from pawse.core import FORWARD_TABLE, REVERSE_TABLE, _farnsworth_scale, _wpm_to_dps


def mc(**kw) -> Codec:
    kw.setdefault("farnsworth_wpm", None)
    return Codec(**kw)


class TestTables:
    def test_reverse_is_inverse_of_forward(self):
        for char, code in FORWARD_TABLE.items():
            assert REVERSE_TABLE[code] == char

    def test_no_duplicate_codes(self):
        codes = list(FORWARD_TABLE.values())
        assert len(codes) == len(set(codes))


class TestTimingHelpers:
    def test_wpm_to_dps_paris(self):
        assert _wpm_to_dps(1.0) == pytest.approx(50 / 60)
        assert _wpm_to_dps(12.0) == pytest.approx(12 * 50 / 60)

    def test_farnsworth_no_slowdown(self):
        assert _farnsworth_scale(20.0, 20.0) == pytest.approx(1.0)
        assert _farnsworth_scale(20.0, 25.0) == pytest.approx(1.0)

    def test_farnsworth_slowdown(self):
        assert _farnsworth_scale(25.0, 10.0) > 1.0

    def test_farnsworth_none(self):
        assert _farnsworth_scale(25.0, None) == pytest.approx(1.0)


class TestToMorse:
    def test_single_letters(self):
        m = mc()
        assert m.to_morse("E") == "."
        assert m.to_morse("T") == "-"
        assert m.to_morse("A") == ".-"
        assert m.to_morse("S") == "..."
        assert m.to_morse("O") == "---"

    def test_case_insensitive(self):
        m = mc()
        assert m.to_morse("sos") == m.to_morse("SOS")

    def test_word_space(self):
        m = mc()
        assert "  " in m.to_morse("E T")

    def test_multi_word(self):
        m = mc()
        assert len(m.to_morse("HI HI").split("  ")) == 2

    def test_unknown_chars_skipped(self):
        m = mc()
        assert m.to_morse("~") == ""
        assert m.to_morse("A~B") == m.to_morse("AB")

    def test_empty_string(self):
        assert mc().to_morse("") == ""

    def test_digits(self):
        m = mc()
        assert m.to_morse("0") == "-----"
        assert m.to_morse("9") == "----."

    def test_punctuation(self):
        m = mc()
        assert m.to_morse(".") == ".-.-.-"
        assert m.to_morse("?") == "..--.."


class TestFromMorse:
    def test_single_letters(self):
        m = mc()
        assert m.from_morse(".") == "E"
        assert m.from_morse("-") == "T"
        assert m.from_morse(".-") == "A"

    def test_word_gap(self):
        assert mc().from_morse(".  -") == "E T"

    def test_unknown_symbol_replaced_with_question_mark(self):
        assert mc().from_morse("......") == "?"

    def test_empty(self):
        assert mc().from_morse("") == ""

    def test_roundtrip(self):
        m = mc()
        for text in ("SOS", "HELLO", "CQ CQ", "0 1 2"):
            assert m.from_morse(m.to_morse(text)) == text


class TestAudioGeneration:
    def test_to_audio_returns_float32(self):
        assert mc().to_audio("E").dtype == np.float32

    def test_to_audio_not_empty(self):
        assert mc().to_audio("SOS").size > 0

    def test_to_audio_empty_text(self):
        assert mc().to_audio("").size == 0

    def test_volume_respected(self):
        assert np.max(np.abs(mc(volume=1.0).to_audio("E"))) > np.max(
            np.abs(mc(volume=0.5).to_audio("E"))
        )

    def test_audio_within_range(self):
        assert np.max(np.abs(mc(volume=1.0).to_audio("SOS"))) <= 1.0 + 1e-4

    def test_dash_longer_than_dot(self):
        m = mc()
        assert m._morse_to_bool_arr("-").sum() > m._morse_to_bool_arr(".").sum()


class TestWavRoundTrip:
    def _roundtrip(self, text: str, **kw) -> str:
        kw.setdefault("farnsworth_wpm", None)
        m = Codec(**kw)
        with tempfile.TemporaryDirectory() as td:
            path = m.to_wav(pathlib.Path(td) / "test", text)
            return m.from_wav(path)

    def test_sos(self):
        assert self._roundtrip("SOS") == "SOS"

    def test_single_letter_e(self):
        assert self._roundtrip("E") == "E"

    def test_single_letter_t(self):
        assert self._roundtrip("T") == "T"

    def test_hello(self):
        assert self._roundtrip("HELLO") == "HELLO"

    def test_two_words(self):
        assert self._roundtrip("HI HI") == "HI HI"

    @pytest.mark.parametrize("text", ["MOM", "TOO", "OM", "OOO"])
    def test_all_dash_letters(self, text):
        assert self._roundtrip(text) == text

    @pytest.mark.parametrize("text", ["EEE", "IE", "EIE"])
    def test_all_dot_letters(self, text):
        assert self._roundtrip(text) == text

    def test_pangram(self):
        assert self._roundtrip("THE QUICK BROWN FOX") == "THE QUICK BROWN FOX"

    def test_digits_roundtrip(self):
        assert self._roundtrip("73 88") == "73 88"

    @pytest.mark.parametrize("wpm", [15.0, 20.0, 30.0])
    def test_roundtrip_various_wpm(self, wpm):
        assert self._roundtrip("CQ DE K1ABC", wpm=wpm) == "CQ DE K1ABC"

    def test_roundtrip_with_farnsworth(self):
        m = Codec(wpm=25.0, farnsworth_wpm=10.0)
        with tempfile.TemporaryDirectory() as td:
            path = m.to_wav(pathlib.Path(td) / "f", "HELLO WORLD")
            assert m.from_wav(path) == "HELLO WORLD"

    def test_to_wav_returns_path_with_wav_suffix(self):
        m = mc()
        with tempfile.TemporaryDirectory() as td:
            p = m.to_wav(pathlib.Path(td) / "out", "E")
            assert p.suffix == ".wav"
            assert p.exists()

    def test_from_wav_silent_file(self):
        import scipy.io.wavfile as wavfile

        m = mc()
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "silent.wav"
            wavfile.write(path, 8000, np.zeros(8000, dtype=np.int16))
            assert m.from_wav(path) == ""


class TestEnvFollow:
    def test_flat_signal(self):
        m = mc()
        sig = np.ones(100, dtype=np.float32)
        env = m._env_follow(sig, 10)
        assert env.shape == sig.shape
        np.testing.assert_allclose(env[5:-5], 1.0, atol=1e-5)

    def test_zero_signal(self):
        env = mc()._env_follow(np.zeros(100, dtype=np.float32), 10)
        np.testing.assert_allclose(env, 0.0)


class TestSelfConsistency:
    """Regression benchmarks: encode then decode N random messages and
    assert exact-match accuracy stays above a floor per config. Seeds are
    fixed so this is deterministic across CI runs."""

    N = 200

    @staticmethod
    def _messages(seed: int) -> list[str]:
        import random

        rng = random.Random(seed)
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        msgs = []
        for _ in range(TestSelfConsistency.N):
            words = [
                "".join(rng.choice(alphabet) for _ in range(rng.randint(1, 6)))
                for _ in range(rng.randint(1, 5))
            ]
            msgs.append(" ".join(words))
        return msgs

    @staticmethod
    def _accuracy(codec: Codec, msgs: list[str], noise_db: float | None = None, seed: int = 0) -> float:
        rng = np.random.default_rng(seed)
        ok = 0
        for text in msgs:
            audio = codec.to_audio(text)
            if noise_db is not None and audio.size:
                rms = float(np.sqrt(np.mean(audio**2))) or 1e-9
                noise = rng.standard_normal(audio.size).astype(np.float32)
                audio = audio + noise * rms * 10 ** (-noise_db / 20)
            if codec.from_audio(audio, codec.sample_rate) == text:
                ok += 1
        return ok / len(msgs)

    @pytest.mark.parametrize(
        "name,codec_kwargs,noise_db,floor",
        [
            ("default",      dict(wpm=25),          None, 0.98),
            ("slow",         dict(wpm=15),          None, 0.98),
            ("fast",         dict(wpm=35),          None, 0.98),
            ("farnsworth",   dict(wpm=25, farnsworth_wpm=10),   None, 0.98),
            ("snr_20db",     dict(wpm=25),          20.0, 0.98),
            ("snr_10db",     dict(wpm=25),          10.0, 0.95),
        ],
    )
    def test_accuracy_floor(self, name, codec_kwargs, noise_db, floor):
        codec_kwargs.setdefault("farnsworth_wpm", None)
        acc = self._accuracy(
            Codec(**codec_kwargs), self._messages(seed=42), noise_db=noise_db, seed=1
        )
        assert acc >= floor, f"{name}: {acc:.1%} < floor {floor:.0%}"


class TestRunLengths:
    def test_basic(self):
        mask = np.array([True, True, False, False, False, True])
        assert Codec._run_lengths(mask) == [(True, 2), (False, 3), (True, 1)]

    def test_single(self):
        assert Codec._run_lengths(np.array([False])) == [(False, 1)]

    def test_empty(self):
        assert Codec._run_lengths(np.array([], dtype=bool)) == []
