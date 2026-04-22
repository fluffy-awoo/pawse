from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

from pawse import Codec
from pawse.core import FORWARD_TABLE, REVERSE_TABLE, _farnsworth_scale, _wpm_to_dps


def mc(**kw) -> Codec:
    kw.setdefault("fs", None)
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


class TestEncode:
    def test_single_letters(self):
        m = mc()
        assert m.encode("E") == "."
        assert m.encode("T") == "-"
        assert m.encode("A") == ".-"
        assert m.encode("S") == "..."
        assert m.encode("O") == "---"

    def test_case_insensitive(self):
        m = mc()
        assert m.encode("sos") == m.encode("SOS")

    def test_word_space(self):
        m = mc()
        assert "  " in m.encode("E T")

    def test_multi_word(self):
        m = mc()
        assert len(m.encode("HI HI").split("  ")) == 2

    def test_unknown_chars_skipped(self):
        m = mc()
        assert m.encode("~") == ""
        assert m.encode("A~B") == m.encode("AB")

    def test_empty_string(self):
        assert mc().encode("") == ""

    def test_digits(self):
        m = mc()
        assert m.encode("0") == "-----"
        assert m.encode("9") == "----."

    def test_punctuation(self):
        m = mc()
        assert m.encode(".") == ".-.-.-"
        assert m.encode("?") == "..--.."


class TestDecode:
    def test_single_letters(self):
        m = mc()
        assert m.decode(".") == "E"
        assert m.decode("-") == "T"
        assert m.decode(".-") == "A"

    def test_word_gap(self):
        assert mc().decode(".  -") == "E T"

    def test_unknown_symbol_replaced_with_question_mark(self):
        assert mc().decode("......") == "?"

    def test_empty_decode(self):
        assert mc().decode("") == ""

    def test_roundtrip_encode_decode(self):
        m = mc()
        for text in ("SOS", "HELLO", "CQ CQ", "0 1 2"):
            assert m.decode(m.encode(text)) == text


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
        kw.setdefault("fs", None)
        m = Codec(**kw)
        with tempfile.TemporaryDirectory() as td:
            path = m.to_wav(pathlib.Path(td) / "test", text)
            return m.from_wav(path)

    def test_sos(self):
        assert self._roundtrip("SOS") == "SOS"

    def test_single_letter_e(self):
        assert self._roundtrip("E") == "E"

    @pytest.mark.xfail(
        reason="Single dash has no reference dot; decoder calibrates dot_len from the dash and misclassifies it.",
        strict=True,
    )
    def test_single_letter_t(self):
        assert self._roundtrip("T") == "T"

    def test_hello(self):
        assert self._roundtrip("HELLO") == "HELLO"

    def test_two_words(self):
        assert self._roundtrip("HI HI") == "HI HI"

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


class TestRunLengths:
    def test_basic(self):
        mask = np.array([True, True, False, False, False, True])
        assert Codec._run_lengths(mask) == [(True, 2), (False, 3), (True, 1)]

    def test_single(self):
        assert Codec._run_lengths(np.array([False])) == [(False, 1)]

    def test_empty(self):
        assert Codec._run_lengths(np.array([], dtype=bool)) == []
