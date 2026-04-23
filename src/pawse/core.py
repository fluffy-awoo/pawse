from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass

import numpy as np
import scipy.io.wavfile as wavfile

try:
    import sounddevice as sd  # type: ignore
except (
    ModuleNotFoundError,
    OSError,
):  # sound playback is optional; OSError when PortAudio missing
    sd = None  # pragma: no cover


DOT: str = "."
DASH: str = "-"
DASH_WIDTH = 3  # dash is 3 dots
CHAR_SPACE = 3  # dots between letters (PARIS timing)
WORD_SPACE = 7  # dots between words

FORWARD_TABLE: dict[str, str] = {
    "A": ".-",
    "B": "-...",
    "C": "-.-.",
    "D": "-..",
    "E": ".",
    "F": "..-.",
    "G": "--.",
    "H": "....",
    "I": "..",
    "J": ".---",
    "K": "-.-",
    "L": ".-..",
    "M": "--",
    "N": "-.",
    "O": "---",
    "P": ".--.",
    "Q": "--.-",
    "R": ".-.",
    "S": "...",
    "T": "-",
    "U": "..-",
    "V": "...-",
    "W": ".--",
    "X": "-..-",
    "Y": "-.--",
    "Z": "--..",
    "0": "-----",
    "1": ".----",
    "2": "..---",
    "3": "...--",
    "4": "....-",
    "5": ".....",
    "6": "-....",
    "7": "--...",
    "8": "---..",
    "9": "----.",
    ".": ".-.-.-",
    ",": "--..--",
    "?": "..--..",
    "'": ".----.",
    "!": "-.-.--",
    "/": "-..-.",
    "(": "-.--.",
    ")": "-.--.-",
    "&": ".-...",
    ":": "---...",
    ";": "-.-.-.",
    "=": "-...-",
    "+": ".-.-.",
    "-": "-....-",
    "_": "..--.-",
    '"': ".-..-.",
    "$": "...-..-",
    "@": ".--.-.",
}
REVERSE_TABLE: dict[str, str] = {v: k for k, v in FORWARD_TABLE.items()}


def _wpm_to_dps(wpm: float) -> float:
    """Words-per-minute to dots-per-second (PARIS word = 50 dots)"""
    return wpm * 50.0 / 60.0


def _farnsworth_scale(wpm: float, fs: float | None) -> float:
    """Return > 1 scaling factor if overall speed fs is slower than wpm"""
    if fs is None:
        return 1.0

    slow_word_interval = 1.0 / fs
    standard_word_interval = 1.0 / wpm
    extra_space = slow_word_interval - standard_word_interval

    if extra_space <= 0:
        return 1.0

    standard_word_dots = 50.0  # PARIS definition
    extra_dots = (extra_space / standard_word_interval) * standard_word_dots
    standard_space_dots = 4 * CHAR_SPACE + WORD_SPACE

    return 1.0 + extra_dots / standard_space_dots


@dataclass(slots=True)
class Codec:
    """Encode text to decode WAV for Morse code"""

    wpm: float = 25.0
    frequency: float = 750.0
    farnsworth_wpm: float | None = 10.0
    sample_rate: int = 8_000
    volume: float = 0.9
    click_smooth: int = 2

    def to_morse(self, text: str) -> str:
        """ASCII to Morse symbols with single/double spaces for gaps"""
        return " ".join(
            self._letter_to_morse(ch)
            for ch in text.upper()
            if self._letter_to_morse(ch)
        )

    def _letter_to_morse(self, ch: str) -> str:
        return " " if ch.isspace() else FORWARD_TABLE.get(ch, "")

    def _morse_to_bool_arr(self, code: str) -> np.ndarray:
        dps = _wpm_to_dps(self.wpm)
        base = self.sample_rate / dps
        sp_dot = int(round(base))
        sp_dash = int(round(base * DASH_WIDTH))
        sp_gap_elem = int(round(base))
        scale = _farnsworth_scale(self.wpm, self.farnsworth_wpm)
        sp_gap_char = int(round(base * CHAR_SPACE * scale))
        sp_gap_word = int(round(base * WORD_SPACE * scale))

        dot_arr = np.ones(sp_dot, dtype=np.bool_)
        dash_arr = np.ones(sp_dash, dtype=np.bool_)
        gap_elem = np.zeros(sp_gap_elem, dtype=np.bool_)
        gap_char = np.zeros(sp_gap_char, dtype=np.bool_)
        gap_word = np.zeros(sp_gap_word, dtype=np.bool_)

        out: list[np.ndarray] = []
        prev_elem = prev_space = False

        for symbol in code:
            if symbol in (DOT, DASH) and prev_elem:
                out.append(gap_elem)
            if symbol == DOT:
                out.append(dot_arr)
                prev_elem, prev_space = True, False
            elif symbol == DASH:
                out.append(dash_arr)
                prev_elem, prev_space = True, False
            else:  # space
                if prev_space:
                    out[-1] = gap_word  # upgrade previous char gap to word gap
                else:
                    out.append(gap_char)
                prev_elem, prev_space = False, True

        return np.concatenate(out) if out else np.zeros(0, dtype=np.bool_)

    def _bool_arr_to_tone(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return np.array([], dtype=np.float32)

        wt_len = int(self.click_smooth * self.sample_rate / self.frequency)

        if wt_len % 2 == 0:
            wt_len += 1

        weights = np.concatenate(
            (np.arange(1, wt_len // 2 + 1), np.arange(wt_len // 2 + 1, 0, -1))
        )
        weights = weights / weights.sum()

        pad = int(self.sample_rate * 0.5) + (wt_len - 1) // 2  # 0.5 s leading pad
        padded = np.concatenate(
            (np.zeros(pad, dtype=np.bool_), mask, np.zeros(pad, dtype=np.bool_))
        )
        smooth = (
            padded.astype(np.float32)
            if self.click_smooth <= 0
            else np.correlate(padded.astype(np.float32), weights, "valid")
        )

        t = np.arange(smooth.size, dtype=np.float32)
        tone = np.sin(t * (self.frequency * 2 * math.pi / self.sample_rate))

        return (tone * smooth * self.volume).astype(np.float32)

    def to_audio(self, text: str) -> np.ndarray:
        return self._bool_arr_to_tone(self._morse_to_bool_arr(self.to_morse(text)))

    def to_wav(self, text: str, path: str | pathlib.Path) -> pathlib.Path:
        path = pathlib.Path(path).with_suffix(".wav")
        wavfile.write(
            path, self.sample_rate, (self.to_audio(text) * 32767).astype(np.int16)
        )

        return path

    def play(self, text: str) -> None:
        if sd is None:
            raise RuntimeError("sounddevice not installed")

        sd.play(self.to_audio(text).astype(np.float32), self.sample_rate)

    def _env_follow(self, sig: np.ndarray, win: int) -> np.ndarray:
        return np.convolve(np.abs(sig), np.ones(win, dtype=np.float32) / win, "same")

    @staticmethod
    def _run_lengths(mask: np.ndarray) -> list[tuple[bool, int]]:
        runs: list[tuple[bool, int]] = []
        if mask.size == 0:
            return runs
        cur_val, cur_len = bool(mask[0]), 1

        for val in mask[1:]:
            val_b = bool(val)
            if val_b == cur_val:
                cur_len += 1
            else:
                runs.append((cur_val, cur_len))
                cur_val, cur_len = val_b, 1
        runs.append((cur_val, cur_len))

        return runs

    def from_audio(self, audio: np.ndarray, sample_rate: int) -> str:
        """Decode audio ndarray (mono or stereo) to plaintext"""
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0

        if peak == 0.0:
            return ""

        audio /= peak

        win = max(1, int(sample_rate * 0.005))  # ~5ms
        env = self._env_follow(audio, win)

        # If the envelope is weirdly flat, bail early
        if not np.isfinite(env).all() or env.size == 0:
            return ""

        # Hysteresis: rise on hi, fall on lo
        thr_hi = 0.5 * float(np.max(env))
        if thr_hi <= 0:
            return ""
        thr_lo = 0.6 * thr_hi

        mask = np.zeros_like(env, dtype=np.bool_)
        state = False
        for i, v in enumerate(env):
            if not state and v >= thr_hi:
                state = True
            elif state and v <= thr_lo:
                state = False
            mask[i] = state

        if not mask.any():
            return ""

        runs = self._run_lengths(mask)
        if not runs:
            return ""

        # Drop extremely short leading/trailing runs (edge artifacts)
        if runs and runs[0][1] < 3:
            runs = runs[1:]
        if runs and runs[-1][1] < 3:
            runs = runs[:-1]
        if not runs:
            return ""

        while runs and not runs[0][0]:
            runs = runs[1:]
        while runs and not runs[-1][0]:
            runs = runs[:-1]
        if not runs:
            return ""

        on = np.array([L for is_tone, L in runs if is_tone], dtype=float)
        off = np.array([L for is_tone, L in runs if not is_tone], dtype=float)
        if on.size == 0:
            return ""

        # Estimate dot length. Sort tone lengths and find the largest
        # consecutive ratio jump — that jump is the dot/dash cluster boundary
        # (theoretical 3:1). Robust to class imbalance (e.g. 'Q' = --.- has
        # only one dot among three dashes, where percentile estimators drift).
        # If no clear jump exists the signal is unimodal: fall back to the
        # expected dot length from self.wpm to decide all-dots vs all-dashes.
        sorted_on = np.sort(on)
        bimodal_idx = -1
        if sorted_on.size >= 2:
            on_ratios = sorted_on[1:] / sorted_on[:-1]
            j = int(np.argmax(on_ratios))
            if on_ratios[j] >= 1.6:
                bimodal_idx = j
        if bimodal_idx >= 0:
            dot_len = float(sorted_on[bimodal_idx])
        else:
            expected_dot = self.sample_rate / _wpm_to_dps(self.wpm)
            median_on = float(np.median(sorted_on))
            if abs(median_on - expected_dot) <= abs(median_on - 3 * expected_dot):
                dot_len = median_on
            else:
                dot_len = median_on / 3.0

        # Intra-element cutoff: short "off" inside a character (between dot/dash)
        intra_elem_cut = 1.5 * dot_len

        # Candidate letter/word gaps are off-runs >= intra-element cutoff.
        # Theoretical ratio is 7:3 (word:letter) = 2.33. Sort and find the
        # largest consecutive ratio jump; if it's big enough, that jump is
        # the letter/word cluster boundary. Robust to a single word gap
        # among many letter gaps, and to median sitting between clusters.
        letterish = np.sort(off[off >= intra_elem_cut])
        if letterish.size == 0:
            char_word_cut = float("inf")
        elif letterish.size == 1:
            # Single gap — no ratio to compare. Treat as word gap if it's
            # clearly longer than a letter gap. Farnsworth scaling stretches
            # both letter (3-dot) and word (7-dot) gaps by the same factor,
            # so derive the 5-dot midpoint from the configured fs.
            scale = _farnsworth_scale(self.wpm, self.farnsworth_wpm)
            single_gap_cut = 5.0 * dot_len * scale
            char_word_cut = (
                single_gap_cut if letterish[0] >= single_gap_cut else float("inf")
            )
        else:
            ratios = letterish[1:] / letterish[:-1]
            i = int(np.argmax(ratios))
            if ratios[i] >= 1.6:
                char_word_cut = 0.5 * (letterish[i] + letterish[i + 1])
            else:
                char_word_cut = float("inf")

        # Tone classification margin: allow for slightly stretched dots
        dot_dash_cut = 1.8 * dot_len

        symbols: list[str] = []
        for is_tone, L in runs:
            if is_tone:
                symbols.append(DOT if L < dot_dash_cut else DASH)
            else:
                if L < intra_elem_cut:
                    # intra-element gap (between dot/dash within a letter) -> ignore
                    continue
                # Between letters or between words?
                symbols.append(" " if L < char_word_cut else "  ")

        code = "".join(symbols).strip()
        if not code:
            return ""
        return self._morse_to_text(code)

    def _morse_to_text(self, code: str) -> str:
        words: list[str] = []

        for word in code.split("  "):
            letters = [REVERSE_TABLE.get(sym, "?") for sym in word.split() if sym]
            words.append("".join(letters))

        return " ".join(words)

    def from_wav(self, path: str | pathlib.Path) -> str:
        sr, data = wavfile.read(path)

        return self.from_audio(data, sr)

    def from_morse(self, code: str) -> str:
        return self._morse_to_text(code)
