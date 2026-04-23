# Codec

```python
from pawse import Codec
```

---

## Parameters

```python
Codec(
    wpm=25.0,
    frequency=750.0,
    farnsworth_wpm=None,
    sample_rate=8000,
    volume=0.9,
    click_smooth=2,
)
```

**`wpm`** (default: `25.0`)
Character speed in words per minute.

**`frequency`** (default: `750.0`)
Tone frequency in Hz.

**`farnsworth_wpm`** (default: `None`)
Overall speed in wpm. Must be slower than `wpm`. Keeps characters at full speed
but stretches the gaps between them.

**`sample_rate`** (default: `8000`)
Audio sample rate in Hz.

**`volume`** (default: `0.9`)
Output volume, 0.0–1.0.

**`click_smooth`** (default: `2`)
Smoothing strength applied to tone edges to reduce key clicks. Set to `0` to
disable.

---

## Methods

### `to_morse(text) -> str`

Convert text to a Morse symbol string. Letters are separated by a single space,
words by two spaces. Unknown characters are silently dropped.

```python
mc.to_morse("SOS")      # "... --- ..."
mc.to_morse("HI HI")    # ".... ..   .... .."
```

### `from_morse(code) -> str`

Convert a Morse symbol string back to text. Unknown symbols are replaced with
`?`.

```python
mc.from_morse("... --- ...")  # "SOS"
```

### `to_audio(text) -> np.ndarray`

Return a float32 NumPy array of the audio signal.

```python
audio = mc.to_audio("SOS")
```

### `to_wav(text, path) -> pathlib.Path`

Write audio to a WAV file. Appends `.wav` if the path has no suffix. Returns
the final path.

```python
path = mc.to_wav("SOS", "output")  # writes output.wav
```

### `from_wav(path) -> str`

Decode a WAV file to text.

```python
mc.from_wav("output.wav")  # "SOS"
```

### `from_audio(audio, sample_rate) -> str`

Decode a NumPy audio array to text. Accepts mono or stereo; stereo is averaged
to mono before decoding.

```python
mc.from_audio(audio, sample_rate=8000)  # "SOS"
```

### `play(text) -> None`

Play text as Morse audio. Requires `sounddevice`.
Raises `RuntimeError` if sounddevice is not installed.

```python
mc.play("CQ CQ")
```
