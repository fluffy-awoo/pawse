# CLI

```bash
pawse {encode,decode,to-wav,play} [options]
```

---

## encode

Print the Morse representation of a string.

```bash
pawse encode <text> [options]
```

```bash
$ pawse encode "SOS"
... --- ...

$ pawse encode "hello world"
.... . .-.. .-.. ---   .-- --- .-. .-.. -..
```

---

## decode

Decode a WAV file to text.

```bash
pawse decode <file> [options]
```

```bash
$ pawse decode recording.wav
SOS
```

The decoder auto-detects dot/dash timing from the audio. Pass `--wpm` if the
recording is at a known speed to improve accuracy.

---

## to-wav

Write text as a Morse audio WAV file. Appends `.wav` if the output path has no
suffix.

```bash
pawse to-wav <text> <output> [options]
```

```bash
$ pawse to-wav "CQ CQ DE K1ABC" cq
# writes cq.wav
```

---

## play

Play text as Morse audio. Requires `sounddevice`.

```bash
pawse play <text> [options]
```

---

## Options

These flags are accepted by all subcommands.

**`--wpm WPM`** (default: `25`)
Character speed in words per minute.

**`--frequency HZ`** (default: `750`)
Tone frequency in Hz.

**`--farnsworth-wpm WPM`** (default: off)
Overall speed in wpm. Must be slower than `--wpm`. Keeps characters at full
speed but stretches the gaps between them.

```bash
# Characters sent at 25 wpm, but overall pace is 10 wpm
$ pawse play "CQ CQ" --wpm 25 --farnsworth-wpm 10
```

**`--volume VOL`** (default: `0.9`)
Output volume, 0.0–1.0.
