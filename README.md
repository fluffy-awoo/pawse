# pawse

> **A minimal pure-Python Morse-code encoder/decoder**
> MIT-licensed.

---

## Features

* **Encode** ASCII text to CW audio (NumPy buffer or WAV file).
* **Decode** CW audio (WAV or NumPy array) to uppercase text.
* Optional real-time playback with `sounddevice`.

---

## Requirements

|   Purpose    |      Package      |         Version        |
|--------------|-------------------|------------------------|
|   **Core**   | `numpy`           | ≥ 2.0.0                |
|              | `scipy`           | ≥ 1.10                 |
| **Optional** | `sounddevice`     | ≥ 0.4 &nbsp;*(audio playback)* |

Python **3.9+** required.

---

## Installation

```bash
# core only
pip install git+https://github.com/fluffy-awoo/pawse.git

# with audio playback
pip install "git+https://github.com/fluffy-awoo/pawse.git[audio]"
```

---

## Quick Start

```py
from pymorse import MorseCode

mc = MorseCode(wpm=20, hz=750)

# text to WAV
mc.to_wav("output.wav", "SOS")

# WAV to text
print(mc.from_wav("output.wav"))
```

## Credits

Built on top of [`cduck/morse`](https://github.com/cduck/morse).
