# pawse

> **A minimal python Morse-code encoder/decoder**

---

## Features

* **Encode** ASCII text to CW audio (NumPy buffer or WAV file).
* **Decode** CW audio (WAV or NumPy array) to uppercase text.
* Optional real-time playback with `sounddevice`.

---

## Requirements

|   Purpose    |      Package      |         Version                 |
|--------------|-------------------|---------------------------------|
|   **Core**   | `numpy`           | >= 2.0.0                        |
|              | `scipy`           | >= 1.10                         |
| **Optional** | `sounddevice`     | >= 0.4 &nbsp;*(audio playback)* |

Python **3.10+** required.

---

## Installation

```bash
# core only
pip install git+https://github.com/fluffy-awoo/pawse.git

# with audio playback
pip install "git+https://github.com/fluffy-awoo/pawse.git[audio]"
```

---

## Example

```py
from pawse import Codec

mc = Codec(wpm=20, frequency=750)

# text to WAV
mc.to_wav("SOS", "output.wav")

# WAV to text
print(mc.from_wav("output.wav"))
```

## Credits

Built on top of [`cduck/morse`](https://github.com/cduck/morse).
