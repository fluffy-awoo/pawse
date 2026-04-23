from __future__ import annotations

import argparse
import sys

from .core import Codec


def _codec(args: argparse.Namespace) -> Codec:
    return Codec(
        wpm=args.wpm,
        frequency=args.frequency,
        farnsworth_wpm=args.farnsworth_wpm,
        volume=args.volume,
    )


def _add_codec_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--wpm",
        type=float,
        default=25.0,
        metavar="WPM",
        help="character speed in words per minute (default: 25)",
    )
    p.add_argument(
        "--frequency",
        type=float,
        default=750.0,
        metavar="HZ",
        help="tone frequency in Hz (default: 750)",
    )
    p.add_argument(
        "--farnsworth-wpm",
        type=float,
        default=None,
        metavar="WPM",
        dest="farnsworth_wpm",
        help="overall speed in wpm — stretches gaps only, keeping characters at --wpm",
    )
    p.add_argument(
        "--volume",
        type=float,
        default=0.9,
        metavar="VOL",
        help="output volume 0.0–1.0 (default: 0.9)",
    )


def cmd_encode(args: argparse.Namespace) -> None:
    print(_codec(args).to_morse(args.text))


def cmd_decode(args: argparse.Namespace) -> None:
    print(_codec(args).from_wav(args.file))


def cmd_play(args: argparse.Namespace) -> None:
    try:
        _codec(args).play(args.text)
        import sounddevice as sd

        sd.wait()
    except RuntimeError as e:
        sys.exit(str(e))


def cmd_to_wav(args: argparse.Namespace) -> None:
    _codec(args).to_wav(args.text, args.output)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="pawse", description="Morse code encoder/decoder"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_encode = sub.add_parser("encode", help="Text to Morse symbols")
    p_encode.add_argument("text")
    _add_codec_args(p_encode)
    p_encode.set_defaults(func=cmd_encode)

    p_decode = sub.add_parser("decode", help="WAV file to text")
    p_decode.add_argument("file")
    _add_codec_args(p_decode)
    p_decode.set_defaults(func=cmd_decode)

    p_play = sub.add_parser("play", help="Play text as Morse audio")
    p_play.add_argument("text")
    _add_codec_args(p_play)
    p_play.set_defaults(func=cmd_play)

    p_wav = sub.add_parser("to-wav", help="Write text as Morse WAV file")
    p_wav.add_argument("text")
    p_wav.add_argument("output")
    _add_codec_args(p_wav)
    p_wav.set_defaults(func=cmd_to_wav)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()
