"""Run once online, then set ASR_MODEL to the output directory and ASR_LOCAL_ONLY=1."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--output", type=Path, default=Path("models/asr-small"))
    args = parser.parse_args()
    from faster_whisper.utils import download_model
    result = download_model(args.model, output_dir=str(args.output.resolve()))
    print(f"ASR_MODEL={result}\nASR_LOCAL_ONLY=1")


if __name__ == "__main__":
    main()

