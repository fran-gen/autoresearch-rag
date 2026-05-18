from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate showcase narration clips with Qwen3-TTS.")
    parser.add_argument(
        "--segments",
        type=Path,
        default=Path("scripts/showcase_tts_segments.json"),
        help="JSON file containing narration segments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/showcase_tts"),
        help="Directory where WAV files will be written.",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="Qwen3-TTS model name or local path.",
    )
    parser.add_argument(
        "--language",
        default="English",
        help="Generation language.",
    )
    parser.add_argument(
        "--speaker",
        default="Ryan",
        help="CustomVoice speaker name.",
    )
    parser.add_argument(
        "--instruct",
        default="Clear, confident English narration for a short product demo video. Moderate pace, natural pauses, neutral accent.",
        help="Optional voice instruction.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Torch device selection.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "bfloat16", "float16", "float32"],
        help="Torch dtype for model loading.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Generation token limit passed to Qwen3-TTS.",
    )
    parser.add_argument(
        "--attn-implementation",
        default=None,
        help="Optional attention implementation, for example flash_attention_2.",
    )
    parser.add_argument(
        "--only-slug",
        action="append",
        default=[],
        help="Generate only a specific segment slug. Can be passed more than once.",
    )
    return parser.parse_args()


def choose_device(torch_module, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_dtype(torch_module, requested: str, device: str):
    if requested != "auto":
        return getattr(torch_module, requested)
    if device == "cuda":
        return torch_module.bfloat16
    if device == "mps":
        return torch_module.float32
    return torch_module.float32


def move_model_to_device(model, device: str) -> None:
    if device == "cuda":
        return
    underlying_model = getattr(model, "model", None)
    if underlying_model is None:
        return
    underlying_model.to(device)
    model.device = underlying_model.device


def load_segments(path: Path) -> list[dict[str, str]]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"Expected a non-empty list of segments in {path}")
    for item in data:
        if not isinstance(item, dict) or not item.get("slug") or not item.get("text"):
            raise ValueError("Each segment must include at least 'slug' and 'text'.")
    return data


def main() -> None:
    args = parse_args()

    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    segments = load_segments(args.segments)
    indexed_segments = [dict(segment, segment_index=index) for index, segment in enumerate(segments, start=1)]
    if args.only_slug:
        wanted_slugs = set(args.only_slug)
        segments = [segment for segment in indexed_segments if segment["slug"] in wanted_slugs]
        missing_slugs = wanted_slugs - {segment["slug"] for segment in segments}
        if missing_slugs:
            raise ValueError(f"Unknown segment slug(s): {', '.join(sorted(missing_slugs))}")
    else:
        segments = indexed_segments
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(torch, args.device)
    dtype = choose_dtype(torch, args.dtype, device)

    model_kwargs: dict[str, object] = {
        "dtype": dtype,
    }
    if device == "cuda":
        model_kwargs["device_map"] = "cuda:0"
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model {args.model} on {device} with dtype={dtype}...")
    model = Qwen3TTSModel.from_pretrained(args.model, **model_kwargs)
    move_model_to_device(model, device)

    for segment in segments:
        text = str(segment["text"]).strip()
        slug = str(segment["slug"]).strip()
        title = str(segment.get("title") or slug)
        index = int(segment["segment_index"])
        if not text:
            continue

        output_path = args.output_dir / f"{index:02d}_{slug}.wav"
        print(f"Generating {output_path.name}: {title}")
        wavs, sample_rate = model.generate_custom_voice(
            text=text,
            language=args.language,
            speaker=args.speaker,
            instruct=args.instruct,
            max_new_tokens=args.max_new_tokens,
        )
        sf.write(output_path, wavs[0], sample_rate)

    print(f"Done. Wrote {len(segments)} clips to {args.output_dir}")


if __name__ == "__main__":
    main()
