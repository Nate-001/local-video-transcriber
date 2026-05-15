from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Callable, Iterable


def _register_nvidia_dlls() -> None:
    """On Windows, the nvidia-* pip packages drop CUDA DLLs into
    site-packages/nvidia/<lib>/bin. Python doesn't search those by default,
    so ctranslate2 fails to load cublas64_12.dll / cudnn*.dll. Register them.

    `nvidia` is a PEP 420 namespace package, so it has no __file__; iterate
    its __path__ entries instead.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia  # type: ignore
    except ImportError:
        return
    bin_dirs: list[str] = []
    for nvidia_root in map(Path, nvidia.__path__):
        if not nvidia_root.is_dir():
            continue
        for sub in nvidia_root.iterdir():
            bin_dir = sub / "bin"
            if bin_dir.is_dir():
                bin_dirs.append(str(bin_dir))
                try:
                    os.add_dll_directory(str(bin_dir))
                except (OSError, FileNotFoundError):
                    pass
    # Some ctranslate2 code paths still resolve DLLs via PATH; cover both.
    if bin_dirs:
        os.environ["PATH"] = os.pathsep.join(bin_dirs + [os.environ.get("PATH", "")])


_register_nvidia_dlls()

from faster_whisper import WhisperModel  # noqa: E402


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "cuda", compute_type: str | None = None):
        if compute_type is None:
            compute_type = "float16" if device == "cuda" else "int8"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(
        self,
        media_path: str,
        language: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> dict:
        segments_iter, info = self.model.transcribe(
            media_path,
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        duration = info.duration or 0.0
        segments = []
        for seg in segments_iter:
            segments.append({
                "id": seg.id,
                "start": float(seg.start),
                "end": float(seg.end),
                "text": seg.text.strip(),
            })
            if on_progress and duration:
                on_progress(min(1.0, seg.end / duration))
        return {
            "language": info.language,
            "language_probability": info.language_probability,
            "duration": duration,
            "segments": segments,
        }


def write_outputs(result: dict, base_path: Path) -> dict[str, Path]:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    txt_path = base_path.with_suffix(".txt")
    txt_path.write_text(
        "\n".join(s["text"] for s in result["segments"]) + "\n",
        encoding="utf-8",
    )
    paths["txt"] = txt_path

    srt_path = base_path.with_suffix(".srt")
    srt_path.write_text(_render_srt(result["segments"]), encoding="utf-8")
    paths["srt"] = srt_path

    vtt_path = base_path.with_suffix(".vtt")
    vtt_path.write_text(_render_vtt(result["segments"]), encoding="utf-8")
    paths["vtt"] = vtt_path

    json_path = base_path.with_suffix(".json")
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    paths["json"] = json_path

    return paths


def _render_srt(segments: Iterable[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_time(seg['start'])} --> {_srt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _render_vtt(segments: Iterable[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{_vtt_time(seg['start'])} --> {_vtt_time(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def _srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _vtt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
