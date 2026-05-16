from __future__ import annotations

import json
import mimetypes
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import chat as chat_module
from transcriber import Transcriber, write_outputs

BASE_DIR = Path(__file__).parent
UPLOADS_DIR = BASE_DIR / "uploads"
OUTPUTS_DIR = BASE_DIR / "outputs"
DOWNLOADS_DIR = BASE_DIR / "downloads"
STATIC_DIR = BASE_DIR / "static"
for d in (UPLOADS_DIR, OUTPUTS_DIR, DOWNLOADS_DIR):
    d.mkdir(exist_ok=True)

ALLOWED_FORMATS = {"txt", "srt", "vtt", "json"}
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}
ALLOWED_DEVICES = {"cuda", "cpu", "auto"}
ALLOWED_SOURCES = {"upload", "path", "url"}

app = FastAPI(title="Local Video Transcriber")

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

_models: dict[tuple[str, str], Transcriber] = {}
_models_lock = threading.Lock()


def _save_meta(job_dir: Path, **fields) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    meta_path = job_dir / "meta.json"
    existing: dict = {}
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            existing = {}
    existing.update(fields)
    meta_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _load_meta(job_dir: Path) -> dict:
    meta_path = job_dir / "meta.json"
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_legacy_media(job_id: str) -> Path | None:
    for f in UPLOADS_DIR.glob(f"{job_id}_*"):
        return f
    sub = DOWNLOADS_DIR / job_id
    if sub.is_dir():
        files = [p for p in sub.iterdir() if p.is_file()]
        if files:
            return files[0]
    return None


def _save_summary(job_dir: Path, model: str, text: str, error: str | None = None) -> None:
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "text": text,
        "actions": chat_module.extract_action_items(text),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "error": error,
    }
    (job_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_summary(job_dir: Path) -> dict | None:
    p = job_dir / "summary.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _hydrate_existing_jobs() -> None:
    """Populate _jobs from any transcripts already on disk in outputs/."""
    for job_dir in OUTPUTS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        json_path = chat_module.find_transcript_json(job_dir)
        if json_path is None:
            continue
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        stem = json_path.stem
        meta = _load_meta(job_dir)
        media_path = meta.get("media_path")
        if not media_path:
            legacy = _find_legacy_media(job_dir.name)
            if legacy:
                media_path = str(legacy)
        with _jobs_lock:
            if job_dir.name in _jobs:
                continue
            _jobs[job_dir.name] = {
                "status": "done",
                "progress": 1.0,
                "filename": meta.get("title") or stem,
                "source_type": meta.get("source_type", "loaded"),
                "model": "(loaded)",
                "device": "(loaded)",
                "media_path": media_path,
                "mtime": json_path.stat().st_mtime,
                "result": {
                    "language": data.get("language", "?"),
                    "language_probability": data.get("language_probability", 0.0),
                    "duration": data.get("duration", 0.0),
                    "segment_count": len(data.get("segments", [])),
                    "stem": stem,
                },
            }


def _get_model(model_size: str, device: str) -> Transcriber:
    key = (model_size, device)
    with _models_lock:
        if key not in _models:
            _models[key] = Transcriber(model_size=model_size, device=device)
        return _models[key]


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        _jobs[job_id].update(fields)


_INVALID_STEM_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _safe_stem(name: str, fallback: str) -> str:
    """Turn an arbitrary video title into a filesystem-safe file stem."""
    cleaned = _INVALID_STEM_CHARS.sub("", name)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().strip(".")
    if len(cleaned) > 120:
        cleaned = cleaned[:120].rstrip()
    return cleaned or fallback


def _download_url(url: str, dest_dir: Path, job_id: str) -> tuple[Path, str]:
    import yt_dlp

    dest_dir.mkdir(parents=True, exist_ok=True)

    def hook(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes", 0)
            if total:
                _update_job(job_id, progress=min(0.99, done / total))

    ydl_opts = {
        # Prefer a single mp4 so the browser can play it directly in <video>.
        "format": "best[ext=mp4]/best",
        "outtmpl": str(dest_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if "entries" in info:
            info = info["entries"][0]
        path = Path(ydl.prepare_filename(info))
        title = (info.get("title") or "").strip()

    if not path.exists():
        # yt-dlp may post-process to a different extension
        matches = list(dest_dir.glob(f"{path.stem}.*"))
        if not matches:
            raise FileNotFoundError(f"Downloaded file not found in {dest_dir}")
        path = matches[0]
    return path, title or path.stem


def _run_job(
    job_id: str,
    source_type: str,
    source_value: str,
    stem: str,
    model_size: str,
    language: str | None,
    device: str,
) -> None:
    title: str | None = None
    try:
        if source_type == "url":
            _update_job(job_id, status="downloading", progress=0.0)
            media_path, title = _download_url(source_value, DOWNLOADS_DIR / job_id, job_id)
            stem = _safe_stem(title, fallback=media_path.stem)
            _update_job(job_id, filename=title)
        else:
            media_path = Path(source_value)

        _update_job(job_id, status="loading_model", progress=0.0)
        model = _get_model(model_size, device)

        _update_job(job_id, status="transcribing")

        def on_progress(p: float) -> None:
            _update_job(job_id, progress=max(0.0, min(1.0, p)))

        result = model.transcribe(str(media_path), language=language, on_progress=on_progress)

        out_base = OUTPUTS_DIR / job_id / stem
        write_outputs(result, out_base)

        # Persist where the original media lives so the player can find it later.
        meta_fields = {"media_path": str(media_path), "source_type": source_type}
        if title:
            meta_fields["title"] = title
        _save_meta(out_base.parent, **meta_fields)
        _update_job(job_id, media_path=str(media_path))

        # Auto-generate summary as part of the job (best-effort).
        _update_job(job_id, status="summarizing")
        summary_model = chat_module.resolve_default_model()
        if summary_model and result["segments"]:
            try:
                transcript_text = chat_module.format_transcript(result["segments"])
                summary_text = chat_module.summarize(transcript_text, summary_model)
                _save_summary(out_base.parent, summary_model, summary_text)
            except Exception as e:
                _save_summary(
                    out_base.parent, summary_model, "",
                    error=f"{type(e).__name__}: {e}",
                )
        elif not summary_model:
            _save_summary(out_base.parent, "", "", error="No Ollama chat models available")

        _update_job(
            job_id,
            status="done",
            progress=1.0,
            result={
                "language": result["language"],
                "language_probability": result["language_probability"],
                "duration": result["duration"],
                "segment_count": len(result["segments"]),
                "stem": stem,
            },
        )
    except Exception as e:
        _update_job(job_id, status="error", error=f"{type(e).__name__}: {e}")


@app.post("/api/transcribe")
async def start_transcription(
    file: UploadFile | None = File(None),
    source_type: str = Form("upload"),
    path: str = Form(""),
    url: str = Form(""),
    model: str = Form("base"),
    language: str = Form(""),
    device: str = Form("cuda"),
):
    if model not in ALLOWED_MODELS:
        raise HTTPException(400, f"model must be one of {sorted(ALLOWED_MODELS)}")
    if device not in ALLOWED_DEVICES:
        raise HTTPException(400, f"device must be one of {sorted(ALLOWED_DEVICES)}")
    if source_type not in ALLOWED_SOURCES:
        raise HTTPException(400, f"source_type must be one of {sorted(ALLOWED_SOURCES)}")

    job_id = uuid.uuid4().hex[:12]

    if source_type == "upload":
        if file is None or not file.filename:
            raise HTTPException(400, "No file uploaded")
        safe_name = Path(file.filename).name
        upload_path = UPLOADS_DIR / f"{job_id}_{safe_name}"
        with upload_path.open("wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
        source_value = str(upload_path)
        display_name = safe_name
        stem = Path(safe_name).stem
    elif source_type == "path":
        if not path.strip():
            raise HTTPException(400, "Missing 'path'")
        candidate = Path(path.strip()).expanduser()
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(400, f"File not found: {candidate}")
        source_value = str(candidate)
        display_name = candidate.name
        stem = candidate.stem
    else:  # url
        if not url.strip():
            raise HTTPException(400, "Missing 'url'")
        source_value = url.strip()
        display_name = source_value
        stem = "transcript"  # replaced after download

    with _jobs_lock:
        _jobs[job_id] = {
            "status": "queued",
            "progress": 0.0,
            "filename": display_name,
            "source_type": source_type,
            "model": model,
            "device": device,
        }

    threading.Thread(
        target=_run_job,
        args=(job_id, source_type, source_value, stem, model, language.strip() or None, device),
        daemon=True,
    ).start()

    return {"job_id": job_id}


@app.get("/api/jobs")
def list_jobs():
    with _jobs_lock:
        done = [(jid, dict(j)) for jid, j in _jobs.items() if j.get("status") == "done"]
    items = []
    for jid, j in done:
        items.append({
            "job_id": jid,
            "filename": j.get("filename"),
            "mtime": j.get("mtime", 0.0),
            "has_media": bool(_resolve_media(jid)),
            "result": j.get("result"),
        })
    items.sort(key=lambda x: x["mtime"], reverse=True)
    return {"jobs": items}


@app.post("/api/jobs/{job_id}/rename")
def rename_job(job_id: str, payload: dict = Body(...)):
    name = (payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Missing 'name'")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(404, "Unknown job id")
        job["filename"] = name
    _save_meta(OUTPUTS_DIR / job_id, title=name)
    return {"ok": True, "name": name}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str):
    with _jobs_lock:
        _jobs.pop(job_id, None)
    with _chats_lock:
        _chats.pop(job_id, None)
    shutil.rmtree(OUTPUTS_DIR / job_id, ignore_errors=True)
    shutil.rmtree(DOWNLOADS_DIR / job_id, ignore_errors=True)
    for f in UPLOADS_DIR.glob(f"{job_id}_*"):
        try:
            f.unlink()
        except OSError:
            pass
    return {"ok": True}


def _resolve_media(job_id: str) -> Path | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if job:
        mp = job.get("media_path")
        if mp:
            p = Path(mp)
            if p.exists() and p.is_file():
                return p
    legacy = _find_legacy_media(job_id)
    if legacy and legacy.exists():
        return legacy
    return None


@app.api_route("/api/media/{job_id}", methods=["GET", "HEAD"])
def media(job_id: str, request: Request):
    media_path = _resolve_media(job_id)
    if not media_path:
        raise HTTPException(404, "Media file not available")

    file_size = media_path.stat().st_size
    media_type, _ = mimetypes.guess_type(str(media_path))
    media_type = media_type or "application/octet-stream"

    range_header = request.headers.get("range") or request.headers.get("Range")
    if not range_header:
        return FileResponse(
            media_path,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Content-Length": str(file_size)},
        )

    m = re.match(r"bytes=(\d*)-(\d*)", range_header)
    if not m:
        raise HTTPException(416, "Invalid Range header")
    start = int(m.group(1)) if m.group(1) else 0
    end = int(m.group(2)) if m.group(2) else file_size - 1
    if start >= file_size or end < start:
        raise HTTPException(416, "Requested range not satisfiable")
    end = min(end, file_size - 1)
    length = end - start + 1

    def iter_range():
        with media_path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    return StreamingResponse(iter_range(), status_code=206, headers=headers, media_type=media_type)


@app.get("/api/status/{job_id}")
def get_status(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    return job


@app.get("/api/download/{job_id}/{fmt}")
def download(job_id: str, fmt: str):
    if fmt not in ALLOWED_FORMATS:
        raise HTTPException(400, f"fmt must be one of {sorted(ALLOWED_FORMATS)}")
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    if job.get("status") != "done":
        raise HTTPException(409, f"Job not ready (status={job.get('status')})")
    stem = job["result"]["stem"]
    out_path = OUTPUTS_DIR / job_id / f"{stem}.{fmt}"
    if not out_path.exists():
        raise HTTPException(404, "Output file missing")
    media_types = {
        "txt": "text/plain",
        "srt": "application/x-subrip",
        "vtt": "text/vtt",
        "json": "application/json",
    }
    return FileResponse(out_path, filename=f"{stem}.{fmt}", media_type=media_types[fmt])


_chats: dict[str, list[dict]] = {}
_chats_lock = threading.Lock()


@app.get("/api/chat/models")
def chat_models():
    return {"models": chat_module.list_models(), "default": chat_module.DEFAULT_MODEL}


@app.get("/api/summary/{job_id}")
def get_summary(job_id: str):
    job_dir = OUTPUTS_DIR / job_id
    summary = _load_summary(job_dir)
    if not summary:
        raise HTTPException(404, "No summary saved for this job")
    return summary


@app.post("/api/summary/{job_id}")
def generate_summary(job_id: str, payload: dict = Body(default={})):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    if job.get("status") != "done":
        raise HTTPException(409, f"Transcript not ready (status={job.get('status')})")

    job_dir = OUTPUTS_DIR / job_id
    try:
        segments, _ = chat_module.load_transcript_segments(job_dir)
    except FileNotFoundError:
        raise HTTPException(404, "Transcript file missing")
    transcript_text = chat_module.format_transcript(segments)

    model = (payload.get("model") or "").strip() or chat_module.resolve_default_model()
    if not model:
        raise HTTPException(503, "No Ollama chat models available")

    def generate():
        collected: list[str] = []
        try:
            for chunk in chat_module.summarize_stream(transcript_text, model):
                collected.append(chunk)
                yield chunk
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if "Connection" in type(e).__name__ or "ConnectError" in err or "refused" in err.lower():
                err = "Ollama not reachable at http://localhost:11434"
            yield f"\n\n[error: {err}]"
            _save_summary(job_dir, model, "".join(collected), error=err)
            return
        full = "".join(collected)
        _save_summary(job_dir, model, full)

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.get("/api/chat/{job_id}")
def get_chat(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    with _chats_lock:
        return {"messages": list(_chats.get(job_id, []))}


@app.post("/api/chat/{job_id}")
def post_chat(job_id: str, payload: dict = Body(...)):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job id")
    if job.get("status") != "done":
        raise HTTPException(409, f"Transcript not ready (status={job.get('status')})")

    message = (payload.get("message") or "").strip()
    if not message:
        raise HTTPException(400, "Missing 'message'")
    model = (payload.get("model") or chat_module.DEFAULT_MODEL).strip()

    job_dir = OUTPUTS_DIR / job_id
    try:
        segments, _ = chat_module.load_transcript_segments(job_dir)
    except FileNotFoundError:
        raise HTTPException(404, "Transcript file missing")
    transcript_text = chat_module.format_transcript(segments)

    with _chats_lock:
        history = list(_chats.get(job_id, []))

    def generate():
        collected: list[str] = []
        try:
            for chunk in chat_module.chat_stream(transcript_text, history, message, model=model):
                collected.append(chunk)
                yield chunk
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if "Connection" in type(e).__name__ or "ConnectError" in err or "refused" in err.lower():
                yield (
                    "\n\n[error: Ollama is not reachable at http://localhost:11434. "
                    "Install from https://ollama.com and run e.g. `ollama pull llama3.2`.]"
                )
            else:
                yield f"\n\n[error: {err}]"
            return
        full = "".join(collected)
        if full.strip():
            with _chats_lock:
                hist = _chats.setdefault(job_id, [])
                hist.append({"role": "user", "content": message})
                hist.append({"role": "assistant", "content": full})

    return StreamingResponse(generate(), media_type="text/plain; charset=utf-8")


@app.delete("/api/chat/{job_id}")
def reset_chat(job_id: str):
    with _chats_lock:
        _chats.pop(job_id, None)
    return {"ok": True}


app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


_hydrate_existing_jobs()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
