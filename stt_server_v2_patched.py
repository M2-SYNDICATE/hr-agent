# stt_server.py
import os
import base64
from typing import Optional, List, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from faster_whisper import WhisperModel

APP_SAMPLE_RATE = 16000  # целевой SR float32
app = FastAPI(title="Whisper STT Service")

# -----------------------------
#            Models
# -----------------------------
class STTRequest(BaseModel):
    audio_b64: str
    sample_rate: int
    language: Optional[str] = None           # 'ru', 'en', ...
    beam_size: int = 1
    vad_filter: bool = False
    # расширенные опции:
    return_words: bool = False               # слова с таймингами
    no_speech_threshold: Optional[float] = 0.6
    log_prob_threshold: Optional[float] = -1.0
    compression_ratio_threshold: Optional[float] = 2.4
    temperature: float = 0.0                 # 0.0..1.0 (0.0 => beam search)
    condition_on_previous_text: bool = False  # лучше пунктуация/согласованность
    task: Literal["transcribe", "translate"] = "transcribe"

class Word(BaseModel):
    start: float
    end: float
    word: str
    prob: Optional[float] = None

class Segment(BaseModel):
    start: float
    end: float
    text: str
    avg_prob: Optional[float] = None
    no_speech_prob: Optional[float] = None

class STTResponse(BaseModel):
    text: str
    language: str
    final_punct: Optional[str] = Field(None, description="'.', '!', '?' если последние символы содержат финальную пунктуацию")
    is_question: bool = False
    segments: List[Segment] = []
    words: Optional[List[Word]] = None
    endpoint_hints: List[float] = Field(default_factory=list, description="кандидаты таймингов для endpointing (секунды)")

# -----------------------------
#            Utils
# -----------------------------
def _try_load_model(model_size: str, device: str, candidates: List[str]) -> WhisperModel:
    last_err = None
    for ct in candidates:
        try:
            print(f"[whisper] trying compute_type={ct} on device={device}")
            return WhisperModel(model_size, device=device, compute_type=ct, cpu_threads=int(os.getenv("WHISPER_CPU_THREADS", "4")))
        except Exception as e:
            print(f"[whisper] failed compute_type={ct}: {e}")
            last_err = e
    raise RuntimeError(f"Failed to load WhisperModel on device={device} with {candidates}: {last_err}")

@app.on_event("startup")
def _load_model():
    device = os.getenv("WHISPER_DEVICE", "cuda").lower()   # 'cpu'|'cuda'
    pref = os.getenv("WHISPER_COMPUTE", "float16").lower()   # 'auto'|'int8'|'float16'...
    model_size = os.getenv("WHISPER_MODEL_SIZE", "small")

    if device == "cuda":
        candidates = [pref] if pref != "auto" else ["float16", "int8_float16", "int8", "bfloat16", "float32"]
    else:
        candidates = [pref] if pref != "auto" else ["int8", "int8_float32", "bfloat16", "float32"]

    app.state.whisper = _try_load_model(model_size=model_size, device=device, candidates=candidates)
    print(f"[whisper] loaded model='{model_size}' on device='{device}'")

def _resample_linear(x: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or len(x) == 0:
        return x.astype(np.float32)
    duration = len(x) / float(orig_sr)
    new_len = max(1, int(round(duration * target_sr)))
    old_idx = np.linspace(0.0, duration, num=len(x), endpoint=False, dtype=np.float32)
    new_idx = np.linspace(0.0, duration, num=new_len, endpoint=False, dtype=np.float32)
    y = np.interp(new_idx, old_idx, x.astype(np.float32))
    return y.astype(np.float32)

# простая эвристика вопроса/финальной пунктуации + подсказки endpoint-а
def _post_text_hints(text: str) -> tuple[Optional[str], bool]:
    t = (text or "").strip()
    final_punct = None
    is_question = False
    if t.endswith("?"):
        final_punct = "?"
        is_question = True
    elif t.endswith("!"):
        final_punct = "!"
    elif t.endswith("."):
        final_punct = "."
    return final_punct, is_question

# -----------------------------
#          Endpoint
# -----------------------------
@app.post("/stt/transcribe", response_model=STTResponse)
def transcribe(req: STTRequest):
    if not req.audio_b64:
        raise HTTPException(status_code=400, detail="Empty audio payload")

    try:
        pcm_bytes = base64.b64decode(req.audio_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64: {e}")

    try:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PCM16 buffer: {e}")

    if audio_i16.size == 0:
        return STTResponse(text="", language=req.language or "ru")

    # i16 -> f32 [-1..1]
    audio_f32 = (audio_i16.astype(np.float32) / 32768.0)
    audio_16k = _resample_linear(audio_f32, req.sample_rate, APP_SAMPLE_RATE)

    model: WhisperModel = app.state.whisper
    # faster-whisper iterator возвращает сегменты с полями start/end/text, avg_logprob, no_speech_prob, words
    segments_iter, info = model.transcribe(
        audio_16k,
        language="ru",
        vad_filter=bool(req.vad_filter),
        beam_size=max(1, req.beam_size),
        task=req.task,
        temperature=req.temperature,
        condition_on_previous_text=req.condition_on_previous_text,
        no_speech_threshold=req.no_speech_threshold,
        log_prob_threshold=req.log_prob_threshold,
        compression_ratio_threshold=req.compression_ratio_threshold,
        word_timestamps=req.return_words,
        initial_prompt=None,  # можно подставить, если есть доменная лексика
    )

    # соберём всё в удобный ответ
    segs: List[Segment] = []
    words: List[Word] = []

    txt_parts: List[str] = []
    endpoint_hints: List[float] = []

    for seg in segments_iter:
        txt_parts.append(seg.text.strip())
        segs.append(Segment(
            start=float(seg.start or 0.0),
            end=float(seg.end or 0.0),
            text=seg.text.strip(),
            avg_prob=float(getattr(seg, "avg_logprob", None)) if getattr(seg, "avg_logprob", None) is not None else None,
            no_speech_prob=float(getattr(seg, "no_speech_prob", None)) if getattr(seg, "no_speech_prob", None) is not None else None,
        ))
        # хинт: конец сегмента — хороший кандидат endpoint-а
        endpoint_hints.append(float(seg.end or 0.0))

        if req.return_words and getattr(seg, "words", None):
            for w in seg.words:
                words.append(Word(
                    start=float(w.start or 0.0),
                    end=float(w.end or 0.0),
                    word=w.word,
                    prob=float(getattr(w, "probability", None)) if getattr(w, "probability", None) is not None else None
                ))

    text = " ".join([t for t in (p.strip() for p in txt_parts) if t]).strip()
    final_punct, is_question = _post_text_hints(text)

    return STTResponse(
        text=text,
        language=req.language or (info.language if getattr(info, "language", None) else "ru"),
        final_punct=final_punct,
        is_question=is_question,
        segments=segs,
        words=words or None,
        endpoint_hints=endpoint_hints,
    )


# --- новый быстрый эндпоинт без base64 ---
from fastapi import Request

@app.post("/stt/transcribe_raw", response_model=STTResponse)
async def transcribe_raw(request: Request):
    try:
        pcm_bytes = await request.body()
        if not pcm_bytes:
            raise HTTPException(status_code=400, detail="Empty audio payload")
        sample_rate = int(request.headers.get("X-Sample-Rate", "16000"))
        lang = request.headers.get("X-Language", "ru")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Bad headers/body: {e}")

    try:
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid PCM16 buffer: {e}")

    if audio_i16.size == 0:
        return STTResponse(text="", language=lang)

    audio_f32 = (audio_i16.astype(np.float32) / 32768.0)
    audio_16k = _resample_linear(audio_f32, sample_rate, APP_SAMPLE_RATE)

    model: WhisperModel = app.state.whisper
    segments_iter, info = model.transcribe(
        audio_16k,
        language=lang,
        vad_filter=False,                    # VAD здесь выключен
        beam_size=1,
        task="transcribe",
        temperature=0.0,
        condition_on_previous_text=False,
        word_timestamps=False,
    )
    segs, words, txt_parts, endpoint_hints = [], [], [], []
    for seg in segments_iter:
        txt = (seg.text or "").strip()
        if not txt:
            continue
        txt_parts.append(txt)
        segs.append(Segment(start=float(seg.start or 0.0), end=float(seg.end or 0.0), text=txt))
        endpoint_hints.append(float(seg.end or 0.0))
    text = " ".join([t for t in (p.strip() for p in txt_parts) if t]).strip()
    final_punct, is_question = _post_text_hints(text)
    return STTResponse(
        text=text,
        language=lang,
        final_punct=final_punct,
        is_question=is_question,
        segments=segs,
        endpoint_hints=endpoint_hints,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
