# ===== SERVER SIDE (server.py) =====
# Этот код запускает сервер с FastAPI, который принимает POST-запросы с текстом для TTS,
# генерирует аудио потоково и стримит его клиенту в формате bytes (PCM16 LE).

import torch
import numpy as np
import time
from ruaccent import RUAccent

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    infer_process,
)

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import asyncio

# ===== Настройки =====
MODEL_CFG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)
MODEL_FILE = "/home/user/.cache/huggingface/hub/models--ESpeech--ESpeech-TTS-1_RL-V2/snapshots/f582b6e5897fe8a5835059405a8439d13bdf7684/espeech_tts_rlv2.pt"
VOCAB_FILE = "/home/user/.cache/huggingface/hub/models--ESpeech--ESpeech-TTS-1_RL-V2/snapshots/f582b6e5897fe8a5835059405a8439d13bdf7684/vocab.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# ===== Модель и вокодер =====
print("Loading model...")
model = load_model(DiT, MODEL_CFG, MODEL_FILE, vocab_file=VOCAB_FILE).to(device).eval()
print("Loading vocoder...")
vocoder = load_vocoder().to(device).eval()

# ===== Accentizer =====
accentizer = RUAccent()
accentizer.load(omograph_model_size='turbo3.1', use_dictionary=True)

def process_text_with_accent(text):
    return accentizer.process_all(text)

# ===== Асинхронный генератор аудио =====
async def tts_generate_stream(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    speed: float = 1.0,
    nfe_step: int = 24,
    cross_fade_duration: float = 0.15,
    chunk_size: int = 4096,
):
    """
    Асинхронный генератор: принимает текст → возвращает аудио чанками (байты PCM16 LE).
    """

    # Preprocess reference
    ref_audio_proc, processed_ref_text = preprocess_ref_audio_text(
        ref_audio_path, process_text_with_accent(ref_text)
    )

    # Генерация
    start_time = time.time()
    with torch.no_grad():
        final_wave, final_sample_rate, _ = infer_process(
            ref_audio_proc,
            processed_ref_text,
            process_text_with_accent(text),
            model,
            vocoder,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            speed=speed,
        )
    end_time = time.time()

    # Преобразуем float32 → int16 PCM
    audio_int16 = (final_wave * 32767).astype(np.int16).tobytes()

    # Отдаём чанками
    for i in range(0, len(audio_int16), chunk_size):
        yield audio_int16[i:i+chunk_size]

    # Статистика (на сервере)
    audio_duration = len(final_wave) / final_sample_rate
    generation_time = end_time - start_time
    speed_ratio = audio_duration / generation_time
    print(f"✅ Аудио сгенерировано ({audio_duration:.2f}s) за {generation_time:.2f}s (x{speed_ratio:.2f})")

# ===== FastAPI сервер =====
app = FastAPI()

class TTSRequest(BaseModel):
    text: str
    ref_audio: str  # Путь к референсному аудио на сервере
    ref_text: str
    speed: float = 1.0
    nfe_step: int = 24
    cross_fade_duration: float = 0.15
    chunk_size: int = 4096

@app.post("/tts")
async def tts_endpoint(request: TTSRequest):
    async def stream_generator():

        async for chunk in tts_generate_stream(
            request.text,
            request.ref_audio,
            request.ref_text,
            request.speed,
            request.nfe_step,
            request.cross_fade_duration,
            request.chunk_size
        ):
            yield chunk

    return StreamingResponse(stream_generator(), media_type="audio/pcm")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5174)
