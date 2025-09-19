
# agent_with_bargein_and_coalesce.py
import openai
import os
import sys
import signal
from multiprocessing import Process
import logging
import uuid
import json
import inspect
import asyncio
from typing import Optional, Sequence, List
from time import time
import aiohttp
import numpy as np
from dotenv import load_dotenv
from rich import print
from pytimers import timer
import requests
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AutoSubscribe,
    JobContext,
    RoomOutputOptions,
    WorkerOptions,
    cli,
    llm,
    tts,
    APIConnectOptions,
)
from livekit.agents.stt import (
    STT, STTCapabilities, SpeechEvent, SpeechEventType, SpeechData, StreamAdapter,
)
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

import aiohttp
import uuid
from typing import Optional

from livekit import rtc
from livekit.agents import tts, APIConnectOptions
from livekit.agents.tts.stream_adapter import StreamAdapter as TTSStreamAdapter
from livekit.agents.tokenize import basic as tokenize_basic
from livekit.agents.utils.audio import AudioByteStream


# ---------- ваш pipeline (заглушка, если нет импорта) ----------

from validate_user_answer  import *


load_dotenv()
logger = logging.getLogger("voice")
logging.basicConfig(level=logging.INFO)

TOKEN = "4mfnqusmo3hwmzv8rc0ngi1ej2bhnt"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
#started = False
# -----------------------------
#           STT (HTTP)
# -----------------------------
class HTTPWhisperSTT(STT):
    """
    Быстрый HTTP-клиент к вашему STT-серверу:
    - пытается raw octet-stream /stt/transcribe_raw (без base64), при 404 откатывается на JSON
    - VAD внутри Whisper выключен (VAD делаем только в turn-detector)
    - beam_size=1, temperature=0.0, condition_on_previous_text=False
    """

    def __init__(self, base_url: str = "http://localhost:8010"):
        super().__init__(capabilities=STTCapabilities(streaming=False, interim_results=False))
        self.base_url = base_url.rstrip("/")
        self._sess: Optional[aiohttp.ClientSession] = None  # ленивая инициализация

    async def _ensure_session(self):
        if self._sess is None or getattr(self._sess, "closed", True):
            self._sess = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15, sock_connect=5, sock_read=10)
            )


    async def _recognize_impl(
        self,
        buffer: Sequence[rtc.AudioFrame],
        language: Optional[str] = None,
        **kwargs,
    ) -> SpeechEvent:
        # нормализация входа
        if buffer is None:
            frames: List[rtc.AudioFrame] = []
        elif isinstance(buffer, rtc.AudioFrame):
            frames = [buffer]
        else:
            frames = list(buffer)

        if not frames:
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])

        sr = getattr(frames[0], "sample_rate", 16000) or 16000

        # склейка + моно
        chunks = []
        for f in frames:
            data = getattr(f, "data", None)
            if not data:
                continue
            arr = np.frombuffer(data, dtype=np.int16)
            ch = getattr(f, "num_channels", 1) or 1
            if ch > 1:
                n = (arr.size // ch) * ch
                if n == 0:
                    continue
                arr = arr[:n].reshape(-1, ch).mean(axis=1).astype(np.int16)
            chunks.append(arr)

        if not chunks:
            return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[])

        pcm_mono = np.concatenate(chunks, axis=0).tobytes()

        # --- путь 1: быстрый octet-stream без base64 ---
        use_raw = True
        headers = {"X-Sample-Rate": str(int(sr)), "X-Language": "ru"}

        # --- путь 2 (fallback): совместимый JSON с base64 ---
        import base64
        audio_b64 = base64.b64encode(pcm_mono).decode("ascii")
        payload_json = {
            "audio_b64": audio_b64,
            "sample_rate": int(sr),
            "language": "ru",
            "beam_size": 1,
            "vad_filter": False,                   # VAD только в turn-detector
            "condition_on_previous_text": False,
            "temperature": 0.0,
        }

        # гарантируем сессию
        await self._ensure_session()

        text = ""
        try:
            async with self._sess.post(
                f"{self.base_url}/stt/transcribe_raw",
                data=pcm_mono,
                headers=headers,
            ) as resp:
                if resp.status == 404:
                    use_raw = False
                else:
                    resp.raise_for_status()
                    data = await resp.json()
                    text = (data.get("text") or "").strip()
        except Exception:
            use_raw = False

        if not use_raw:
            async with self._sess.post(
                f"{self.base_url}/stt/transcribe", json=payload_json
            ) as resp:
                resp.raise_for_status()
                data = await resp.json()
                text = (data.get("text") or "").strip()

        print(f"[green bold]USER FRAGMENT (turn-detector final) => {text}")
        alt = SpeechData(text=text, language=language or "ru")
        return SpeechEvent(type=SpeechEventType.FINAL_TRANSCRIPT, alternatives=[alt])





        
class CustomTTS(tts.TTS):
    def __init__(
        self,
        base_url: str = "http://localhost:5174",
        ref_audio: str = "new_voice.mp3",
        ref_text: str = "Если вы хотите чтоб ваш голос стал немножко нежнее",
        speed: float = 1.1,
        nfe_step: int = 24,
        cross_fade_duration: float = 0.15,
        chunk_size: int = 4096,
        sample_rate: int = 22050,
    ):
        super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=sample_rate, num_channels=1)
        self.base_url = base_url
        self.ref_audio = ref_audio
        self.ref_text = ref_text
        self.speed = speed
        self.nfe_step = nfe_step
        self.cross_fade_duration = cross_fade_duration
        self.chunk_size = chunk_size


    def synthesize(self, text: str, *, conn_options: Optional[APIConnectOptions] = None) -> tts.ChunkedStream:
        if conn_options is None:
            conn_options = tts.DEFAULT_API_CONNECT_OPTIONS
        return _CustomChunkedStream(tts=self, input_text=text, conn_options=conn_options)

class _CustomChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: CustomTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: CustomTTS = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        req_id = str(uuid.uuid4())
        payload = {
            "text": self.input_text,
            "ref_audio": self._tts.ref_audio,
            "ref_text": self._tts.ref_text,
            "speed": self._tts.speed,
            "nfe_step": self._tts.nfe_step,
            "cross_fade_duration": self._tts.cross_fade_duration,
            "chunk_size": self._tts.chunk_size,
        }
        output_emitter.initialize(request_id=req_id, sample_rate=self._tts.sample_rate, num_channels=1, mime_type="audio/pcm", stream=False)

        timeout = aiohttp.ClientTimeout(total=None, sock_connect=self._conn_options.timeout)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.post(f"{self._tts.base_url}/tts", json=payload) as resp:
                resp.raise_for_status()
                async for chunk in resp.content.iter_any():
                    if chunk:
                        output_emitter.push(chunk)
        output_emitter.flush()
        output_emitter.end_input()

async def _maybe_await(x):
    if inspect.isawaitable(x):
        return await x
    return x


# -----------------------------
#            Agent
# -----------------------------
class VoiceAgent(Agent):
    """
    - Жёсткий barge-in: стопаем TTS сразу при начале речи пользователя.
    - Post-turn coalescing: подряд идущие короткие пользовательские ходы склеиваем
      в один текст и только затем отправляем в pipeline.
    """
    def __init__(self, pipeline: AIHRPipeline, started, room_name, stt_url="http://localhost:8010", tts_url="http://localhost:5174", ):
        stt_http = HTTPWhisperSTT(base_url=stt_url)
        vad = silero.VAD.load(
            min_silence_duration=0.75,
            # при наличии в вашей сборке можно добавить:
            min_speech_duration=0.25
        )
        adapted_stt = StreamAdapter(stt=stt_http, vad=vad)
       # super().__init__(instructions="not-needed", stt=adapted_stt, tts=CustomTTS(base_url=tts_url))
       # --- TTS ---
        base_tts = CustomTTS(base_url=tts_url)
        sent_tok = tokenize_basic.SentenceTokenizer()  # базовый токенайзер предложений из SDK
        wrapped_tts = TTSStreamAdapter(tts=base_tts, sentence_tokenizer=sent_tok)
        
        super().__init__(instructions="not-needed", stt=adapted_stt, tts=wrapped_tts)
        
        
        self.pipeline = pipeline
        self.started = started
        self.room_name = room_name
        # barge-in
        self._tts_handle = None

        # coalescer (поверх turn-detector):
        self._parts: List[str] = []
        self._merge_gap_sec = 0.35   # время, в течение которого ждём следующий «ход» пользователя для склейки
        self._max_accum_sec = 0.60   # максимум, сколько можно копить перед отправкой
        self._gap_timer: Optional[asyncio.TimerHandle] = None
        self._max_timer: Optional[asyncio.TimerHandle] = None
        self._loop = asyncio.get_event_loop()
        self._flushing = False

    # ---- barge-in helpers ----
    def _stop_tts(self):
        try:
            if self._tts_handle and hasattr(self._tts_handle, "stop"):
                self._tts_handle.stop()
        except Exception:
            pass
        finally:
            self._tts_handle = None

    async def _say(self, text: str):
        res = await _maybe_await(self.session.say(text))
        self._tts_handle = res  # SpeechHandle или None/awaitable результат

    # ---- timers for coalescing ----
    def _cancel_gap(self):
        if self._gap_timer:
            self._gap_timer.cancel()
            self._gap_timer = None

    def _cancel_all(self):
        self._cancel_gap()
        if self._max_timer:
            self._max_timer.cancel()
            self._max_timer = None

    def _schedule_after_gap(self):
        # перезапускаем gap-таймер (ждём, вдруг будет ещё один ход подряд)
        self._cancel_gap()
        self._gap_timer = self._loop.call_later(self._merge_gap_sec, self._flush_cb)
        # общий максимум — один раз при первом фрагменте пачки
        if not self._max_timer:
            self._max_timer = self._loop.call_later(self._max_accum_sec, self._flush_cb)

    def _flush_cb(self):
        if not self._flushing:
            asyncio.create_task(self._flush_async())

    async def _flush_async(self):
        if self._flushing:
            return
        self._flushing = True
        try:
            self._cancel_all()
            final_text = " ".join([p for p in self._parts if p]).strip()
            self._parts.clear()

            if not final_text:
                return

            print(f"[cyan bold]FINAL USER TEXT (COALESCED) => {final_text}")
            data = await asyncio.to_thread(self.pipeline.process_user_input, final_text)
            print(data)
            reply = data.get("message", "")
            if nq := data.get("next_question"):
                reply = f"{reply}\n\n{nq}"
            if reply.strip():
                reply = reply.replace("Всё", "")
                reply = reply.replace("не задаю вопрос", "")
                reply = reply.replace("x86", "икс восемьдесят шесть")
                await self._say(reply)
            if data.get("interview_complete"):
                     requests.post(f"http://localhost:2856/interview_report?room_name={self.room_name}", headers=HEADERS, json={"report": data.get("collected_data")})
        except Exception as e:
            logger.exception("Pipeline error: %s", e)
        finally:
            self._flushing = False

    # ---- LiveKit hooks ----
    async def on_user_turn_started(self, chat_ctx: llm.ChatContext):
        # ПОЛЬЗОВАТЕЛЬ начал говорить — немедленно перебиваем ассистента 
        if self.started:
            self._stop_tts()
        # НЕ флашим буфер — ждём завершения мини-хода, чтобы дозаклеить
    @timer
    async def on_user_turn_completed(self, chat_ctx: llm.ChatContext, new_message: llm.ChatMessage):
        # Turn-detector сказал: ход завершён. Но мы дадим короткую возможность «договорить» следующей репликой.
        text = (new_message.text_content or "").strip()
        if not text:
            # даже если пусто — всё равно попробуем дождаться продолжения
            self._schedule_after_gap()
            return

        self._parts.append(text)
        print(f"[magenta]coalesce parts={len(self._parts)} last='{text[:60]}'[/magenta]")

        # ждём ещё _merge_gap_sec — если быстро придёт следующий ход, он приклеится
        self._schedule_after_gap()

async def prewarm():
    pass

def detach_unix():
    os.setsid()                         # новая сессия/группа
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    # перенаправим stdio, чтобы не зависеть от TTY родителя
    sys.stdin = open(os.devnull, 'r')
    sys.stdout = open('/tmp/child.out', 'a', buffering=1)
    sys.stderr = open('/tmp/child.err', 'a', buffering=1)

def analize_and_send_report(pipeline, room_name):
     detach_unix()
     report = pipeline.analyze_results()
     requests.post(f"http://localhost:2856/interview_report?room_name={room_name}", headers=HEADERS, json={"report": report})

# -----------------------------
#           Entrypoint
# -----------------------------
async def entrypoint(ctx: JobContext):
    started = False
    logger.info(f"starting agent, room: {ctx.room.name}")

    import requests
    scenario_data= requests.post("http://localhost:2856/scenario/get_scenario", headers=HEADERS, data=json.dumps({"room": ctx.room.name})).json()
    print(scenario_data)
    scenario = scenario_data.get("scenario", {})
    vacancy = scenario_data.get("vacancy", "")
    print(scenario, vacancy)
    if scenario:
        pipeline = AIHRPipeline(scenario, vacancy)

        await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

        # turn-detector (многоязычный) + параметры endpointing / interruptions
        session = AgentSession(
            turn_detection=MultilingualModel(),
            vad=silero.VAD.load(min_silence_duration=0.60, min_speech_duration=0.25),
            allow_interruptions=True,          # разрешаем перебивание ассистента
            min_interruption_duration=0.25,    # барг-ин с ~350 мс речи
            min_interruption_words=1,          # одного слова достаточно
            min_endpointing_delay=0.35,        # не завершать ход мгновенно
            max_endpointing_delay=3.0,         # но и не тянуть бесконечно
        )
        agent = VoiceAgent(pipeline, started, ctx.room.name)
        await session.start(
            agent=agent,
            room=ctx.room,
            room_output_options=RoomOutputOptions(
                transcription_enabled=True,
                audio_enabled=True,
            ),
        )
        print(f"{agent.started = }")
        await _maybe_await(session.say(pipeline.start_interview()))
        agent.started = True
        print(f"{agent.started = }")

async def request_fnc(req):
    # Можно сгенерировать уникальный identity для ассистента
    identity = f"assistant-{uuid.uuid4().hex[:8]}"
    await req.accept(
        name="Настя HR (bot)",     # <- имя, которое увидят в комнате
        identity=identity,       # <- уникальный identity
        metadata=json.dumps({"avatar": "ava_for_bot.jpg"}),
        # attributes={"role": "assistant"}  # опционально
    )


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint, request_fnc=request_fnc))

