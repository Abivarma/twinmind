"""Speech-to-text: pause-based chunking + pluggable on-device Whisper backends."""
from __future__ import annotations

import logging
import platform
import re

import numpy as np

log = logging.getLogger("twin.stt")

SAMPLE_RATE = 16000
FRAME = 480  # 30 ms at 16 kHz
PREROLL = 10  # frames of silence kept before speech starts

# Whisper invents these on near-silent audio; drop them when they are the whole chunk.
_HALLUCINATIONS = {
    "thank you", "thank you.", "thanks for watching!", "thanks for watching.", "you",
    "bye.", "bye", ".", "subtitles by the amara.org community", "thank you very much.",
}


class Chunker:
    """Buffers 16 kHz int16 PCM and releases chunks cut at natural pauses."""

    def __init__(self, min_sec=2.5, max_sec=12.0, silence_rms=0.008, pause_sec=0.6):
        self.min_samples = int(min_sec * SAMPLE_RATE)
        self.max_samples = int(max_sec * SAMPLE_RATE)
        self.silence_rms = silence_rms
        self.pause_frames = max(1, int(pause_sec * SAMPLE_RATE / FRAME))
        self._buf: list[np.ndarray] = []
        self._len = 0
        self._pending = np.zeros(0, dtype=np.float32)
        self._trailing_silent = 0
        self._voiced = 0
        self._frames = 0
        self.samples_seen = 0          # total samples fed (for timestamps)
        self._chunk_start = 0

    def feed(self, pcm: bytes) -> list[tuple[float, np.ndarray]]:
        """Returns [(start_seconds, float32 audio)] for chunks that are ready."""
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        audio = np.concatenate([self._pending, audio])
        out = []
        n_frames = len(audio) // FRAME
        for i in range(n_frames):
            frame = audio[i * FRAME:(i + 1) * FRAME]
            if self._len == 0:
                self._chunk_start = self.samples_seen
            self._buf.append(frame)
            self._len += FRAME
            self._frames += 1
            self.samples_seen += FRAME
            if float(np.sqrt(np.mean(frame * frame))) >= self.silence_rms:
                self._voiced += 1
                self._trailing_silent = 0
            else:
                self._trailing_silent += 1
                if self._voiced == 0 and len(self._buf) > PREROLL:
                    # no speech yet: keep only a short pre-roll instead of leading silence
                    self._buf.pop(0)
                    self._len -= FRAME
                    self._frames -= 1
                    self._chunk_start += FRAME
            paused = self._len >= self.min_samples and self._trailing_silent >= self.pause_frames
            if paused or self._len >= self.max_samples:
                chunk = self._take()
                if chunk is not None:
                    out.append(chunk)
        self._pending = audio[n_frames * FRAME:]
        return out

    def flush(self) -> list[tuple[float, np.ndarray]]:
        chunk = self._take()
        return [chunk] if chunk is not None else []

    def _take(self):
        if not self._buf:
            return None
        audio = np.concatenate(self._buf)
        voiced_ratio = self._voiced / max(1, self._frames)
        start = self._chunk_start / SAMPLE_RATE
        self._buf, self._len, self._voiced, self._frames, self._trailing_silent = [], 0, 0, 0, 0
        if voiced_ratio < 0.08:  # essentially silence: don't feed Whisper
            return None
        return start, audio


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if text.lower() in _HALLUCINATIONS:
        return ""
    return text


class Transcriber:
    name = "base"

    def transcribe(self, audio: np.ndarray) -> dict:
        raise NotImplementedError

    def transcribe_file(self, path: str) -> list[dict]:
        raise NotImplementedError


class MLXTranscriber(Transcriber):
    """Whisper on the Apple GPU via MLX. Fastest option on M-series Macs."""

    name = "mlx-whisper"
    REPOS = {
        "tiny": "mlx-community/whisper-tiny-mlx",
        "base": "mlx-community/whisper-base-mlx",
        "small": "mlx-community/whisper-small-mlx",
        "medium": "mlx-community/whisper-medium-mlx",
        "large-v3": "mlx-community/whisper-large-v3-mlx",
        "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    }

    def __init__(self, model: str, language: str = "", translate: bool = False):
        import mlx_whisper  # noqa: F401  (fail fast if not installed)

        self._mlx = mlx_whisper
        self.repo = self.REPOS.get(model, model)
        self.opts = {"language": language or None, "task": "translate" if translate else "transcribe"}

    def transcribe(self, audio):
        r = self._mlx.transcribe(audio, path_or_hf_repo=self.repo,
                                 condition_on_previous_text=False, **self.opts)
        return {"text": clean_text(r.get("text", "")), "language": r.get("language")}

    def transcribe_file(self, path):
        r = self._mlx.transcribe(path, path_or_hf_repo=self.repo, **self.opts)
        return [{"start": s["start"], "text": clean_text(s["text"])} for s in r.get("segments", [])]


class FasterWhisperTranscriber(Transcriber):
    """CPU Whisper (CTranslate2). Works everywhere; slower than MLX on a Mac."""

    name = "faster-whisper"

    def __init__(self, model: str, language: str = "", translate: bool = False):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(model, device="auto", compute_type="int8")
        self.opts = {"language": language or None, "task": "translate" if translate else "transcribe"}

    def transcribe(self, audio):
        segments, info = self.model.transcribe(audio, beam_size=1, vad_filter=True,
                                               condition_on_previous_text=False, **self.opts)
        return {"text": clean_text(" ".join(s.text for s in segments)), "language": info.language}

    def transcribe_file(self, path):
        segments, _ = self.model.transcribe(path, vad_filter=True, **self.opts)
        return [{"start": s.start, "text": clean_text(s.text)} for s in segments]


class NullTranscriber(Transcriber):
    """No STT installed: the UI still works for notes, chat and memory."""

    name = "none"

    def transcribe(self, audio):
        return {"text": "", "language": None}

    def transcribe_file(self, path):
        return []


def make_transcriber(cfg: dict) -> Transcriber:
    s = cfg["stt"]
    backend = s["backend"]
    args = (s["model"], s["language"], s["translate"])
    if backend == "none":
        return NullTranscriber()
    candidates = []
    if backend in ("auto", "mlx"):
        if backend == "mlx" or (platform.system() == "Darwin" and platform.machine() == "arm64"):
            candidates.append(MLXTranscriber)
    if backend in ("auto", "faster-whisper"):
        candidates.append(FasterWhisperTranscriber)
    for cls in candidates:
        try:
            t = cls(*args)
            log.info("STT backend: %s (%s)", t.name, s["model"])
            return t
        except ImportError as e:
            log.warning("%s unavailable: %s", cls.name, e)
    log.warning("No STT backend installed; install requirements-mac.txt. Transcription disabled.")
    return NullTranscriber()
