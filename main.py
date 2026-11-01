#!/usr/bin/env python3
import os
import json
import shutil
import tempfile
import threading
import subprocess
from typing import List, Dict, Any, Optional

import torch
import numpy as np
import scipy.io.wavfile as wavfile
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="AI Cinematic Anime Engine - GitHub CPU Worker (Svara TTS & FFmpeg)")

DEFAULT_TG_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
DEFAULT_TG_CHAT_ID = os.getenv("TELEGRAM_STORAGE_CHAT_ID", "")
DEFAULT_STREAM_BRIDGE = os.getenv(
    "TELEGRAM_STREAM_BRIDGE_URL", "https://api.bnai.dpdns.org/v1/media/telegram"
).rstrip("/")

# Open-Source Commercial-Safe Models (Apache 2.0 / CC-BY)
SVARA_MODEL_ID = os.getenv("SVARA_MODEL_ID", "kenpath/svara-tts-v1")
SNAC_MODEL_ID = os.getenv("SNAC_MODEL_ID", "hubertsiuzdak/snac_24khz")
VITS_HINDI_MODEL_ID = os.getenv("VITS_HINDI_MODEL_ID", "facebook/mms-tts-hin")
# Set SVARA_ENGINE_MODE="FULL_LLM" on 16GB+ RAM or GPU; "FAST_CPU_VITS" on 7GB GitHub Actions CPU
SVARA_ENGINE_MODE = os.getenv("SVARA_ENGINE_MODE", "AUTO")

TTS_LOCK = threading.Lock()
TTS_CACHE: Dict[str, Any] = {}

# Voice acoustic profiles matching WF-01 & WF-03 catalog (Rule 39)
VOICE_ACOUSTIC_MAP = {
    "svara_hi_male_hero_01": {
        "speaker_tag": "hindi_male",
        "gender": "male",
        "base_pitch": 0.96,
        "base_rate": 1.02,
        "eq_filter": "equalizer=f=140:width_type=h:width=100:g=3,equalizer=f=3200:width_type=h:width=1000:g=2",
    },
    "svara_hi_female_lead_01": {
        "speaker_tag": "hindi_female",
        "gender": "female",
        "base_pitch": 1.22,
        "base_rate": 1.04,
        "eq_filter": "equalizer=f=240:width_type=h:width=100:g=2,equalizer=f=4500:width_type=h:width=1200:g=3",
    },
    "svara_hi_narrator_epic_01": {
        "speaker_tag": "hindi_male",
        "gender": "male",
        "base_pitch": 0.88,
        "base_rate": 0.96,
        "eq_filter": "equalizer=f=110:width_type=h:width=80:g=5,equalizer=f=2800:width_type=h:width=800:g=2",
    },
    "supertonic_hi_male_deep_02": {
        "speaker_tag": "hindi_male",
        "gender": "male",
        "base_pitch": 0.84,
        "base_rate": 0.95,
        "eq_filter": "equalizer=f=95:width_type=h:width=70:g=6,equalizer=f=2400:width_type=h:width=800:g=2",
    },
    "supertonic_hi_female_calm_02": {
        "speaker_tag": "hindi_female",
        "gender": "female",
        "base_pitch": 1.16,
        "base_rate": 0.97,
        "eq_filter": "equalizer=f=220:width_type=h:width=100:g=2,equalizer=f=3800:width_type=h:width=1000:g=2",
    },
}

EMOTION_TAG_MAP = {
    "neutral": "neutral",
    "anger": "angry",
    "joy": "happy",
    "sadness": "sad",
    "fear": "fearful",
}


# ==============================================================================
# PYDANTIC SCHEMAS (MATCHING storywf03.json)
# ==============================================================================
class TelegramUploadConfig(BaseModel):
    enabled: bool = True
    bot_token: Optional[str] = None
    chat_id: Optional[str] = None
    stream_bridge_url: Optional[str] = None


class TtsSynthesisRequest(BaseModel):
    project_id: str
    episode_no: int
    scene_order: int
    provider: str = "Svara TTS"
    voice_text: str
    voice_id: str = "svara_hi_male_hero_01"
    emotion: str = "neutral"
    pitch_multiplier: float = 1.0
    rate_multiplier: float = 1.0
    telegram_upload: Optional[TelegramUploadConfig] = None


class SfxTrackItem(BaseModel):
    sfx_code: str = "DEFAULT_SILENCE_00"
    audio_url: Optional[str] = None
    fallback_tier: Optional[str] = None


class SceneManifestItem(BaseModel):
    scene_order: int
    duration_sec: float = 5.0
    routing_type: str = "STATIC"
    camera_motion_type: str = "KEN_BURNS"
    video_input_url: Optional[str] = None
    image_input_url: Optional[str] = None
    voice_audio_url: Optional[str] = None
    has_dialogue: bool = False
    subtitle_text: str = ""
    bgm_track_url: Optional[str] = None
    bgm_tier: Optional[str] = None
    sfx_tracks: List[SfxTrackItem] = []


class FfmpegEpisodeRequest(BaseModel):
    project_id: str
    episode_no: int
    engine: str = "GITHUB_CPU_FFMPEG"
    ducking_db: float = -8.0
    crossfade_ms: int = 500
    target_loudness_lufs: float = -14.0
    burn_subtitles: bool = True
    filter_complex: Optional[str] = None
    raw_ffmpeg_command: Optional[str] = None
    scenes_manifest: List[SceneManifestItem]
    telegram_upload: Optional[TelegramUploadConfig] = None


# ==============================================================================
# HELPERS: TELEGRAM UPLOAD & ASSET DOWNLOADER
# ==============================================================================
def upload_to_telegram(
    file_path: str, caption: str, tg_cfg: Optional[TelegramUploadConfig]
) -> Dict[str, Optional[str]]:
    bot_token = (tg_cfg.bot_token if tg_cfg and tg_cfg.bot_token else DEFAULT_TG_BOT_TOKEN).strip()
    chat_id = (tg_cfg.chat_id if tg_cfg and tg_cfg.chat_id else DEFAULT_TG_CHAT_ID).strip()
    bridge_base = (
        tg_cfg.stream_bridge_url if tg_cfg and tg_cfg.stream_bridge_url else DEFAULT_STREAM_BRIDGE
    ).rstrip("/")

    if not bot_token or not chat_id or not os.path.exists(file_path):
        return {"file_id": None, "telegram_url": None}

    url = f"https://api.telegram.org/bot{bot_token}/sendDocument"
    for attempt in range(3):
        try:
            with open(file_path, "rb") as f:
                res = requests.post(
                    url,
                    data={"chat_id": chat_id, "caption": caption[:1024]},
                    files={"document": f},
                    timeout=180,
                )
            if res.ok:
                result = res.json().get("result", {})
                media_obj = (
                    result.get("document")
                    or result.get("video")
                    or result.get("audio")
                    or result.get("voice")
                    or {}
                )
                file_id = media_obj.get("file_id")
                if file_id:
                    stream_url = f"{bridge_base}/{file_id}" if bridge_base else None
                    return {"file_id": file_id, "telegram_url": stream_url}
        except Exception as exc:
            print(f"[TELEGRAM UPLOAD RETRY {attempt + 1}/3] {exc}")
    return {"file_id": None, "telegram_url": None}


def download_file(url: str, dest_path: str) -> bool:
    if not url or not url.startswith("http"):
        return False
    try:
        headers = {"Bypass-Tunnel-Reminder": "true", "User-Agent": "AnimeEngineCPUWorker/1.0"}
        with requests.get(url, headers=headers, stream=True, timeout=60) as r:
            if not r.ok:
                return False
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)
        return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
    except Exception as exc:
        print(f"[DOWNLOAD WARN] Failed {url}: {exc}")
        return False


def probe_duration(file_path: str, default_sec: float = 5.0) -> float:
    try:
        out = subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ],
            text=True,
            timeout=15,
        ).strip()
        val = float(out)
        return val if val > 0.2 else default_sec
    except Exception:
        return default_sec


# ==============================================================================
# OPEN-SOURCE SVARA-TTS (kenpath/svara-tts-v1) & VITS LOADER
# ==============================================================================
def load_svara_llm_engine():
    if "svara_llm" in TTS_CACHE:
        return TTS_CACHE["svara_llm"]

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from snac import SNAC

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.bfloat16

    print(f"[SVARA-TTS] Loading {SVARA_MODEL_ID} & SNAC 24kHz on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(SVARA_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        SVARA_MODEL_ID,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    snac_model = SNAC.from_pretrained(SNAC_MODEL_ID).eval().to(device)
    TTS_CACHE["svara_llm"] = (tokenizer, model, snac_model, device)
    return TTS_CACHE["svara_llm"]


def load_fast_hindi_vits_engine():
    if "hindi_vits" in TTS_CACHE:
        return TTS_CACHE["hindi_vits"]

    from transformers import VitsModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[SVARA-VITS] Loading commercial-safe {VITS_HINDI_MODEL_ID} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(VITS_HINDI_MODEL_ID)
    model = VitsModel.from_pretrained(VITS_HINDI_MODEL_ID).to(device)
    model.eval()
    TTS_CACHE["hindi_vits"] = (tokenizer, model, device)
    return TTS_CACHE["hindi_vits"]


def decode_svara_snac_tokens(generated_ids: List[int], snac_model: Any, device: str) -> Optional[np.ndarray]:
    """Decodes Svara-TTS SNAC 7-token audio frames into 24kHz PCM float waveform."""
    AUDIO_CODE_BASE_OFFSET = 128266
    audio_tokens = [t - AUDIO_CODE_BASE_OFFSET for t in generated_ids if t >= AUDIO_CODE_BASE_OFFSET]
    num_frames = len(audio_tokens) // 7
    if num_frames == 0:
        return None

    audio_tokens = audio_tokens[: num_frames * 7]
    codes_0, codes_1, codes_2 = [], [], []

    for i in range(num_frames):
        base = i * 7
        codes_0.append(audio_tokens[base] % 4096)
        codes_1.extend([
            (audio_tokens[base + 1] - 4096) % 4096,
            (audio_tokens[base + 4] - (4 * 4096)) % 4096,
        ])
        codes_2.extend([
            (audio_tokens[base + 2] - (2 * 4096)) % 4096,
            (audio_tokens[base + 3] - (3 * 4096)) % 4096,
            (audio_tokens[base + 5] - (5 * 4096)) % 4096,
            (audio_tokens[base + 6] - (6 * 4096)) % 4096,
        ])

    codes = [
        torch.tensor(codes_0, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(codes_1, dtype=torch.long, device=device).unsqueeze(0),
        torch.tensor(codes_2, dtype=torch.long, device=device).unsqueeze(0),
    ]
    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)
    return audio_hat.squeeze().cpu().numpy()


def roman_hinglish_to_devanagari_fallback(text: str) -> str:
    """Ensures Roman Hinglish words also pronounce cleanly on Devanagari VITS models."""
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
        # If text has no Devanagari characters, transliterate Roman Hinglish -> Devanagari
        if not any("\u0900" <= ch <= "\u097f" for ch in text):
            return transliterate(text.lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    except Exception:
        pass
    return text


def synthesize_open_source_speech(
    text: str,
    voice_id: str,
    emotion: str,
    raw_wav_path: str,
) -> str:
    """Runs Svara-TTS v1 (or fast VITS on low-RAM CPU runners) with zero Microsoft/Edge code."""
    profile = VOICE_ACOUSTIC_MAP.get(voice_id, VOICE_ACOUSTIC_MAP["svara_hi_male_hero_01"])
    emotion_tag = EMOTION_TAG_MAP.get(emotion.lower(), "neutral")
    use_full_svara = (
        SVARA_ENGINE_MODE.upper() == "FULL_LLM"
        or (SVARA_ENGINE_MODE.upper() == "AUTO" and torch.cuda.is_available())
    )

    with TTS_LOCK:
        if use_full_svara:
            try:
                tokenizer, model, snac_model, device = load_svara_llm_engine()
                prompt = f"{profile['speaker_tag']} ({emotion_tag}): {text.strip()}"
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=1200,
                        do_sample=True,
                        temperature=0.65,
                        top_p=0.9,
                        repetition_penalty=1.1,
                    )
                gen_tokens = output_ids[0][inputs["input_ids"].shape[1] :].tolist()
                waveform = decode_svara_snac_tokens(gen_tokens, snac_model, device)
                if waveform is not None and len(waveform) > 100:
                    pcm16 = np.int16(np.clip(waveform, -1.0, 1.0) * 32767)
                    wavfile.write(raw_wav_path, 24000, pcm16)
                    return raw_wav_path
            except Exception as e:
                print(f"[SVARA LLM FALLBACK TO VITS] {e}")

        # Commercial-Safe Open-Source VITS Synthesis (Runs in <0.8s on GitHub Actions CPU)
        tokenizer, model, device = load_fast_hindi_vits_engine()
        norm_text = roman_hinglish_to_devanagari_fallback(text.strip())
        inputs = tokenizer(norm_text, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model(**inputs).waveform
        waveform = output.squeeze().cpu().numpy()
        sr = int(model.config.sampling_rate)
        pcm16 = np.int16(np.clip(waveform, -1.0, 1.0) * 32767)
        wavfile.write(raw_wav_path, sr, pcm16)
        return raw_wav_path


def run_tts_pipeline(req: TtsSynthesisRequest) -> Dict[str, Any]:
    if not req.voice_text or not req.voice_text.strip():
        raise HTTPException(status_code=400, detail="voice_text cannot be empty")

    profile = VOICE_ACOUSTIC_MAP.get(req.voice_id, VOICE_ACOUSTIC_MAP["svara_hi_male_hero_01"])

    with tempfile.TemporaryDirectory() as tmpdir:
        raw_wav = os.path.join(tmpdir, "raw_open_source.wav")
        final_wav = os.path.join(
            tmpdir, f"proj_{req.project_id[:8]}_ep{req.episode_no}_sc{req.scene_order}_voice.wav"
        )

        synthesize_open_source_speech(req.voice_text, req.voice_id, req.emotion, raw_wav)

        # Combine character base pitch/rate with scene emotion multipliers from storywf03
        total_pitch = max(
            0.70, min(1.45, float(profile["base_pitch"]) * float(req.pitch_multiplier or 1.0))
        )
        total_rate = max(
            0.75, min(1.50, float(profile["base_rate"]) * float(req.rate_multiplier or 1.0))
        )
        effective_tempo = max(0.5, min(2.0, total_rate / total_pitch))

        in_sr = 16000
        try:
            in_sr, _ = wavfile.read(raw_wav)
        except Exception:
            in_sr = 16000

        af_chain = (
            f"asetrate={in_sr}*{total_pitch:.3f},aresample=44100,atempo={effective_tempo:.3f},"
            f"{profile['eq_filter']},loudnorm=I=-16:TP=-1.5:LRA=11"
        )

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_wav,
                "-af",
                af_chain,
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                final_wav,
            ],
            check=True,
            timeout=60,
        )

        file_size = os.path.getsize(final_wav)
        duration_sec = probe_duration(final_wav, 4.0)

        caption = (
            f"🎙️ [{req.provider}] Project: {req.project_id} | Ep: {req.episode_no} | "
            f"Scene: {req.scene_order} | Voice: {req.voice_id} ({req.emotion})"
        )
        tg_res = upload_to_telegram(final_wav, caption, req.telegram_upload)

        return {
            "ok": True,
            "provider": req.provider,
            "license": "OPEN_SOURCE_COMMERCIAL_SAFE",
            "voice_id": req.voice_id,
            "duration_sec": round(duration_sec, 2),
            "file_size_bytes": file_size,
            "telegram_file_id": tg_res["file_id"],
            "tg_file_id": tg_res["file_id"],
            "telegram_url": tg_res["telegram_url"],
            "audio_url": tg_res["telegram_url"],
        }


# ==============================================================================
# ENDPOINT 1A & 1B: /v1/tts/svara (Primary) & /v1/tts/supertonic (Fallback)
# ==============================================================================
@app.post("/v1/tts/svara")
def synthesize_svara_endpoint(req: TtsSynthesisRequest):
    return run_tts_pipeline(req)


@app.post("/v1/tts/supertonic")
def synthesize_supertonic_endpoint(req: TtsSynthesisRequest):
    return run_tts_pipeline(req)


# ==============================================================================
# SUBTITLE (.ASS) & SCENE CAMERA MOTION BUILDERS
# ==============================================================================
def format_ass_time(seconds: float) -> str:
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    if centis >= 100:
        centis = 99
    return f"{hrs}:{mins:02d}:{secs:02d}.{centis:02d}"


def generate_ass_subtitle_file(
    subtitle_events: List[Dict[str, Any]], ass_path: str
) -> str:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Noto Sans Devanagari,30,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.5,1.0,2,40,40,36,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header.strip()]
    for ev in subtitle_events:
        text = (ev.get("text") or "").replace("\n", "\\N").strip()
        if not text:
            continue
        start_str = format_ass_time(ev["start"])
        end_str = format_ass_time(ev["end"])
        lines.append(f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return ass_path


def render_single_scene_segment(
    sc: SceneManifestItem,
    tmpdir: str,
    idx: int,
) -> Dict[str, Any]:
    seg_video_out = os.path.join(tmpdir, f"seg_video_{idx:03d}.mp4")
    seg_voice_out = os.path.join(tmpdir, f"seg_voice_{idx:03d}.wav")

    raw_voice = os.path.join(tmpdir, f"raw_voice_{idx:03d}.wav")
    has_voice_file = False
    if sc.voice_audio_url:
        has_voice_file = download_file(sc.voice_audio_url, raw_voice)

    scene_dur = float(sc.duration_sec or 5.0)
    if has_voice_file:
        voice_dur = probe_duration(raw_voice, scene_dur)
        scene_dur = max(scene_dur, voice_dur + 0.35)
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                raw_voice,
                "-af",
                f"apad=whole_dur={scene_dur:.2f}",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                seg_voice_out,
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{scene_dur:.2f}",
                "-c:a",
                "pcm_s16le",
                seg_voice_out,
            ],
            check=True,
        )

    seg_sfx_out = os.path.join(tmpdir, f"seg_sfx_{idx:03d}.wav")
    downloaded_sfx = []
    for s_idx, sfx_item in enumerate(sc.sfx_tracks or []):
        if sfx_item.audio_url:
            sfx_path = os.path.join(tmpdir, f"sfx_{idx:03d}_{s_idx}.mp3")
            if download_file(sfx_item.audio_url, sfx_path):
                downloaded_sfx.append(sfx_path)

    if downloaded_sfx:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                downloaded_sfx[0],
                "-af",
                f"volume=0.35,apad=whole_dur={scene_dur:.2f}",
                "-t",
                f"{scene_dur:.2f}",
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                seg_sfx_out,
            ],
            check=True,
        )
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                "anullsrc=r=44100:cl=stereo",
                "-t",
                f"{scene_dur:.2f}",
                "-c:a",
                "pcm_s16le",
                seg_sfx_out,
            ],
            check=True,
        )

    visual_url = sc.video_input_url or sc.image_input_url
    raw_visual = os.path.join(tmpdir, f"raw_visual_{idx:03d}")
    has_visual = download_file(visual_url or "", raw_visual)

    total_frames = max(24, int(round(scene_dur * 24)))
    cam_type = (sc.camera_motion_type or "KEN_BURNS").upper()

    if has_visual:
        is_video_stream = False
        try:
            probe_out = subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,codec_name",
                    "-of",
                    "json",
                    raw_visual,
                ],
                text=True,
            )
            streams = json.loads(probe_out).get("streams", [])
            if streams:
                codec = streams[0].get("codec_name", "")
                nb = int(streams[0].get("nb_frames") or 1)
                if codec in ("h264", "hevc", "vp9", "mpeg4") and nb > 1:
                    is_video_stream = True
        except Exception:
            is_video_stream = False

        if is_video_stream:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-stream_loop",
                    "-1",
                    "-i",
                    raw_visual,
                    "-t",
                    f"{scene_dur:.2f}",
                    "-vf",
                    "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=24,format=yuv420p",
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    seg_video_out,
                ],
                check=True,
            )
        else:
            if cam_type in ("KEN_BURNS", "ZOOM"):
                vf = (
                    f"scale=2560:1440,zoompan=z='min(zoom+0.0012,1.25)':"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
                    f"d={total_frames}:s=1280x720:fps=24,format=yuv420p"
                )
            elif cam_type == "PAN":
                vf = (
                    f"scale=2560:1440,zoompan=z=1.15:"
                    f"x='(iw-iw/zoom)*(on/{total_frames})':y='ih/2-(ih/zoom/2)':"
                    f"d={total_frames}:s=1280x720:fps=24,format=yuv420p"
                )
            else:
                vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=24,format=yuv420p"

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loop",
                    "1",
                    "-i",
                    raw_visual,
                    "-t",
                    f"{scene_dur:.2f}",
                    "-vf",
                    vf,
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    seg_video_out,
                ],
                check=True,
            )
    else:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "lavfi",
                "-i",
                f"color=c=#101420:s=1280x720:r=24:d={scene_dur:.2f}",
                "-vf",
                "format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                seg_video_out,
            ],
            check=True,
        )

    return {
        "video_path": seg_video_out,
        "voice_path": seg_voice_out,
        "sfx_path": seg_sfx_out,
        "duration_sec": scene_dur,
        "subtitle_text": sc.subtitle_text,
    }


# ==============================================================================
# ENDPOINT 2: /v1/ffmpeg/render-episode (Branch F Master Episode Assembler)
# ==============================================================================
@app.post("/v1/ffmpeg/render-episode")
def render_episode(req: FfmpegEpisodeRequest):
    if not req.scenes_manifest:
        raise HTTPException(status_code=400, detail="scenes_manifest cannot be empty")

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            sorted_scenes = sorted(req.scenes_manifest, key=lambda x: x.scene_order)
            rendered_segments = []
            subtitle_events = []
            cursor_sec = 0.0

            for idx, sc in enumerate(sorted_scenes):
                seg = render_single_scene_segment(sc, tmpdir, idx)
                rendered_segments.append(seg)
                dur = seg["duration_sec"]
                if seg["subtitle_text"]:
                    subtitle_events.append(
                        {
                            "start": cursor_sec + 0.1,
                            "end": cursor_sec + max(0.5, dur - 0.1),
                            "text": seg["subtitle_text"],
                        }
                    )
                cursor_sec += dur

            total_ep_duration = max(1.0, cursor_sec)

            vid_list_txt = os.path.join(tmpdir, "vid_list.txt")
            voice_list_txt = os.path.join(tmpdir, "voice_list.txt")
            sfx_list_txt = os.path.join(tmpdir, "sfx_list.txt")

            with open(vid_list_txt, "w") as fv, open(voice_list_txt, "w") as fa, open(
                sfx_list_txt, "w"
            ) as fs:
                for seg in rendered_segments:
                    fv.write(f"file '{seg['video_path']}'\n")
                    fa.write(f"file '{seg['voice_path']}'\n")
                    fs.write(f"file '{seg['sfx_path']}'\n")

            concat_vid = os.path.join(tmpdir, "concat_video.mp4")
            concat_voice = os.path.join(tmpdir, "concat_voice.wav")
            concat_sfx = os.path.join(tmpdir, "concat_sfx.wav")

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", vid_list_txt, "-c", "copy", concat_vid],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", voice_list_txt, "-c", "copy", concat_voice],
                check=True,
            )
            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", sfx_list_txt, "-c", "copy", concat_sfx],
                check=True,
            )

            bgm_wav = os.path.join(tmpdir, "bgm_master.wav")
            bgm_raw_dl = os.path.join(tmpdir, "bgm_raw.mp3")
            first_bgm_url = next(
                (s.bgm_track_url for s in sorted_scenes if s.bgm_track_url), None
            )
            if first_bgm_url and download_file(first_bgm_url, bgm_raw_dl):
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-stream_loop",
                        "-1",
                        "-i",
                        bgm_raw_dl,
                        "-t",
                        f"{total_ep_duration:.2f}",
                        "-ar",
                        "44100",
                        "-ac",
                        "2",
                        "-c:a",
                        "pcm_s16le",
                        bgm_wav,
                    ],
                    check=True,
                )
            else:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "lavfi",
                        "-i",
                        "anullsrc=r=44100:cl=stereo",
                        "-t",
                        f"{total_ep_duration:.2f}",
                        "-c:a",
                        "pcm_s16le",
                        bgm_wav,
                    ],
                    check=True,
                )

            ass_path = os.path.join(tmpdir, "episode_subtitles.ass")
            has_subs = req.burn_subtitles and len(subtitle_events) > 0
            if has_subs:
                generate_ass_subtitle_file(subtitle_events, ass_path)

            final_mp4 = os.path.join(
                tmpdir, f"proj_{req.project_id[:8]}_ep_{req.episode_no}_master.mp4"
            )

            audio_filter = (
                "[2:a]volume=0.28[bgm_low];"
                "[bgm_low][1:a]sidechaincompress=threshold=0.025:ratio=6:attack=40:release=400[bgm_ducked];"
                "[1:a][bgm_ducked][3:a]amix=inputs=3:duration=first:dropout_transition=0.5[a_mixed];"
                f"[a_mixed]loudnorm=I={req.target_loudness_lufs}:TP=-1.5:LRA=11[audio_master]"
            )

            if has_subs:
                escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
                full_filter = f"{audio_filter};[0:v]subtitles='{escaped_ass}'[video_final]"
                video_map = "[video_final]"
            else:
                full_filter = audio_filter
                video_map = "0:v"

            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    concat_vid,
                    "-i",
                    concat_voice,
                    "-i",
                    bgm_wav,
                    "-i",
                    concat_sfx,
                    "-filter_complex",
                    full_filter,
                    "-map",
                    video_map,
                    "-map",
                    "[audio_master]",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "20",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    final_mp4,
                ],
                check=True,
                timeout=600,
            )

            file_size = os.path.getsize(final_mp4)
            caption = (
                f"🎬 [MASTER EPISODE] Project: {req.project_id} | Episode: {req.episode_no} | "
                f"Scenes: {len(sorted_scenes)} | Duration: {round(total_ep_duration, 1)}s"
            )
            tg_res = upload_to_telegram(final_mp4, caption, req.telegram_upload)

            return {
                "ok": True,
                "engine": req.engine,
                "episode_no": req.episode_no,
                "duration_sec": round(total_ep_duration, 2),
                "file_size_bytes": file_size,
                "telegram_file_id": tg_res["file_id"],
                "tg_file_id": tg_res["file_id"],
                "telegram_url": tg_res["telegram_url"],
                "final_video_url": tg_res["telegram_url"],
                "output_url": tg_res["telegram_url"],
            }
    except subprocess.CalledProcessError as cpe:
        raise HTTPException(status_code=500, detail=f"FFmpeg execution error: {cpe}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Episode rendering failed: {exc}")


@app.get("/health")
def health():
    return {
        "ok": True,
        "worker": "github-cpu-worker",
        "tts_engine": "Svara-TTS v1 + Open-Source Hindi VITS (Apache 2.0 / Commercial Safe)",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
