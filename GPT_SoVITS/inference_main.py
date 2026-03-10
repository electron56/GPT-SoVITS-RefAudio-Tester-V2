import logging
import os
import re
import sys
from typing import Dict, Generator, List, Optional, Tuple, Union

import torch

NOW_DIR = os.getcwd()
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _path in (NOW_DIR, os.path.join(NOW_DIR, "GPT_SoVITS"), THIS_DIR):
    if _path not in sys.path:
        sys.path.append(_path)

from process_ckpt import get_sovits_version_from_path_fast
from TTS_infer_pack.TTS import NO_PROMPT_ERROR, TTS, TTS_Config

logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)

CUT_METHODS = {
    0: "cut0",
    1: "cut1",
    2: "cut2",
    3: "cut3",
    4: "cut4",
    5: "cut5",
}

LANGUAGE_LABEL_TO_CODE_V1 = {
    "Chinese": "all_zh",
    "English": "en",
    "Japanese": "all_ja",
    "Chinese-English Mix": "zh",
    "Japanese-English Mix": "ja",
    "Auto": "auto",
}

LANGUAGE_LABEL_TO_CODE_V2 = {
    "Chinese": "all_zh",
    "English": "en",
    "Japanese": "all_ja",
    "Cantonese": "all_yue",
    "Korean": "all_ko",
    "Chinese-English Mix": "zh",
    "Japanese-English Mix": "ja",
    "Cantonese-English Mix": "yue",
    "Korean-English Mix": "ko",
    "Auto": "auto",
    "Auto (Cantonese Priority)": "auto_yue",
}

LANGUAGE_ALIAS_TO_CODE = {
    "zh": "all_zh",
    "zh-cn": "all_zh",
    "cn": "all_zh",
    "chinese": "all_zh",
    "en": "en",
    "english": "en",
    "jp": "all_ja",
    "ja": "all_ja",
    "japanese": "all_ja",
    "yue": "all_yue",
    "cantonese": "all_yue",
    "ko": "all_ko",
    "korean": "all_ko",
}

TTS_CONFIG_PATH = os.path.join("GPT_SoVITS", "configs", "tts_infer.yaml")

_tts_pipeline: Optional[TTS] = None
_state: Dict[str, Union[str, bool, None]] = {
    "version": "v2",
    "sovits_path": None,
    "gpt_path": None,
    "device": None,
    "is_half": None,
}


def custom_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def _detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _detect_half_precision(device: str) -> bool:
    return str(device).startswith("cuda")


def _language_map_for_version(version: str) -> Dict[str, str]:
    return LANGUAGE_LABEL_TO_CODE_V1 if version == "v1" else LANGUAGE_LABEL_TO_CODE_V2


def get_current_model_version() -> str:
    global _state
    if _tts_pipeline is not None:
        _state["version"] = _tts_pipeline.configs.version
    return str(_state.get("version") or "v2")


def get_supported_language_map() -> Dict[str, str]:
    return dict(_language_map_for_version(get_current_model_version()))


def get_supported_languages() -> List[str]:
    return list(get_supported_language_map().keys())


def get_default_language() -> str:
    languages = get_supported_languages()
    return languages[0] if languages else "Chinese"


def get_sample_steps_default() -> int:
    return 32 if get_current_model_version() == "v3" else 8


def is_super_sampling_supported() -> bool:
    return get_current_model_version() == "v3"


def _normalize_language_to_code(language: Optional[str], fallback_code: str = "all_zh") -> str:
    language_map = get_supported_language_map()
    supported_codes = set(language_map.values())

    if not language:
        return fallback_code if fallback_code in supported_codes else next(iter(supported_codes))

    raw = str(language).strip()
    if raw in language_map:
        code = language_map[raw]
    elif raw in supported_codes:
        code = raw
    else:
        code = LANGUAGE_ALIAS_TO_CODE.get(raw.lower(), fallback_code)

    if code not in supported_codes:
        code = fallback_code if fallback_code in supported_codes else next(iter(supported_codes))
    return code


def normalize_ref_language(language: Optional[str]) -> str:
    code = _normalize_language_to_code(language)
    for label, mapped_code in get_supported_language_map().items():
        if mapped_code == code:
            return label
    return get_default_language()


def detect_sovits_version(sovits_path: str) -> str:
    _, model_version, _ = get_sovits_version_from_path_fast(sovits_path)
    return model_version


def initialize(
    gpt_path: Optional[str] = None,
    sovits_path: Optional[str] = None,
    device: Optional[str] = None,
    is_half: Optional[bool] = None,
) -> str:
    global _tts_pipeline

    device = device or _detect_device()
    if is_half is None:
        is_half = _detect_half_precision(device)
    if str(device) == "cpu":
        is_half = False

    tts_config = TTS_Config(TTS_CONFIG_PATH)
    tts_config.device = device
    tts_config.is_half = bool(is_half)

    if gpt_path and os.path.exists(gpt_path):
        tts_config.t2s_weights_path = gpt_path
    if sovits_path and os.path.exists(sovits_path):
        tts_config.vits_weights_path = sovits_path

    _tts_pipeline = TTS(tts_config)
    _state["gpt_path"] = gpt_path or _tts_pipeline.configs.t2s_weights_path
    _state["sovits_path"] = sovits_path or _tts_pipeline.configs.vits_weights_path
    _state["version"] = _tts_pipeline.configs.version
    _state["device"] = str(device)
    _state["is_half"] = bool(is_half)
    return _tts_pipeline.configs.version


def _ensure_initialized() -> None:
    if _tts_pipeline is None:
        initialize(
            gpt_path=_state.get("gpt_path"),
            sovits_path=_state.get("sovits_path"),
            device=_state.get("device") or None,
            is_half=_state.get("is_half"),
        )


def change_sovits_weights(sovits_path: str) -> str:
    if not sovits_path or not os.path.exists(sovits_path):
        raise FileNotFoundError(f"SoVITS model not found: {sovits_path}")

    _ensure_initialized()
    assert _tts_pipeline is not None
    _tts_pipeline.init_vits_weights(sovits_path)

    _state["sovits_path"] = sovits_path
    _state["version"] = _tts_pipeline.configs.version
    return _tts_pipeline.configs.version


def change_gpt_weights(gpt_path: str) -> str:
    if not gpt_path or not os.path.exists(gpt_path):
        raise FileNotFoundError(f"GPT model not found: {gpt_path}")

    _ensure_initialized()
    assert _tts_pipeline is not None
    _tts_pipeline.init_t2s_weights(gpt_path)

    _state["gpt_path"] = gpt_path
    return gpt_path


def get_tts_wav(
    ref_wav_path: str,
    prompt_text: str,
    prompt_language: str,
    text: str,
    text_language: str,
    how_to_cut: int = 0,
    top_k: int = 20,
    top_p: float = 0.6,
    temperature: float = 0.6,
    speed_factor: float = 1.0,
    repetition_penalty: float = 1.35,
    seed: Union[int, float] = -1,
    sample_steps: Optional[int] = None,
    super_sampling: bool = False,
) -> Generator[Tuple[int, torch.Tensor], None, None]:
    if not ref_wav_path:
        raise ValueError("Reference audio path is required.")
    if not os.path.exists(ref_wav_path):
        raise FileNotFoundError(f"Reference audio not found: {ref_wav_path}")

    _ensure_initialized()
    assert _tts_pipeline is not None

    prompt_text = (prompt_text or "").strip()
    text = (text or "").strip()
    if not text:
        raise ValueError("Synthesis text is empty.")

    prompt_lang_code = _normalize_language_to_code(prompt_language)
    text_lang_code = _normalize_language_to_code(text_language)

    if sample_steps is None:
        sample_steps = get_sample_steps_default()

    if not is_super_sampling_supported():
        super_sampling = False

    try:
        seed_value = int(seed)
    except (TypeError, ValueError):
        seed_value = -1

    cut_method = CUT_METHODS.get(int(how_to_cut), "cut0")

    inputs = {
        "text": text,
        "text_lang": text_lang_code,
        "ref_audio_path": ref_wav_path,
        "aux_ref_audio_paths": [],
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang_code,
        "top_k": int(top_k),
        "top_p": float(top_p),
        "temperature": float(temperature),
        "text_split_method": cut_method,
        "batch_size": 1,
        "batch_threshold": 0.75,
        "split_bucket": False,
        "speed_factor": float(speed_factor),
        "fragment_interval": 0.3,
        "seed": seed_value,
        "parallel_infer": True,
        "repetition_penalty": float(repetition_penalty),
        "sample_steps": int(sample_steps),
        "super_sampling": bool(super_sampling),
        "return_fragment": False,
        "streaming_mode": False,
    }

    try:
        for output in _tts_pipeline.run(inputs):
            yield output
    except NO_PROMPT_ERROR as exc:
        version = get_current_model_version()
        raise ValueError(f"Prompt text is required for {version} models.") from exc
