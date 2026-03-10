import argparse
import csv
import logging
import os
import random
import re
import shutil
from typing import Dict, List, Optional

import gradio as gr
import librosa
from tqdm import tqdm

from GPT_SoVITS import inference_main

logging.getLogger("PIL.Image").propagate = False

SOVITS_WEIGHT_ROOTS = [
    "SoVITS_weights",
    "SoVITS_weights_v2",
    "SoVITS_weights_v3",
    "SoVITS_weights_v4",
    "SoVITS_weights_v2Pro",
    "SoVITS_weights_v2ProPlus",
]

GPT_WEIGHT_ROOTS = [
    "GPT_weights",
    "GPT_weights_v2",
    "GPT_weights_v3",
    "GPT_weights_v4",
    "GPT_weights_v2Pro",
    "GPT_weights_v2ProPlus",
]

CUT_LABEL_TO_VALUE = {
    "No Cut": 0,
    "Every 4 Sentences": 1,
    "Every 50 Characters": 2,
    "By Chinese Full Stop": 3,
    "By English Full Stop": 4,
    "By Punctuation": 5,
}

LEGACY_REF_LANGUAGE_ALIASES = {
    "ZH": "Chinese",
    "zh": "Chinese",
    "JP": "Japanese",
    "jp": "Japanese",
    "JA": "Japanese",
    "ja": "Japanese",
    "EN": "English",
    "en": "English",
    "En": "English",
    "YUE": "Cantonese",
    "yue": "Cantonese",
    "KO": "Korean",
    "ko": "Korean",
}

# Runtime state
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
g_ref_folder = ""
g_batch = 10
g_index = 0
g_ref_list: List[Dict[str, str]] = []
g_ref_list_max_index = 0
g_ref_audio_path_list: List[Optional[str]] = []
g_SoVITS_names: List[str] = []
g_GPT_names: List[str] = []


def custom_sort_key(value: str):
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part for part in parts]


def check_audio_duration(path: str) -> bool:
    try:
        wav16k, _ = librosa.load(path, sr=16000)
        return 48000 <= wav16k.shape[0] <= 160000
    except Exception as exc:
        print(f"Error when checking audio {path}: {exc}")
        return False


def remove_noncompliant_audio_from_list() -> None:
    global g_ref_list, g_ref_list_max_index
    print("Checking audio duration ...")
    filtered = []
    for item in tqdm(g_ref_list):
        if check_audio_duration(item["path"]):
            filtered.append(item)
    g_ref_list = filtered
    g_ref_list_max_index = max(len(g_ref_list) - 1, 0)


def load_ref_list_file(path: str) -> None:
    global g_ref_list, g_ref_list_max_index
    records: List[Dict[str, str]] = []

    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="|")
        for row in reader:
            if not row:
                continue
            if len(row) < 4:
                row = row + [""] * (4 - len(row))
            audio_path = row[0].strip()
            if g_ref_folder:
                audio_path = os.path.join(g_ref_folder, os.path.basename(audio_path))

            records.append(
                {
                    "path": audio_path,
                    "lang": row[2].strip(),
                    "text": row[3].strip(),
                }
            )

    g_ref_list = records
    g_ref_list_max_index = max(len(g_ref_list) - 1, 0)


def get_weights_names():
    sovits_names: List[str] = []
    for root in SOVITS_WEIGHT_ROOTS:
        os.makedirs(root, exist_ok=True)
        for name in os.listdir(root):
            if name.lower().endswith(".pth"):
                sovits_names.append(f"{root}/{name}")

    gpt_names: List[str] = []
    for root in GPT_WEIGHT_ROOTS:
        os.makedirs(root, exist_ok=True)
        for name in os.listdir(root):
            if name.lower().endswith(".ckpt"):
                gpt_names.append(f"{root}/{name}")

    return sorted(sovits_names, key=custom_sort_key), sorted(gpt_names, key=custom_sort_key)


def refresh_model_list():
    global g_SoVITS_names, g_GPT_names
    g_SoVITS_names, g_GPT_names = get_weights_names()

    status = (
        f"Refreshed. SoVITS: {len(g_SoVITS_names)} | GPT: {len(g_GPT_names)}"
    )
    return (
        {"choices": g_SoVITS_names, "__type__": "update"},
        {"choices": g_GPT_names, "__type__": "update"},
        status,
    )


def _normalize_ref_language(raw_language: Optional[str]) -> str:
    if raw_language is None:
        return inference_main.get_default_language()

    lang = str(raw_language).strip()
    if lang in inference_main.get_supported_languages():
        return lang

    if lang in LEGACY_REF_LANGUAGE_ALIASES:
        candidate = LEGACY_REF_LANGUAGE_ALIASES[lang]
        if candidate in inference_main.get_supported_languages():
            return candidate

    return inference_main.normalize_ref_language(lang)


def reload_data(index: int, batch: int):
    global g_index, g_batch
    g_index = index
    g_batch = batch
    return g_ref_list[index:index + batch]


def change_index(index: int, batch: int):
    global g_index, g_batch, g_ref_audio_path_list

    g_ref_audio_path_list = []
    g_index, g_batch = index, batch
    datas = reload_data(index, batch)

    output = []

    # Reference audio widgets
    for item in datas:
        output.append(
            {
                "__type__": "update",
                "label": f"Ref Audio {os.path.basename(item['path'])}",
                "value": item["path"],
            }
        )
        g_ref_audio_path_list.append(item["path"])

    for _ in range(g_batch - len(datas)):
        output.append(
            {
                "__type__": "update",
                "label": "Ref Audio",
                "value": None,
            }
        )
        g_ref_audio_path_list.append(None)

    # Reference language widgets
    for item in datas:
        output.append(_normalize_ref_language(item["lang"]))
    for _ in range(g_batch - len(datas)):
        output.append(None)

    # Reference text widgets
    for item in datas:
        output.append(item["text"])
    for _ in range(g_batch - len(datas)):
        output.append(None)

    # Test audio widgets
    for _ in range(g_batch):
        output.append(None)

    # Save buttons
    for _ in datas:
        output.append({"__type__": "update", "value": "Save", "interactive": True})
    for _ in range(g_batch - len(datas)):
        output.append({"__type__": "update", "value": "Save", "interactive": False})

    return output


def previous_index(index: int, batch: int):
    if (index - batch) >= 0:
        new_index = index - batch
    else:
        new_index = 0
    return new_index, *change_index(new_index, batch)


def next_index(index: int, batch: int):
    if (index + batch) <= g_ref_list_max_index:
        new_index = index + batch
    else:
        new_index = index
    return new_index, *change_index(new_index, batch)


def copy_proved_ref_audio(index: int, text: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    filename = re.sub(r"[/\\:*?\"<>|]", "", text or "untitled")
    if not filename:
        filename = "untitled"

    try:
        source = g_ref_audio_path_list[int(index)]
        if not source:
            raise FileNotFoundError("Reference audio path is empty.")
        shutil.copy2(source, os.path.join(out_dir, f"{filename}.wav"))
        return {"__type__": "update", "value": "Saved", "interactive": False}
    except Exception as exc:
        print(exc)
        return {"__type__": "update", "value": "Save Failed", "interactive": True}


def _build_language_update(current_text_language: Optional[str]):
    languages = inference_main.get_supported_languages()
    if not languages:
        return {"__type__": "update", "choices": [], "value": None}

    value = current_text_language if current_text_language in languages else languages[0]
    return {"__type__": "update", "choices": languages, "value": value}


def _build_version_updates(version: str):
    is_v3 = version == "v3"
    sample_value = 32 if is_v3 else 8

    sample_steps_update = {
        "__type__": "update",
        "visible": is_v3,
        "interactive": is_v3,
        "value": sample_value,
    }

    super_sampling_update = {
        "__type__": "update",
        "visible": is_v3,
        "interactive": is_v3,
        "value": False,
    }

    return (
        f"SoVITS Version: {version}",
        sample_steps_update,
        super_sampling_update,
    )


def on_change_sovits_weights(sovits_path: str, current_text_language: str):
    try:
        version = inference_main.change_sovits_weights(sovits_path)
        language_update = _build_language_update(current_text_language)
        version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
        status = f"Loaded SoVITS: {os.path.basename(sovits_path)}"
        return (
            language_update,
            version_text,
            sample_steps_update,
            super_sampling_update,
            status,
        )
    except Exception as exc:
        print(exc)
        version = inference_main.get_current_model_version()
        language_update = _build_language_update(current_text_language)
        version_text, sample_steps_update, super_sampling_update = _build_version_updates(version)
        return (
            language_update,
            version_text,
            sample_steps_update,
            super_sampling_update,
            f"Failed to load SoVITS: {exc}",
        )


def on_change_gpt_weights(gpt_path: str):
    try:
        inference_main.change_gpt_weights(gpt_path)
        return f"Loaded GPT: {os.path.basename(gpt_path)}"
    except Exception as exc:
        print(exc)
        return f"Failed to load GPT: {exc}"


def generate_test_audio(
    test_text: str,
    text_language: str,
    how_to_cut: str,
    top_k: float,
    top_p: float,
    temperature: float,
    speed_factor: float,
    repetition_penalty: float,
    seed: float,
    sample_steps: float,
    super_sampling: bool,
    *widgets,
):
    output = []

    if not test_text or not test_text.strip():
        return [None for _ in range(g_batch)]

    cut_mode = CUT_LABEL_TO_VALUE.get(how_to_cut, 0)

    for i in range(g_batch):
        ref_audio_path = g_ref_audio_path_list[i]
        if not ref_audio_path:
            output.append(None)
            continue

        ref_lang = widgets[i]
        ref_text = widgets[i + g_batch]

        try:
            generated = inference_main.get_tts_wav(
                ref_audio_path,
                ref_text or "",
                ref_lang,
                test_text,
                text_language,
                how_to_cut=cut_mode,
                top_k=int(top_k),
                top_p=float(top_p),
                temperature=float(temperature),
                speed_factor=float(speed_factor),
                repetition_penalty=float(repetition_penalty),
                seed=int(seed),
                sample_steps=int(sample_steps),
                super_sampling=bool(super_sampling),
            )
            sample_rate, audio_array = next(generated)
            output.append((sample_rate, audio_array))
        except Exception as exc:
            print(f"Skip {ref_audio_path}: {exc}")
            output.append(None)

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--list", type=str, default="ref.list", help="Reference list file path")
    parser.add_argument("-p", "--port", type=int, default=14285, help="WebUI port")
    parser.add_argument(
        "-f",
        "--folder",
        type=str,
        default="",
        help="Optional folder for reference audio; basename in list will be joined to this folder.",
    )
    parser.add_argument("-b", "--batch", type=int, default=10, help="Batch size per page")
    parser.add_argument(
        "-cd",
        "--check_duration",
        action="store_true",
        default=False,
        help="Check whether each reference audio duration is between 3 and 10 seconds.",
    )
    parser.add_argument(
        "-r",
        "--random_order",
        action="store_true",
        default=False,
        help="Randomize the order of reference audios.",
    )
    args = parser.parse_args()

    g_ref_audio_widget_list = []
    g_ref_lang_widget_list = []
    g_ref_text_widget_list = []
    g_test_audio_widget_list = []
    g_save_widget_list = []

    g_ref_folder = args.folder
    g_batch = args.batch

    load_ref_list_file(args.list)

    if args.check_duration:
        remove_noncompliant_audio_from_list()

    if args.random_order:
        random.shuffle(g_ref_list)

    g_SoVITS_names, g_GPT_names = get_weights_names()

    if not g_GPT_names or not g_SoVITS_names:
        print("No model found. Put .pth files into SoVITS_weights* and .ckpt files into GPT_weights*.")
        raise SystemExit(1)

    current_version = inference_main.initialize(g_GPT_names[0], g_SoVITS_names[0])
    language_choices = inference_main.get_supported_languages()
    default_language = language_choices[0] if language_choices else inference_main.get_default_language()
    version_text = f"SoVITS Version: {current_version}"
    is_v3 = current_version == "v3"

    with gr.Blocks(title="GPT-SoVITS RefAudio Tester WebUI") as app:
        gr.Markdown("# GPT-SoVITS RefAudio Tester WebUI")
        gr.Markdown("Batch reference-audio preview with compatibility for GPT-SoVITS v1-v4.")

        with gr.Group():
            gr.Markdown("## Model Selection")
            with gr.Row():
                dropdownGPT = gr.Dropdown(label="GPT Model", choices=g_GPT_names, value=g_GPT_names[0], interactive=True)
                dropdownSoVITS = gr.Dropdown(
                    label="SoVITS Model", choices=g_SoVITS_names, value=g_SoVITS_names[0], interactive=True
                )
                textboxOutputFolder = gr.Textbox(label="Copy approved reference audio to", value="output/", interactive=True)
                btnRefresh = gr.Button("Refresh Models")

            with gr.Row():
                textboxModelVersion = gr.Textbox(label="Model Version", value=version_text, interactive=False)
                textboxStatus = gr.Textbox(label="Status", value="Ready", interactive=False)

            gr.Markdown("## Synthesis Options")
            with gr.Row():
                textboxTestText = gr.Textbox(
                    label="Preview Text",
                    interactive=True,
                    placeholder="Input text used for synthesis preview",
                )
                dropdownTextLanguage = gr.Dropdown(
                    label="Synthesis Language",
                    choices=language_choices,
                    value=default_language,
                    interactive=True,
                )
                dropdownHowToCut = gr.Dropdown(
                    label="Text Split Method",
                    choices=list(CUT_LABEL_TO_VALUE.keys()),
                    value="Every 4 Sentences",
                    interactive=True,
                )

            with gr.Row():
                sliderTopK = gr.Slider(minimum=1, maximum=100, step=1, label="top_k", value=20, interactive=True)
                sliderTopP = gr.Slider(minimum=0, maximum=1, step=0.01, label="top_p", value=0.6, interactive=True)
                sliderTemperature = gr.Slider(
                    minimum=0, maximum=1.5, step=0.01, label="temperature", value=0.6, interactive=True
                )
                sliderSpeedFactor = gr.Slider(
                    minimum=0.6, maximum=1.65, step=0.01, label="speed_factor", value=1.0, interactive=True
                )

            with gr.Row():
                sliderRepetitionPenalty = gr.Slider(
                    minimum=0.8,
                    maximum=2.0,
                    step=0.01,
                    label="repetition_penalty",
                    value=1.35,
                    interactive=True,
                )
                numberSeed = gr.Number(label="seed (-1 random)", value=-1, precision=0, interactive=True)
                sliderSampleSteps = gr.Slider(
                    minimum=4,
                    maximum=128,
                    step=4,
                    label="sample_steps (v3 only)",
                    value=32 if is_v3 else 8,
                    visible=is_v3,
                    interactive=is_v3,
                )
                checkboxSuperSampling = gr.Checkbox(
                    label="super_sampling (v3 only)",
                    value=False,
                    visible=is_v3,
                    interactive=is_v3,
                )

            gr.Markdown("## Batch Preview")
            with gr.Row():
                sliderStartIndex = gr.Slider(
                    minimum=0,
                    maximum=g_ref_list_max_index,
                    step=max(g_batch, 1),
                    label="Start Index",
                    value=0,
                    interactive=True,
                )
                sliderBatchSize = gr.Slider(
                    minimum=1,
                    maximum=100,
                    step=1,
                    label="Batch Size",
                    value=g_batch,
                    interactive=False,
                )
                btnPreBatch = gr.Button("Previous Batch")
                btnNextBatch = gr.Button("Next Batch")
                btnInference = gr.Button("Generate Preview Audio", variant="primary")

            gr.Markdown("## Preview List")
            with gr.Row():
                with gr.Column():
                    for i in range(g_batch):
                        with gr.Row():
                            ref_no = gr.Number(value=i, visible=False)
                            ref_audio = gr.Audio(label="Ref Audio", visible=True, scale=4)
                            ref_lang = gr.Textbox(label="Ref Language", visible=True, scale=1)
                            ref_text = gr.Textbox(label="Ref Text", visible=True, scale=4)
                            test_audio = gr.Audio(label="Generated", visible=True, scale=4)
                            save = gr.Button(value="Save", scale=1)

                            g_ref_audio_widget_list.append(ref_audio)
                            g_ref_lang_widget_list.append(ref_lang)
                            g_ref_text_widget_list.append(ref_text)
                            g_test_audio_widget_list.append(test_audio)
                            g_save_widget_list.append(save)

                            save.click(
                                copy_proved_ref_audio,
                                inputs=[ref_no, ref_text, textboxOutputFolder],
                                outputs=[save],
                            )

            btnRefresh.click(
                fn=refresh_model_list,
                inputs=[],
                outputs=[dropdownSoVITS, dropdownGPT, textboxStatus],
            )

            dropdownSoVITS.change(
                on_change_sovits_weights,
                inputs=[dropdownSoVITS, dropdownTextLanguage],
                outputs=[
                    dropdownTextLanguage,
                    textboxModelVersion,
                    sliderSampleSteps,
                    checkboxSuperSampling,
                    textboxStatus,
                ],
            )

            dropdownGPT.change(
                on_change_gpt_weights,
                inputs=[dropdownGPT],
                outputs=[textboxStatus],
            )

            sliderStartIndex.change(
                change_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                ],
            )

            btnPreBatch.click(
                previous_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    sliderStartIndex,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                ],
            )

            btnNextBatch.click(
                next_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    sliderStartIndex,
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                ],
            )

            btnInference.click(
                generate_test_audio,
                inputs=[
                    textboxTestText,
                    dropdownTextLanguage,
                    dropdownHowToCut,
                    sliderTopK,
                    sliderTopP,
                    sliderTemperature,
                    sliderSpeedFactor,
                    sliderRepetitionPenalty,
                    numberSeed,
                    sliderSampleSteps,
                    checkboxSuperSampling,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                ],
                outputs=[*g_test_audio_widget_list],
            )

            app.load(
                change_index,
                inputs=[sliderStartIndex, sliderBatchSize],
                outputs=[
                    *g_ref_audio_widget_list,
                    *g_ref_lang_widget_list,
                    *g_ref_text_widget_list,
                    *g_test_audio_widget_list,
                    *g_save_widget_list,
                ],
            )

    app.launch(
        server_name="0.0.0.0",
        inbrowser=True,
        quiet=True,
        share=False,
        server_port=args.port,
    )
