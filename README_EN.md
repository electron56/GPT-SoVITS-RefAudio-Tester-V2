# GPT-SoVITS RefAudio Tester (v1-v4 Compatible)

[中文说明](README.md)
[日本語版](README_JA.md)

This repository keeps a batch preview WebUI workflow for reference-audio testing, and now uses the latest GPT-SoVITS inference core (`TTS_infer_pack`) for model compatibility.

## What is upgraded

- Inference core migrated from legacy direct model calls to the upstream `TTS` pipeline.
- Compatible with SoVITS/GPT model families:
  - `v1`
  - `v2`
  - `v2Pro`
  - `v2ProPlus`
  - `v3`
  - `v4`
- Batch preview flow is preserved:
  - paged list loading
  - generate preview audio for a batch
  - one-click save approved references

## Model directories

Put your weights in any of the following directories:

- SoVITS (`.pth`)
  - `SoVITS_weights`
  - `SoVITS_weights_v2`
  - `SoVITS_weights_v2Pro`
  - `SoVITS_weights_v2ProPlus`
  - `SoVITS_weights_v3`
  - `SoVITS_weights_v4`
- GPT (`.ckpt`)
  - `GPT_weights`
  - `GPT_weights_v2`
  - `GPT_weights_v2Pro`
  - `GPT_weights_v2ProPlus`
  - `GPT_weights_v3`
  - `GPT_weights_v4`

## Runtime dependencies

Use the synced upstream dependency files:

- `requirements.txt`
- `extra-req.txt` (optional)

Install PyTorch first according to your runtime target, then install the rest:

```powershell
# NVIDIA GPU (CUDA 12.1)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -r extra-req.txt
```

Note: this project assumes GPT-SoVITS related pretrained assets (for example BERT/CN-HuBERT) are available under `GPT_SoVITS/pretrained_models`.

## Run

```powershell
python webui.py -l ref.list -f <ref_audio_folder> -b 10
```

Arguments:

- `-l`, `--list`: reference list file (default: `ref.list`)
- `-p`, `--port`: WebUI port (default: `14285`)
- `-f`, `--folder`: optional base folder for reference audios in list file
- `-b`, `--batch`: page batch size
- `-cd`, `--check_duration`: filter refs outside 3-10 seconds
- `-r`, `--random_order`: randomize reference list order

Reference list format:

```text
<wav_path_or_name>|<speaker>|<language>|<prompt_text>
```

Legacy language values in list files such as `ZH/JP/EN` are automatically normalized.

## New synthesis controls in WebUI

- UI language switching: Chinese / Japanese / English
- The "Synthesis Language" dropdown displays localized labels based on UI language (inference values stay unchanged).
- `top_k`
- `top_p`
- `temperature`
- `speed_factor`
- `repetition_penalty`
- `seed`
- `sample_steps` (visible for `v3`)
- `super_sampling` (visible for `v3`)

## Notes

- For `v3/v4` models, empty prompt text may fail for that row.
- If one row fails in a batch, other rows continue.
- The WebUI dynamically updates synthesis language choices according to loaded SoVITS version.
