# GPT-SoVITS RefAudio Tester（兼容 v1-v4）

[English Version](README_EN.md)
[日本語版](README_JA.md)

本仓库保留了用于参考音频测试的批量预览 WebUI 工作流，并已切换到最新的 GPT-SoVITS 推理核心（`TTS_infer_pack`），以提升模型兼容性。

## 更新内容

- 推理核心已从旧版直接调用模型迁移到上游 `TTS` 管线。
- 兼容以下 SoVITS / GPT 模型系列：
  - `v1`
  - `v2`
  - `v2Pro`
  - `v2ProPlus`
  - `v3`
  - `v4`
- 保留批量预览流程：
  - 分页加载列表
  - 批量生成预览音频
  - 一键保存通过筛选的参考音频

## 模型目录

请将权重放入以下任意目录：

- SoVITS（`.pth`）
  - `SoVITS_weights`
  - `SoVITS_weights_v2`
  - `SoVITS_weights_v2Pro`
  - `SoVITS_weights_v2ProPlus`
  - `SoVITS_weights_v3`
  - `SoVITS_weights_v4`
- GPT（`.ckpt`）
  - `GPT_weights`
  - `GPT_weights_v2`
  - `GPT_weights_v2Pro`
  - `GPT_weights_v2ProPlus`
  - `GPT_weights_v3`
  - `GPT_weights_v4`

## 运行依赖

使用与上游同步的依赖文件：

- `requirements.txt`
- `extra-req.txt`（可选）

请先根据你的运行环境安装 PyTorch，再安装其余依赖：

```powershell
# NVIDIA GPU（CUDA 12.1）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 仅 CPU
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -r extra-req.txt
```

说明：本项目需要自行准备 GPT-SoVITS 相关预训练资源，并放在 `GPT_SoVITS/pretrained_models` 下。

## 启动方式

```powershell
python webui.py -l ref.list -f <ref_audio_folder> -b 10
```

参数说明：

- `-l`, `--list`：参考列表文件（默认：`ref.list`）
- `-p`, `--port`：WebUI 端口（默认：`14285`）
- `-f`, `--folder`：列表文件中参考音频路径的可选基目录
- `-b`, `--batch`：分页批大小
- `-cd`, `--check_duration`：过滤不在 3-10 秒范围内的参考音频
- `-r`, `--random_order`：随机打乱参考列表顺序

参考列表格式：

```text
<wav_path_or_name>|<speaker>|<language>|<prompt_text>
```

列表中的旧语言标记（如 `ZH/JP/EN`）会自动规范化。

## WebUI 新增合成控制项

- 支持界面语言切换：中文 / 日本語 / English
- “合成语言”下拉框会随界面语言显示本地化名称（实际推理值保持不变）
- `top_k`
- `top_p`
- `temperature`
- `speed_factor`
- `repetition_penalty`
- `seed`
- `sample_steps`（`v3` 可见）
- `super_sampling`（`v3` 可见）

## 注意事项

- 对于 `v3/v4` 模型，若某行提示文本为空，可能会导致该行失败。
- 若批处理中某一行失败，其它行仍会继续执行。
- WebUI 会根据已加载的 SoVITS 版本动态更新可选合成语言。
