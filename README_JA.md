# GPT-SoVITS RefAudio Tester（v1-v4 対応）

[中文说明](README.md)  
[English Version](README_EN.md)

このリポジトリは、参照音声テスト用のバッチプレビュー WebUI ワークフローを維持しつつ、最新の GPT-SoVITS 推論コア（`TTS_infer_pack`）へ移行して、モデル互換性を向上させています。

## 更新内容

- 推論コアを旧来の直接モデル呼び出しから、上流の `TTS` パイプラインへ移行。
- 以下の SoVITS / GPT 系列に対応:
  - `v1`
  - `v2`
  - `v2Pro`
  - `v2ProPlus`
  - `v3`
  - `v4`
- バッチプレビューの流れを維持:
  - ページ単位でのリスト読み込み
  - バッチ単位でのプレビュー音声生成
  - 承認した参照音声のワンクリック保存

## モデルディレクトリ

重みファイルを以下いずれかのディレクトリに配置してください。

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

## 実行依存

上流と同期した依存ファイルを使用します。

- `requirements.txt`
- `extra-req.txt`（任意）

まず実行環境に合わせて PyTorch をインストールし、その後に残りの依存をインストールしてください。

```powershell
# NVIDIA GPU（CUDA 12.1）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU のみ
# pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
pip install -r extra-req.txt
```

注意: 本プロジェクトでは、GPT-SoVITS 関連の事前学習済みアセット（例: BERT/CN-HuBERT）が `GPT_SoVITS/pretrained_models` に配置されている前提です。

## 起動

```powershell
python webui.py -l ref.list -f <ref_audio_folder> -b 10
```

引数:

- `-l`, `--list`: 参照リストファイル（既定: `ref.list`）
- `-p`, `--port`: WebUI ポート（既定: `14285`）
- `-f`, `--folder`: リスト内参照音声の任意ベースフォルダ
- `-b`, `--batch`: ページあたりバッチサイズ
- `-cd`, `--check_duration`: 3〜10 秒以外の参照音声を除外
- `-r`, `--random_order`: 参照リスト順をランダム化

参照リスト形式:

```text
<wav_path_or_name>|<speaker>|<language>|<prompt_text>
```

`ZH/JP/EN` などの旧言語値は自動で正規化されます。

## WebUI の新しい合成制御

- UI 言語切替: 中文 / 日本語 / English
- 「合成言語」ドロップダウンは UI 言語に応じて表示名をローカライズ（推論時の内部値は不変）
- `top_k`
- `top_p`
- `temperature`
- `speed_factor`
- `repetition_penalty`
- `seed`
- `sample_steps`（`v3` で表示）
- `super_sampling`（`v3` で表示）

## 注意事項

- `v3/v4` モデルでは、行のプロンプトテキストが空の場合、その行が失敗する可能性があります。
- バッチ内の 1 行が失敗しても、他の行は継続します。
- WebUI は、読み込まれた SoVITS バージョンに応じて選択可能な合成言語を動的に更新します。
