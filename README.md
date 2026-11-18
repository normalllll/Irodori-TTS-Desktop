# Irodori-TTS-Desktop

[简体中文](README_zh.md)

A desktop app for Japanese speech generation and LoRA voice training, built with PySide6
and QFluentWidgets on a pinned [Irodori-TTS](https://github.com/Aratako/Irodori-TTS).

## Features

- Generate speech with a base model or a trained voice, from a voice description and
  reference recordings. Each model remembers the settings last used with it.
- Use several base models side by side: Irodori-TTS v4.1 Small, v4 Large, and base models
  imported from disk. A voice runs on the installed base it was trained for.
- Read long text in segments, and insert the model's emoji style tags from a palette.
- Generate with less video memory, or faster with a 4-step model.
- Import recordings without transcripts: one model writes them, a second one cross-checks
  the readings, and only the recordings where they disagree need review.
- Train a LoRA voice with settings recommended from your recordings and your GPU; stop,
  resume, train a finished voice further, and listen to samples while it trains.
- Keep a generation history and export audio, voices or whole projects.
- English and Simplified Chinese interface, light and dark themes.

## Requirements

Windows x64 with an NVIDIA GPU and a current driver. Speech input is Japanese.
Plan disk space for the models (about 3.5 GB for v4.1 Small, 13 GB more for v4 Large, plus
2–5 GB per transcription model) and for training checkpoints.

## Run the package

Extract `Irodori-TTS-Desktop.7z` (Windows 11 opens it directly; otherwise use 7-Zip) and run
`Irodori-TTS-Desktop.exe`. Keep the folder whole; the package brings its own Python and needs
nothing else installed.

## Run from source

```powershell
git submodule update --init --recursive
uv sync --python 3.10 --extra cu128
. ./local-env.ps1
.venv/Scripts/python.exe desktop_app.py
```

The first start opens **Getting started**, which checks the computer and downloads the
v4.1 Small base model and the recommended transcription models; a stopped download continues where
it ended. Without network access, use **Import model files** with a model package (.zip or
.7z) or a folder of model files, for example one exported on another computer
(**Settings → Export downloaded models**). Every file is checked before it is used. A short tour shows where things are;
**Settings → Start the tour again** repeats it.

1. Try the default voice on **Generate**.
2. On **Models**, create a **New model** and **Import recordings**; transcripts are written
   automatically. Recordings with CSV or TXT transcripts can be imported as well.
3. Check the recordings marked **Needs review**.
4. On the **Training** tab, take the recommended settings and start; preprocessing runs
   first when needed.
5. Use **Generate sample** on a saved checkpoint to listen while training, then generate
   and export with the finished voice.

Use recordings you have the right to use. The included examples are synthetic.

### Base models

A base model is identified by its architecture, the model shape its checkpoint records. A
trained voice runs on any installed base of the architecture it was trained on, so a newer
version of a base keeps every voice working; Generate notes when a voice was trained on
another version. A voice whose base is not installed names the base to download, or asks for
the base it was trained on to be imported.

- **Irodori-TTS v4.1 Small** (about 3.5 GB): the default, downloaded in **Getting started**.
- **Irodori-TTS v4 Large** (about 13 GB): optional; slower and needs much more video memory
  (training needs a 16 GB card or more). It is used under the
  [Gemma terms of use](https://huggingface.co/Aratako/Irodori-TTS-v4-Large/blob/main/GEMMA_TERMS_OF_USE.md).
- **Settings → Import base model** adds a base from a folder or a `.safetensors` file whose
  `tokenizer/` folder sits beside it; quantized checkpoints are accepted.

The catalog bases follow the latest commit of their Hugging Face repository; **Settings**
offers an update when a newer one is published. With two or more bases installed, Generate
lets the default voice choose its base; a new training run chooses the base it trains on and
keeps it.

**Settings → Generation model** chooses a variant for generating, for the bases whose variant
is downloaded in **Getting started**; the others use their standard model:

- **Less video memory** (Small about 0.9 GB, Large about 3.9 GB): about a quarter less video
  memory.
- **Fast (4 steps)** (Small only, about 3.1 GB): 4 sampling steps instead of 40; the model sets
  its own steps and guidance.

Trained voices work with their base's variants; training uses the base's standard model.

### Transcription models

Recordings longer than 12 seconds are cut at pauses.

| Model | Download / VRAM | Use it for |
|---|---|---|
| whisper-large-v3 (default) | 3.1 GB / ~8 GB | Most accurate readings; real and everyday speech |
| Qwen3-ASR-1.7B (default cross-check) | 4.7 GB / ~5 GB | Clean kanji and kana; a second opinion |
| anime-whisper | 3.0 GB / ~4 GB | Game and anime voices; writes many words in kana |
| Fun-ASR-Nano | 2.0 GB / ~4 GB | Smallest; misses more on game voices |

## Data

User data lives in the `data` folder next to `Irodori-TTS-Desktop.exe` (in a source checkout,
next to `desktop_app.py`). Use another location with `--data-dir path/to/data`. Keep a project's folder whole when moving or
backing it up.

## Project layout

`upstream/Irodori-TTS` is a submodule holding our
[fork](https://github.com/normalllll/Irodori-TTS) of the original project, where changes to the
computation code live. This repository holds only the desktop app; `irodori_cli.py infer` and
`irodori_cli.py train` run the fork's command-line tools. Development notes are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under GPL-3.0-only ([LICENSE](LICENSE)). It builds on
[Irodori-TTS](https://github.com/Aratako/Irodori-TTS) by Aratako, MIT licensed
(`upstream/Irodori-TTS/LICENSE`). Models and other dependencies keep their own licenses. The package carries FFmpeg, GPLv3
(`runtime/ffmpeg/LICENSE.txt`), and 7-Zip (`runtime/7zip/License.txt`).
