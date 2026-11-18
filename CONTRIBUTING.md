# Contributing

## Layout

- `upstream/Irodori-TTS`: [our fork](https://github.com/normalllll/Irodori-TTS) of the original
  project, a pinned submodule. Change computation code there, as commits that could each be a
  pull request to the original project, in its commit style. Do not copy its files here.
- `irodori_tts/__init__.py` adds the submodule's package and scripts to the import path. Do not
  add import hooks or monkey patches.
- To update: rebase the fork's `desktop-support` branch on the original project, then update the
  submodule and `UPSTREAM_COMMIT` in `irodori_tts/upstream.py`.
- Entry points: `desktop_app.py` (the app) and `irodori_cli.py infer|train` (command line).
- The root `.git` also holds the absorbed submodule metadata; keep it when rewriting history.

## Checks

```powershell
. ./local-env.ps1
.venv/Scripts/python.exe -m tests.run --timeout 180 unittest tests.test_desktop_core tests.test_desktop_qt tests.test_training_binding tests.test_desktop_i18n tests.test_training_preview tests.test_process_cleanup -v
.venv/Scripts/python.exe -m ruff check desktop_app.py irodori_cli.py irodori_tts scripts tests
```

`tests.run` runs each test process in its own Windows Job Object, so workers and their
children are always reaped, and a test that leaves processes behind fails. Qt tests use
offscreen windows and never take focus. Set `IRODORI_TEST_OUTPUT` to keep their logs.

GPU checks are opt-in and each wants a new data directory:

```powershell
.venv/Scripts/python.exe -m tests.desktop_integration --skip-prerequisite --data-dir outputs/new-e2e
.venv/Scripts/python.exe -m tests.desktop_training_preview --data-dir outputs/training-preview-check
.venv/Scripts/python.exe -m tests.desktop_asr --audio <recordings> --data-dir outputs/asr-acceptance
```

The transcription check needs reference data under `outputs/asr-bench/ref`.

## Models and add-ons

The codec, watermark and speech recognition model files are checked against
`irodori_tts/desktop/model_pins.json`. After changing a pinned revision or file list, run
`python -m scripts.pin_models`; `python -m scripts.pin_models --check --offline` fails when the
file is out of date.

Base speech models are not pinned. The catalog in `irodori_tts/desktop/base_models.py` names
their repositories, and the app downloads the latest commit and records it in the resource
folder's `base-models.json`. A base is identified by the architecture in its checkpoint's
`config_json` metadata, and voices are matched to bases by it. Keep the `tokenizer/` folder
beside every base checkpoint: without it the runtime fetches the tokenizer from the text
encoder's repository, which for v4 Large is gated.

Qwen3-ASR and Fun-ASR need packages that conflict with the app's own. They are installed as
add-ons under `<cache>/addons` and put in front of `PYTHONPATH` only in the worker that uses
them; the app environment is never changed.

## Building a package

```powershell
.venv/Scripts/python.exe -m scripts.build_desktop --archive
.venv/Scripts/python.exe -m scripts.verify_desktop
```

The build needs no network: it exports the pinned submodule, adds the application modules,
copies the Python environment, and copies the add-ons from `.cache/addons` (installing them
with `uv` only when they are missing there). Models are not included. It writes
`dist/Irodori-TTS-Desktop`, and with `--archive` also `dist/Irodori-TTS-Desktop.7z`.
`--refresh-app` replaces only the application files of an existing package.

FFmpeg is carried too: torchaudio cannot read or write audio without its libraries. Pass an
FFmpeg 8 or older win64 `shared` build zip once (`--ffmpeg <zip>`, for example from
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds/releases)); it unpacks into
`runtime/ffmpeg`, which later builds and source checkouts use. That build is GPLv3: when you
hand out a package, also offer the FFmpeg source of the same version.

7-Zip must be installed: the package carries its console program (`runtime/7zip`, with its
license) to import 7z model packages, and `--archive` compresses with it. Bytecode is compiled
during the build and shipped, so the first start does not compile. A package keeps the user's
data in `data/` beside the launcher, and downloads in `resources/`; neither goes into the
manifest or the archive, and a rebuild leaves `data/` alone.

`Irodori-TTS-Desktop.exe` is compiled from `scripts/launcher/launcher.c` with the app icon; it
starts the package's own Python and passes its arguments on. The build finds MSVC and the
Windows SDK through `vswhere` (Visual Studio's "Desktop development with C++"); the C runtime
is linked in, so users need nothing installed.

## Translations

UI text lives in `irodori_tts/desktop/locales/en.ts` and `zh_CN.ts`, under stable keys; every
key needs both languages. Keep user data (names, inputs, model IDs) out of the catalogs, and
switch language in place with the existing `bind`/`set_text` helpers.

```powershell
.venv/Scripts/python.exe -m scripts.translations --check
.venv/Scripts/python.exe -m scripts.translations
```

The second command recompiles the checked-in `.qm` files.
