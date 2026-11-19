<!--
Fill in every section. If an AI agent prepared this pull request, it lists only the commands
it actually ran, with their results, and says what it could not check (GPU, real downloads,
the packaged app).
-->

## Summary

<!-- What the change does and why, in a few sentences. -->

Closes #

## Changes

<!-- The main changes, grouped by area. Point out what a reviewer should look at closely:
data formats, files on disk, background workers, behavior users will notice. -->

-

## Testing

<!-- Commands run and their results, and what you checked by hand in the app (steps, base
model, GPU). Say what was not tested. -->

- [ ] `ruff check desktop_app.py irodori_cli.py irodori_tts scripts tests`
- [ ] Affected unit and Qt tests (see Checks in CONTRIBUTING.md):
- [ ] Opt-in GPU checks, if generation, transcription or training changed:
- [ ] Tried in the app:

## Screenshots

<!-- For UI changes: before and after, in light and dark themes, in English and Chinese if
the text changed. Delete this section otherwise. -->

## Checklist

- [ ] Computation code is changed in the `upstream/Irodori-TTS` fork, not copied here.
- [ ] New or changed UI text has keys in both `en.ts` and `zh_CN.ts`, and the `.qm` files are recompiled.
- [ ] Changes to pinned models come with an updated `model_pins.json` (`python -m scripts.pin_models`).
- [ ] Documentation describes the new behavior (README, README_zh, CONTRIBUTING).
- [ ] Data written by earlier versions still opens, or the summary says why not.
- [ ] No tokens, personal recordings or local paths in the code, tests or commits.
