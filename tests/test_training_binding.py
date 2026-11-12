"""Execution-gate tests; fixtures validate metadata, never stand in for GPU training."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf
import torch

from irodori_tts.config import ModelConfig, TrainConfig
from irodori_tts.desktop import INITIALIZATION_RULES_VERSION
from irodori_tts.desktop.materials import (
    edit_material,
    import_materials,
    materials_fingerprint,
)
from irodori_tts.desktop.messages import BusinessError
from irodori_tts.desktop.resources import CODEC_REVISION
from irodori_tts.desktop.storage import (
    create_project,
    digest,
    identity,
    load_project,
    now,
    read_json,
    save_project,
    write_json,
)
from irodori_tts.desktop.training import (
    compact_checkpoints,
    complete_checkpoint,
    create_run,
    delete_run,
    initialization_current,
    latest_checkpoint,
    run_command,
    validate_initialization,
)
from irodori_tts.desktop.worker import initialize, train_run
from irodori_tts.task_control import TaskCancelled, TaskChannel
from tests.test_desktop_core import AdapterFixture

REPO = Path(__file__).resolve().parents[1]


class TrainingBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = create_project(Path(self.temp.name), "binding")
        import_materials(
            self.root, REPO / "examples/desktop", REPO / "examples/desktop/transcripts.csv"
        )
        self.publish_fixture_initialization()

    def tearDown(self):
        self.temp.cleanup()

    def publish_fixture_initialization(self):
        project = load_project(self.root)
        initialized = self.root / "initialized"
        (initialized / "latents").mkdir(parents=True, exist_ok=True)
        rows = []
        for item in project["materials"]:
            latent = initialized / "latents" / (item["id"] + ".pt")
            torch.save(torch.zeros(2, 32), latent)
            rows.append(
                {"text": item["text"], "num_frames": 2, "latent_path": "latents/" + latent.name}
            )
        (initialized / "manifest.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
        checksums = {
            path.relative_to(initialized).as_posix(): digest(path)
            for path in initialized.rglob("*")
            if path.is_file() and path.name != "initialization.json"
        }
        record = {
            "id": identity(),
            "revision": project["revision"],
            "status": "completed",
            "codec_revision": CODEC_REVISION,
            "rules_version": INITIALIZATION_RULES_VERSION,
            "count": len(rows),
            "created": now(),
            "checksums": checksums,
        }
        write_json(initialized / "initialization.json", record)
        project["initialization"] = record
        save_project(self.root, project)

    def make_run(self, resume, values=None):
        metadata = Mock()
        metadata.__enter__ = Mock(return_value=metadata)
        metadata.__exit__ = Mock(return_value=False)
        metadata.metadata.return_value = {"config_json": json.dumps(asdict(ModelConfig()))}
        with (
            patch(
                "irodori_tts.desktop.resources.resolve_resources",
                return_value={"model": "metadata-fixture"},
            ),
            patch("safetensors.safe_open", return_value=metadata),
        ):
            directory = create_run(
                self.root, "snapshot", {"max_steps": 4} if values is None else values, REPO
            )
        if resume:
            import train

            model = AdapterFixture()
            checkpoint = directory / "checkpoints/checkpoint_0000001"
            train.save_checkpoint(
                checkpoint,
                model,
                torch.optim.AdamW(model.parameters()),
                None,
                1,
                ModelConfig(),
                TrainConfig(lora_enabled=True),
            )
            run = read_json(directory / "run.json")
            run.update(
                step=1, status="stopped", checkpoint=checkpoint.relative_to(self.root).as_posix()
            )
            write_json(directory / "run.json", run)
        return directory

    def test_new_run_persists_the_selected_budget_and_schedule(self):
        directory = self.make_run(
            False,
            {
                "max_steps": 13,
                "batch_size": 2,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
            },
        )
        record = read_json(directory / "run.json")
        config = read_json(directory / "config.json")
        self.assertEqual(config, record["config"])
        self.assertEqual(
            tuple(config["train"][key] for key in ("max_steps", "warmup_steps", "save_every")),
            (13, 1, 3),
        )

    def assert_rejected_before_execution(self, directory, resume):
        before = read_json(directory / "run.json")
        with patch("train.main") as main, patch("torch.cuda.reset_peak_memory_stats") as gpu:
            with self.assertRaises(ValueError):
                train_run(
                    {"project": str(self.root), "run": str(directory), "resume": resume},
                    TaskChannel(),
                    {"model": "must-not-load"},
                )
            main.assert_not_called()
            gpu.assert_not_called()
        after = read_json(directory / "run.json")
        self.assertEqual(after["step"], before["step"])
        self.assertEqual(after["checkpoint"], before["checkpoint"])
        self.assertEqual(after["status"], "failed")
        self.assertTrue(after["error"])

    def runs_before_mutation(self):
        return [(resume, self.make_run(resume)) for resume in (False, True)]

    def test_a_second_run_with_the_same_name_gets_a_number(self):
        from irodori_tts.desktop.training import unique_run_name

        self.make_run(False)
        self.assertEqual(unique_run_name(self.root, "snapshot"), "snapshot 2")
        second = self.make_run(False)
        self.assertEqual(read_json(second / "run.json")["name"], "snapshot 2")
        self.assertEqual(unique_run_name(self.root, " other "), "other")

    def test_a_finished_run_trains_on_without_losing_its_end(self):
        import shutil

        from irodori_tts.desktop.training import extend_run, resume_eligibility

        directory = self.make_run(True)
        checkpoints = directory / "checkpoints"
        final = checkpoints / "checkpoint_final"
        (checkpoints / "checkpoint_0000001").rename(final)
        # The trainer's twin of the end: the same weights without trainer state.
        twin = checkpoints / "checkpoint_0000001"
        shutil.copytree(final, twin)
        sample = directory / "samples/s1/generation.json"
        write_json(sample, {"audio": "audio.wav", "request": {"lora_adapter": str(final)}})
        run = read_json(directory / "run.json")
        run.update(step=4, status="completed", checkpoint=final.relative_to(self.root).as_posix())
        write_json(directory / "run.json", run)
        with self.assertRaises(BusinessError):
            extend_run(self.root, directory, 4, 2)
        extend_run(self.root, directory, 10, 3)
        run = read_json(directory / "run.json")
        named = checkpoints / "checkpoint_0000004"
        self.assertFalse(final.exists())
        self.assertTrue(complete_checkpoint(named, require_trainer=True))
        self.assertEqual(run["checkpoint"], named.relative_to(self.root).as_posix())
        self.assertEqual(read_json(sample)["request"]["lora_adapter"], str(named))
        self.assertEqual(run["status"], "stopped")
        self.assertEqual(read_json(directory / "config.json")["train"]["max_steps"], 10)
        self.assertEqual(run["config"]["train"]["save_every"], 3)
        resume_eligibility(self.root, run)
        self.assertIn("--resume", run_command(self.root, directory, {"model": "m"}, resume=True))

    def test_material_edit_between_create_run_and_worker_lock_rejects_both_modes(self):
        runs = self.runs_before_mutation()
        item = load_project(self.root)["materials"][0]
        edit_material(
            self.root, item["id"], text="こんにちは。変更した転写です。", caption="", included=True
        )
        for resume, directory in runs:
            with self.subTest(resume=resume):
                self.assert_rejected_before_execution(directory, resume)

    def test_rebuild_between_create_run_and_worker_lock_rejects_both_modes(self):
        runs = self.runs_before_mutation()
        self.publish_fixture_initialization()
        for resume, directory in runs:
            with self.subTest(resume=resume):
                self.assert_rejected_before_execution(directory, resume)

    def test_external_audio_change_is_reconciled_before_gate(self):
        directory = self.make_run(False)
        project = load_project(self.root)
        with (self.root / project["materials"][0]["path"]).open("ab") as stream:
            stream.write(b"external change")
        self.assert_rejected_before_execution(directory, False)
        self.assertEqual(load_project(self.root)["revision"], project["revision"] + 1)

    def test_run_revision_mismatch_rejects_even_when_initialization_id_matches(self):
        directory = self.make_run(False)
        run = read_json(directory / "run.json")
        run["revision"] -= 1
        write_json(directory / "run.json", run)
        self.assert_rejected_before_execution(directory, False)

    def test_corrupt_artifact_rejects_before_model_load(self):
        directory = self.make_run(False)
        with (self.root / "initialized/manifest.jsonl").open("ab") as stream:
            stream.write(b"broken")
        self.assert_rejected_before_execution(directory, False)

    def test_record_revision_mismatch_rejects(self):
        directory = self.make_run(False)
        record = read_json(self.root / "initialized/initialization.json")
        record["revision"] += 1
        write_json(self.root / "initialized/initialization.json", record)
        self.assert_rejected_before_execution(directory, False)

    def test_unchanged_bindings_build_new_and_resume_commands(self):
        for resume, directory in self.runs_before_mutation():
            with self.subTest(resume=resume):
                command = run_command(
                    self.root, directory, {"model": "resolved-base"}, resume=resume
                )
                self.assertEqual("--resume" in command, resume)
                self.assertIn(str(self.root / "initialized/manifest.jsonl"), command)


def tone(path, seconds, amplitude=0.3, rate=24000):
    samples = np.arange(int(seconds * rate)) / rate
    sf.write(path, (amplitude * np.sin(2 * np.pi * 220 * samples)).astype("float32"), rate)


class MaterialImportTests(unittest.TestCase):
    """Import, edit and staleness rules for the recordings of a project."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.root = create_project(self.base / "projects", "materials")
        self.folder = self.base / "batch 1"
        self.folder.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def items(self):
        return {item["file_name"]: item for item in load_project(self.root)["materials"]}

    def test_csv_short_rows_and_legacy_none_text_never_break_the_page(self):
        tone(self.folder / "a.wav", 1)
        tone(self.folder / "b.wav", 1)
        # Excel on Japanese Windows writes cp932; "b.wav" has no text cell at all.
        (self.folder / "list.csv").write_bytes(
            "file_name,text,caption\na.wav, こんにちは。 \nb.wav\n".encode("cp932")
        )
        import_materials(self.root, self.folder, self.folder / "list.csv")
        items = self.items()
        self.assertEqual(items["a.wav"]["text"], "こんにちは。")
        self.assertEqual(items["a.wav"]["caption"], "")
        self.assertEqual(items["b.wav"]["text"], "")
        self.assertTrue(items["a.wav"]["included"])
        self.assertFalse(items["b.wav"]["included"])
        from irodori_tts.desktop.training_page import problem_of

        self.assertIsNotNone(problem_of({**items["a.wav"], "text": None, "caption": None}))
        project = load_project(self.root)
        project["materials"][0]["text"] = None  # written by an older version
        save_project(self.root, project)
        edit_material(self.root, project["materials"][0]["id"], text="", caption="", included=False)

    def test_transcript_encodings_fall_back_and_name_the_unreadable_file(self):
        tone(self.folder / "a.wav", 1)
        (self.folder / "a.txt").write_bytes("おはようございます。".encode("cp932"))
        import_materials(self.root, self.folder)
        self.assertEqual(self.items()["a.wav"]["text"], "おはようございます。")
        (self.folder / "a.txt").write_bytes(b"\x82\xff\xfe")  # invalid in all three
        with self.assertRaises(BusinessError) as raised:
            import_materials(self.root, self.folder)
        self.assertIn("a.txt", str(raised.exception))

    def test_problem_recordings_are_excluded_at_import_and_return_when_fixed(self):
        tone(self.folder / "long.wav", 13)
        tone(self.folder / "silent.wav", 1, amplitude=0)
        tone(self.folder / "untranscribed.wav", 1)
        tone(self.folder / "good.wav", 1)
        for name in ("long", "silent", "good"):
            (self.folder / f"{name}.txt").write_text("こんにちは。", encoding="utf-8")
        imported = import_materials(self.root, self.folder)
        self.assertEqual(
            {item["file_name"] for item in imported if not item["included"]},
            {"long.wav", "silent.wav", "untranscribed.wav"},
        )
        (self.folder / "untranscribed.txt").write_text("こんばんは。", encoding="utf-8")
        import_materials(self.root, self.folder)
        self.assertTrue(self.items()["untranscribed.wav"]["included"])
        self.assertFalse(self.items()["long.wav"]["included"])

    def test_writing_the_missing_text_in_the_app_includes_the_recording(self):
        tone(self.folder / "a.wav", 1)
        tone(self.folder / "b.wav", 1)
        import_materials(self.root, self.folder)
        items = self.items()
        self.assertFalse(items["a.wav"]["included"])
        edit_material(self.root, items["a.wav"]["id"], text="はい。", caption="", included=False)
        self.assertTrue(self.items()["a.wav"]["included"])
        # A recording the user excluded on purpose stays excluded after a text edit.
        edit_material(self.root, items["b.wav"]["id"], text="", caption="", included=True)
        edit_material(self.root, items["b.wav"]["id"], text="いいえ。", caption="", included=False)
        self.assertFalse(self.items()["b.wav"]["included"])

    def test_reimport_keeps_edits_and_same_name_from_another_folder_is_kept(self):
        tone(self.folder / "a.wav", 1)
        (self.folder / "a.txt").write_text("元の文。", encoding="utf-8")
        first = import_materials(self.root, self.folder)[0]
        edit_material(self.root, first["id"], text="直した文。", caption="明るい", included=False)
        revision = load_project(self.root)["revision"]
        (self.folder / "a.txt").unlink()
        import_materials(self.root, self.folder)
        kept = self.items()["a.wav"]
        self.assertEqual(
            (kept["text"], kept["caption"], kept["included"]), ("直した文。", "明るい", False)
        )
        self.assertEqual(load_project(self.root)["revision"], revision)
        other = self.base / "batch 2"
        other.mkdir()
        tone(other / "a.wav", 2)
        (other / "a.txt").write_text("別の録音。", encoding="utf-8")
        import_materials(self.root, other)
        items = self.items()
        self.assertEqual(set(items), {"a.wav", "a (2).wav"})
        self.assertEqual(items["a.wav"]["text"], "直した文。")
        self.assertEqual(items["a (2).wav"]["text"], "別の録音。")
        self.assertNotEqual(items["a.wav"]["path"], items["a (2).wav"]["path"])

    def test_noop_edits_keep_revision_and_undo_restores_the_initialization(self):
        tone(self.folder / "a.wav", 1)
        (self.folder / "a.txt").write_text("こんにちは。", encoding="utf-8")
        item = import_materials(self.root, self.folder)[0]
        project = load_project(self.root)
        project["initialization"] = {
            "id": "init",
            "revision": project["revision"],
            "fingerprint": materials_fingerprint(project),
            "status": "completed",
        }
        save_project(self.root, project)
        revision = project["revision"]
        edit_material(self.root, item["id"], text=" こんにちは。\n", caption=" ", included=True)
        edit_material(
            self.root,
            item["id"],
            text="こんにちは。",
            caption="",
            included=True,
            replacement=self.folder / "a.wav",
        )
        self.assertEqual(load_project(self.root)["revision"], revision)
        edit_material(self.root, item["id"], text="こんにちは。", caption="", included=False)
        self.assertFalse(initialization_current(load_project(self.root)))
        edit_material(self.root, item["id"], text="こんにちは。", caption="", included=True)
        self.assertTrue(initialization_current(load_project(self.root)))


class ReinitializationTests(unittest.TestCase):
    """A rebuild replaces the previous data only once the new data is complete."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = create_project(Path(self.temp.name), "rebuild")
        import_materials(
            self.root, REPO / "examples/desktop", REPO / "examples/desktop/transcripts.csv"
        )

    def tearDown(self):
        self.temp.cleanup()

    @staticmethod
    def encode(items, *, output_manifest, latent_dir, **_):
        latent_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for item in items:
            torch.save(torch.zeros(2, 32), latent_dir / f"{item['id']}.pt")
            rows.append(
                {
                    "text": item["text"],
                    "caption": item["caption"],
                    "speaker_id": item["speaker_id"],
                    "latent_path": f"latents/{item['id']}.pt",
                    "num_frames": 2,
                }
            )
        output_manifest.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )

    def initialize(self, side_effect=None):
        with patch(
            "irodori_tts.desktop.manifest.prepare_local_manifest",
            side_effect=side_effect or self.encode,
        ):
            return initialize(
                {"project": str(self.root), "rebuild": True}, TaskChannel(), {"codec": "fixture"}
            )

    def test_cancelled_or_failed_rebuild_keeps_previous_data_and_runs_resumable(self):
        first = self.initialize()
        validate_initialization(self.root)
        manifest = (self.root / "initialized/manifest.jsonl").read_bytes()
        run = self.root / "runs/r/run.json"
        write_json(run, {"id": "r", "initialization_id": first["id"], "status": "stopped"})
        for failure in (TaskCancelled(), OSError("disk full")):
            with self.subTest(failure=type(failure).__name__):
                with self.assertRaises(type(failure)):
                    self.initialize(side_effect=failure)
                project = load_project(self.root)
                self.assertEqual(project["initialization"]["id"], first["id"])
                self.assertTrue(initialization_current(project))
                self.assertEqual((self.root / "initialized/manifest.jsonl").read_bytes(), manifest)
                self.assertEqual(validate_initialization(self.root)["id"], first["id"])
                self.assertNotIn("resume_blocked", read_json(run))
                self.assertFalse((self.root / "temp/initialization").exists())
        second = self.initialize()
        self.assertNotEqual(second["id"], first["id"])
        self.assertEqual(validate_initialization(self.root)["id"], second["id"])
        self.assertTrue(read_json(run)["resume_blocked"])
        self.assertFalse((self.root / "temp/initialization.previous").exists())
        self.assertFalse((self.root / "initialized/audio").exists())
        rows = (self.root / "initialized/manifest.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertTrue(all(json.loads(row)["speaker_id"] for row in rows))

    def test_first_failed_initialization_stays_visible(self):
        with self.assertRaises(OSError):
            self.initialize(side_effect=OSError("disk full"))
        init = load_project(self.root)["initialization"]
        self.assertEqual(init["status"], "failed")
        self.assertFalse(initialization_current(load_project(self.root)))


class CheckpointRetentionTests(unittest.TestCase):
    def test_older_checkpoints_keep_adapters_but_only_the_newest_can_resume(self):
        from tests.test_training_preview import checkpoint

        with tempfile.TemporaryDirectory() as directory:
            run_root = Path(directory)
            for step in (2, 4, 6):
                checkpoint(run_root / f"checkpoints/checkpoint_{step:07d}", step)
            latest, step = latest_checkpoint(run_root)
            compact_checkpoints(run_root, latest)
            self.assertEqual(step, 6)
            for path in (run_root / "checkpoints").iterdir():
                self.assertTrue(complete_checkpoint(path))
                self.assertEqual(complete_checkpoint(path, require_trainer=True), path == latest)
            self.assertEqual(latest_checkpoint(run_root), (latest, 6))

    def test_delete_run_removes_only_a_training_record_inside_the_project(self):
        with tempfile.TemporaryDirectory() as directory:
            root = create_project(Path(directory), "delete")
            write_json(root / "runs/r/run.json", {"id": "r"})
            with self.assertRaises(BusinessError):
                delete_run(root, root / "runs")
            delete_run(root, root / "runs/r")
            self.assertFalse((root / "runs/r").exists())
            self.assertTrue((root / "project.json").exists())

    def test_deleting_a_run_waits_for_a_sample_being_let_go_of(self):
        import os
        import threading

        from irodori_tts.desktop import storage

        if os.name != "nt":
            self.skipTest("only Windows refuses to delete an open file")
        with tempfile.TemporaryDirectory() as directory:
            root = create_project(Path(directory), "delete")
            write_json(root / "runs/r/run.json", {"id": "r"})
            sample = root / "runs/r/samples/s/audio.wav"
            sample.parent.mkdir(parents=True)
            sample.write_bytes(b"x")
            # A player that lets go of the sample a moment after the delete starts.
            held = sample.open("rb")
            threading.Timer(0.4, held.close).start()
            delete_run(root, root / "runs/r")
            self.assertFalse((root / "runs/r").exists())
            # Held for good: the record stays listed and the delete says what is left.
            write_json(root / "runs/r/run.json", {"id": "r"})
            sample.parent.mkdir(parents=True)
            sample.write_bytes(b"x")
            with sample.open("rb"), patch.object(storage, "RELEASE_SECONDS", 0.2):
                with self.assertRaises(BusinessError):
                    delete_run(root, root / "runs/r")
            self.assertTrue((root / "runs/r/run.json").is_file())


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(lambda: unittest.main()))
