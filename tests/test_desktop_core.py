from __future__ import annotations

import copy
import csv
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch
import train
from torchdata.stateful_dataloader import StatefulDataLoader

from irodori_tts.checkpoint_io import capture_rng, checkpoint_target, preserve_rng, restore_rng
from irodori_tts.config import ModelConfig, TrainConfig, load_config_yaml
from irodori_tts.desktop import asr
from irodori_tts.desktop.exports import export_audio_file, export_project, import_project, redact
from irodori_tts.desktop.materials import (
    edit_material,
    import_materials,
    mark_reviewed,
    materials_fingerprint,
)
from irodori_tts.desktop.messages import BusinessError
from irodori_tts.desktop.splitting import split_points
from irodori_tts.desktop.storage import (
    clean_project,
    create_project,
    default_settings,
    digest,
    load_project,
    project_lock,
    read_json,
    reconcile_sources,
    save_project,
    within,
    write_json,
)
from irodori_tts.desktop.training import (
    complete_checkpoint,
    normalize_preset,
    write_trainer_config,
)
from irodori_tts.task_control import TaskCancelled


class AdapterFixture(torch.nn.Linear):
    """A serialization fixture, never counted as real training acceptance."""

    def __init__(self):
        super().__init__(2, 2)

    def save_pretrained(self, path):
        (path / "adapter_config.json").write_text("{}")
        torch.save(self.state_dict(), path / "adapter_model.bin")


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_shared_lora_save_failure_retains_previous(self):
        target = self.root / "checkpoint"
        model = AdapterFixture()
        optimizer = torch.optim.AdamW(model.parameters())
        cfg = TrainConfig(lora_enabled=True)
        train.save_checkpoint(target, model, optimizer, None, 3, ModelConfig(), cfg)
        original = digest(target / "trainer_state.pt")
        original_save = torch.save

        def fail_trainer(value, path, *args, **kwargs):
            if Path(path).name == "trainer_state.pt":
                Path(path).write_bytes(b"partial")
                raise OSError("simulated full disk")
            return original_save(value, path, *args, **kwargs)

        with patch("train.torch.save", side_effect=fail_trainer), self.assertRaises(OSError):
            train.save_checkpoint(target, model, optimizer, None, 4, ModelConfig(), cfg)
        self.assertEqual(digest(target / "trainer_state.pt"), original)
        self.assertEqual(train._load_checkpoint_payload(target, map_location="cpu")["step"], 3)
        self.assertTrue(complete_checkpoint(target, require_trainer=True))
        self.assertFalse(list(self.root.glob("*.partial")))

    def test_best_checkpoint_stage_stays_compact_on_deep_windows_paths(self):
        parent = self.root
        while len(str(parent)) < 168:
            parent /= "nested"
        target = parent / "checkpoint_best_val_loss_0000002_0.874315"
        old_stage = target.with_name(f".{target.name}.{'0' * 32}.partial")
        self.assertGreaterEqual(len(str(old_stage / "irodori_lora_metadata.json")), 260)
        with checkpoint_target(target, directory=True, step=2) as stage:
            self.assertLess(len(str(stage / "irodori_lora_metadata.json")), 260)
            self.assertNotIn(target.name, stage.name)
            (stage / "irodori_lora_metadata.json").write_text("{}", encoding="utf-8")
        self.assertEqual((target / "irodori_lora_metadata.json").read_text(), "{}")
        self.assertEqual(json.loads((target / "complete.json").read_text())["step"], 2)

    def test_publish_failure_restores_old_directory_and_crash_gap_recovers(self):
        target = self.root / "checkpoint"
        with checkpoint_target(target, directory=True, step=1) as stage:
            (stage / "state").write_bytes(b"old")
        replace = Path.replace

        def fail_publish(source, destination):
            if str(source).endswith(".partial"):
                raise PermissionError("simulated directory sharing violation")
            return replace(source, destination)

        with patch.object(Path, "replace", fail_publish), self.assertRaises(PermissionError):
            with checkpoint_target(target, directory=True, step=2) as stage:
                (stage / "state").write_bytes(b"new")
        self.assertEqual((target / "state").read_bytes(), b"old")
        with checkpoint_target(target, directory=True, step=2) as stage:
            (stage / "state").write_bytes(b"new")
        self.assertEqual((self.root / ".checkpoint.previous/state").read_bytes(), b"old")

    def test_adapter_remains_usable_without_training_state(self):
        target = self.root / "checkpoint"
        model = AdapterFixture()
        train.save_checkpoint(
            target,
            model,
            torch.optim.AdamW(model.parameters()),
            None,
            1,
            ModelConfig(),
            TrainConfig(lora_enabled=True),
        )
        (target / "trainer_state.pt").unlink()
        self.assertTrue(complete_checkpoint(target))
        self.assertFalse(complete_checkpoint(target, require_trainer=True))

    def test_rng_roundtrip_weights_only_and_validation_isolation(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        state = capture_rng()

        def draws():
            return random.random(), float(np.random.rand()), torch.rand(4)

        expected = draws()
        torch.save(state, self.root / "rng.pt")
        restore_rng(torch.load(self.root / "rng.pt", weights_only=True))
        actual = draws()
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))
        state = capture_rng()
        with preserve_rng():
            draws()
        actual = draws()
        restore_rng(state)
        expected = draws()
        self.assertEqual(expected[:2], actual[:2])
        self.assertTrue(torch.equal(expected[2], actual[2]))

    def test_sampler_state_and_global_rng_continue_together(self):
        class StochasticDataset:
            def __len__(self):
                return 10

            def __getitem__(self, index):
                return index, random.random(), torch.rand(1)

        random.seed(7)
        torch.manual_seed(7)
        loader = StatefulDataLoader(
            StochasticDataset(),
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            num_workers=0,
        )
        iterator = iter(loader)
        next(iterator)
        payload = train._collect_dataloader_state(loader, distributed=False, rank=0, world_size=1)
        expected = next(iterator)
        loader2 = StatefulDataLoader(
            StochasticDataset(),
            batch_size=1,
            shuffle=True,
            generator=torch.Generator().manual_seed(7),
            num_workers=0,
        )
        loader2.load_state_dict(payload["rank_states"][0])
        restore_rng(payload["rng_states"][0])
        actual = next(iter(loader2))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual, strict=True)))

    def test_json_exponent_roundtrip_and_small_dataset_preset(self):
        write_json(self.root / "config.json", {"train": {"learning_rate": 1e-5}})
        self.assertIsInstance(
            load_config_yaml(write_trainer_config(self.root))["train"]["learning_rate"], float
        )
        one = normalize_preset({"max_steps": 3}, 1)
        self.assertEqual(one["valid_ratio"], 0)
        self.assertEqual(one["valid_every"], 0)
        self.assertLess(one["warmup_steps"], one["max_steps"])
        a = train.split_train_valid_indices(num_samples=2, valid_ratio=0.05, seed=0)
        b = train.split_train_valid_indices(num_samples=2, valid_ratio=0.05, seed=0)
        self.assertEqual([len(x) for x in a], [1, 1])
        self.assertTrue(all(torch.equal(x, y) for x, y in zip(a, b, strict=True)))

    def test_manual_training_validation_is_explicit(self):
        config = normalize_preset({"max_steps": 100}, 152)
        self.assertEqual((config["valid_every"], config["valid_ratio"]), (0, 0))
        enabled = normalize_preset({"max_steps": 100, "valid_every": 10}, 152)
        self.assertEqual((enabled["valid_every"], enabled["valid_ratio"]), (10, 0.05))
        disabled = normalize_preset({"max_steps": 100, "valid_every": 0, "batch_size": 2}, 2)
        self.assertEqual((disabled["valid_ratio"], disabled["valid_every"]), (0, 0))

    def test_normalization_does_not_generate_a_count_only_budget(self):
        from irodori_tts.desktop.messages import BusinessError

        for count in (1, 33, 152, 1000):
            with self.subTest(count=count), self.assertRaises(BusinessError):
                normalize_preset({}, count)

    def test_budget_schedule_respects_explicit_values(self):
        one = normalize_preset({"max_steps": 1}, 1)
        self.assertEqual((one["warmup_steps"], one["save_every"]), (0, 1))
        custom = normalize_preset({"max_steps": 950, "warmup_steps": 0, "save_every": 17}, 152)
        self.assertEqual((custom["warmup_steps"], custom["save_every"]), (0, 17))
        for steps in (13, 207, 313, 950, 6251):
            config = normalize_preset({"max_steps": steps}, 152)
            self.assertEqual(config["max_steps"], steps)
            self.assertLess(config["warmup_steps"], steps)
            self.assertGreaterEqual(config["save_every"], 1)
        # Neither a changed dataset count nor a changed batch rewrites an explicit budget.
        for count, batch in ((2, 1), (19, 4), (152, 16)):
            config = normalize_preset({"max_steps": 317, "batch_size": batch}, count)
            self.assertEqual(config["max_steps"], 317)

    def test_a_run_needs_its_settings_filled_in(self):
        from irodori_tts.desktop.messages import BusinessError

        self.assertFalse(normalize_preset({"max_steps": 100}, 152)["length_bucket_enabled"])
        for empty in (
            {"batch_size": 0},
            {"max_steps": 0},
            {"gradient_accumulation_steps": 0},
            {"gradient_checkpointing": None},
        ):
            with self.assertRaises(BusinessError):
                normalize_preset({"max_steps": 100, **empty}, 152)
        self.assertFalse(
            normalize_preset({"max_steps": 100, "gradient_checkpointing": False}, 152)[
                "gradient_checkpointing"
            ]
        )
        with self.assertRaises(BusinessError):
            normalize_preset(
                {"max_steps": 100, "batch_size": 8, "gradient_accumulation_steps": 1}, 3
            )


class FolderNameTests(unittest.TestCase):
    def test_folders_are_named_for_people_to_read(self):
        from irodori_tts.desktop.storage import folder_name, stamp

        self.assertEqual(folder_name("千世"), "千世")
        self.assertEqual(folder_name('a/b:c*?"'), "a_b_c___")
        self.assertEqual(folder_name("  ..voice.. "), "voice")
        self.assertEqual(folder_name("con"), "_con")
        self.assertEqual(folder_name(""), "_")
        self.assertEqual(len(folder_name("x" * 80)), 40)
        self.assertRegex(stamp(), r"^\d{8}-\d{6}-[0-9a-f]{4}$")
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            first = create_project(parent, "千世")
            second = create_project(parent, "千世")
            self.assertEqual((first.name, second.name), ("千世", "千世 2"))
            # The folder carries the name; the ID is still a fresh one of its own.
            self.assertNotEqual(load_project(first)["id"], load_project(second)["id"])


class ProjectTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.project = create_project(self.root / "projects", "中文 空格项目")
        self.source = self.root / "source"
        self.source.mkdir()
        # Existing valid audio; no synthetic fixture is counted as GPU acceptance.
        existing = Path(__file__).resolve().parents[1] / "examples/desktop/voice_design.wav"
        import shutil

        shutil.copyfile(existing, self.source / "example.wav")
        self.csv = self.source / "text.csv"
        with self.csv.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["file_name", "text", "caption"])
            writer.writerow(["example.wav", '日语，包含 "引号" 和逗号,', "说明"])

    def tearDown(self):
        self.temp.cleanup()

    def test_managed_copy_csv_revision_external_change_and_project_lock(self):
        imported = import_materials(self.project, self.source, self.csv)
        project = load_project(self.project)
        item = imported[0]
        self.assertEqual(item["text"], '日语，包含 "引号" 和逗号,')
        original = self.source / "example.wav"
        original.unlink()
        self.assertTrue((self.project / item["path"]).exists())
        with project_lock(self.project), self.assertRaises(ValueError):
            edit_material(self.project, item["id"], text="changed", caption="", included=True)
        edit_material(self.project, item["id"], text="こんにちは。", caption="", included=False)
        edited = load_project(self.project)
        self.assertEqual(edited["revision"], project["revision"] + 1)
        managed = self.project / item["path"]
        audio, rate = sf.read(managed)
        sf.write(managed, audio * 0.9, rate)
        self.assertTrue(reconcile_sources(self.project, edited))
        self.assertEqual(load_project(self.project)["revision"], project["revision"] + 2)

    def test_metadata_failed_replace_preserves_old_record(self):
        path = self.project / "project.json"
        original = path.read_bytes()
        with (
            patch("irodori_tts.desktop.storage.os.replace", side_effect=OSError("full disk")),
            self.assertRaises(OSError),
        ):
            write_json(path, {"broken": True})
        self.assertEqual(path.read_bytes(), original)

    def test_failed_audio_export_preserves_previous_export(self):
        destination = self.root / "export.wav"
        destination.write_bytes(b"previous complete export")
        with (
            patch("irodori_tts.desktop.exports.shutil.copyfile", side_effect=OSError("full disk")),
            self.assertRaises(OSError),
        ):
            export_audio_file(self.source / "example.wav", destination)
        self.assertEqual(destination.read_bytes(), b"previous complete export")

    def test_project_archive_roundtrip_and_path_escape_rejected(self):
        import_materials(self.project, self.source, self.csv)
        archive = self.root / "project.iroproject"
        export_project(self.project, archive)
        moved = import_project(archive, self.root / "迁移 目录")
        self.assertEqual(load_project(moved)["materials"], load_project(self.project)["materials"])
        with self.assertRaises(ValueError):
            within(self.project, "../external")
        malicious = self.root / "escape.iroproject"
        with zipfile.ZipFile(malicious, "w") as zip_file:
            zip_file.writestr("package.json", json.dumps({"kind": "project", "format_version": 1}))
            zip_file.writestr("../escaped", "bad")
        with self.assertRaises(ValueError):
            import_project(malicious, self.root / "imports")
        self.assertFalse((self.root / "imports/escaped").exists())

    def test_diagnostics_redacts_secrets_and_request_text(self):
        value = redact(
            {
                "token": "hf_private",
                "request": {"text": "private transcript"},
                "error": "https://user:password@host/path Bearer secret",
            }
        )
        self.assertNotIn("private", json.dumps(value))
        self.assertNotIn("password@", json.dumps(value))
        self.assertNotIn("Bearer secret", json.dumps(value))

    @unittest.skipUnless(os.name == "nt", "Windows directory-junction boundary")
    def test_cleanup_rejects_actual_windows_directory_junction(self):
        external = self.root / "external"
        external.mkdir()
        marker = external / "keep.txt"
        marker.write_text("keep")
        initialized = self.project / "initialized"
        initialized.mkdir()
        junction = initialized / "junction"
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        try:
            with self.assertRaises(ValueError):
                clean_project(self.project, "initialized")
            self.assertEqual(marker.read_text(), "keep")
        finally:
            os.rmdir(junction)

    def test_managed_import_from_other_drive(self):
        fixture = Path(__file__).resolve().parents[1] / "examples/desktop"
        imported = import_materials(self.project, fixture, fixture / "transcripts.csv")
        self.assertEqual(len(imported), 2)
        self.assertTrue(all((self.project / item["path"]).is_file() for item in imported))


class StoreReviewTests(unittest.TestCase):
    """Settings recovery, portable locations, safe deletes and package identity."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.data = self.root / "data"

    def tearDown(self):
        self.temp.cleanup()

    def test_packages_carry_ffmpeg_from_a_shared_build_zip(self):
        from irodori_tts import _carried_ffmpeg
        from scripts import build_desktop

        archive = self.root / "ffmpeg.zip"
        with zipfile.ZipFile(archive, "w") as stream:
            for name in (
                "ffmpeg-8/LICENSE.txt",
                "ffmpeg-8/bin/avcodec-62.dll",
                "ffmpeg-8/bin/ffmpeg.exe",
                "ffmpeg-8/bin/ffplay.exe",
                "ffmpeg-8/include/libavutil/time.h",
                "ffmpeg-8/doc/LICENSE.txt/../ffmpeg.html",
            ):
                stream.writestr(name, "x")
        unpacked = self.root / "checkout/runtime/ffmpeg"
        build_desktop.unpack_ffmpeg(archive, unpacked)
        self.assertEqual(
            sorted(path.name for path in unpacked.iterdir()),
            ["LICENSE.txt", "avcodec-62.dll", "ffmpeg.exe"],
        )
        package = self.root / "package"
        build_desktop.bundle_ffmpeg(package, unpacked)
        self.assertTrue((package / "runtime/ffmpeg/LICENSE.txt").is_file())
        # A checkout has it under its root, a package beside its app folder.
        self.assertEqual(_carried_ffmpeg(self.root / "checkout"), unpacked)
        self.assertEqual(_carried_ffmpeg(package / "app"), package / "runtime/ffmpeg")
        self.assertIsNone(_carried_ffmpeg(self.root / "elsewhere/app"))

        static = self.root / "static.zip"
        with zipfile.ZipFile(static, "w") as stream:
            stream.writestr("ffmpeg-8/bin/ffmpeg.exe", "x")
        with self.assertRaises(RuntimeError):
            build_desktop.unpack_ffmpeg(static, unpacked)
        self.assertTrue((unpacked / "avcodec-62.dll").is_file())
        with self.assertRaises(RuntimeError):
            build_desktop.bundle_ffmpeg(package, self.root / "missing")

    def test_packages_carry_installed_addons_and_no_models(self):
        from irodori_tts.desktop import asr
        from scripts import build_desktop

        bundle = self.root / "bundle"
        current = {"qwen3-asr"}
        installed = []
        with (
            patch.object(
                asr, "addon_state", side_effect=lambda key, root: {"ready": key in current}
            ),
            patch.object(
                asr,
                "install_addon",
                side_effect=lambda key, root, progress: installed.append((key, root)),
            ),
        ):
            build_desktop.bundle_addons(bundle)
        self.assertEqual(installed, [("funasr", bundle / "resources/addons")])

    def test_every_model_file_has_a_pinned_size_and_hash(self):
        from irodori_tts.desktop import asr
        from irodori_tts.desktop.model_files import (
            fast_model,
            load_pins,
            low_vram_model,
            pinned_models,
        )

        pins = load_pins()
        for spec in [
            *pinned_models(),
            low_vram_model(),
            fast_model(),
            *(m for b in asr.BACKENDS for m in b.models),
        ]:
            entry = pins[spec.repo]
            self.assertEqual(entry["revision"], spec.revision, spec.repo)
            for name in spec.files:
                self.assertRegex(entry["files"][name]["sha256"], r"^[0-9a-f]{64}$", name)

    def test_large_downloads_resume_where_they_stopped_and_are_verified(self):
        import hashlib
        import types

        from irodori_tts.desktop import resources
        from irodori_tts.desktop.messages import BusinessError
        from irodori_tts.task_control import TaskCancelled

        content = bytes(range(256)) * 40
        etag = hashlib.sha256(content).hexdigest()
        requests = []

        class Response:
            def __init__(self, start, ranged):
                self.status_code = 206 if ranged and start else 200
                self.body = content[start:] if self.status_code == 206 else content

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def raise_for_status(self):
                pass

            def iter_bytes(self, size):
                for offset in range(0, len(self.body), 1000):
                    yield self.body[offset : offset + 1000]

        def session(ranged=True):
            def stream(method, url, headers, **_):
                requests.append(headers.get("Range"))
                start = int(headers["Range"][6:-1]) if "Range" in headers else 0
                return Response(start, ranged)

            return types.SimpleNamespace(stream=stream)

        metadata = types.SimpleNamespace(size=len(content), etag=etag)
        # A file the cache already has is used without asking the server (works offline).
        present = self.root / "present.bin"
        present.write_bytes(content)
        with (
            patch("huggingface_hub.try_to_load_from_cache", return_value=str(present)),
            patch("huggingface_hub.get_hf_file_metadata", side_effect=AssertionError("network")),
        ):
            resources.resumable_blob("org/name", "f.bin", "rev")
        with (
            patch.dict(os.environ, {"HF_HOME": str(self.root / "hf")}),
            patch.object(resources, "LARGE_FILE", 1),
            patch("huggingface_hub.get_hf_file_metadata", return_value=metadata),
            patch("huggingface_hub.utils.get_session", return_value=session()),
        ):
            blobs = self.root / "hf/hub/models--org--name/blobs"
            received = []

            def stop_after_half():
                return sum(received) >= len(content) // 2

            with self.assertRaises(TaskCancelled):
                resources.resumable_blob(
                    "org/name", "f.bin", "rev", cancel=stop_after_half, progress=received.append
                )
            kept = (blobs / f"{etag}.partial").stat().st_size
            self.assertGreater(kept, 0)
            received.clear()
            resources.resumable_blob("org/name", "f.bin", "rev", progress=received.append)
            self.assertEqual(requests[-1], f"bytes={kept}-")
            self.assertEqual(received[0], kept)
            self.assertEqual((blobs / etag).read_bytes(), content)
            self.assertFalse((blobs / f"{etag}.partial").exists())
            # A server that ignores the range sends everything again; it replaces the partial.
            (blobs / etag).unlink()
            (blobs / f"{etag}.partial").write_bytes(content[:100])
            with patch("huggingface_hub.utils.get_session", return_value=session(ranged=False)):
                resources.resumable_blob("org/name", "f.bin", "rev")
            self.assertEqual((blobs / etag).read_bytes(), content)
            # Damaged bytes are removed instead of being placed in the cache.
            (blobs / etag).unlink()
            (blobs / f"{etag}.partial").write_bytes(b"x" * len(content))
            with self.assertRaises(BusinessError):
                resources.resumable_blob("org/name", "f.bin", "rev")
            self.assertFalse((blobs / etag).exists())
            self.assertFalse((blobs / f"{etag}.partial").exists())

    def test_a_folder_too_long_for_windows_paths_is_named(self):
        from irodori_tts.desktop import resources

        short, cache = Path("C:/Irodori/data"), Path("C:/Irodori/.cache")
        deep = Path("C:/" + "d" * (resources.MAX_PATH - resources.DATA_DEPTH))
        with patch.object(resources, "long_paths_enabled", return_value=False):
            self.assertEqual(resources.paths_too_long(short, cache), [])
            self.assertEqual(resources.paths_too_long(deep, cache), [deep.resolve()])
        with patch.object(resources, "long_paths_enabled", return_value=True):
            self.assertEqual(resources.paths_too_long(deep, cache), [])

    def test_voice_packages_carry_inference_settings_but_not_local_references(self):
        import zipfile

        from irodori_tts.desktop.exports import export_voice, import_voice
        from irodori_tts.desktop.resources import MODEL_REPO, MODEL_REVISION

        root = self.root / "project"
        checkpoint = root / "runs/r1/checkpoints/checkpoint_0000010"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "adapter_model.safetensors").write_bytes(b"weights")
        write_json(
            root / "runs/r1/inference_settings.json",
            {"duration_scale": 0.85, "references": [{"path": "C:/mine.wav"}]},
        )
        run = {
            "id": "r1",
            "name": "Hana",
            "base": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
            "step": 10,
            "checkpoint": "runs/r1/checkpoints/checkpoint_0000010",
        }
        package = self.root / "hana.zip"
        export_voice(root, run, package)
        with zipfile.ZipFile(package) as archive:
            settings = json.loads(archive.read("inference_settings.json"))
        self.assertEqual(settings, {"duration_scale": 0.85})
        imported = import_voice(package, self.root / "voices")
        self.assertEqual(read_json(imported / "inference_settings.json"), {"duration_scale": 0.85})

    @staticmethod
    def trainer_checkpoint(files: dict, folder: str, *, model: dict | None, weights=b"weights"):
        """The files of a checkpoint folder as the upstream trainer saves it, at folder."""
        from irodori_tts.desktop.resources import MODEL_ARCHITECTURE

        files[f"{folder}/adapter_config.json"] = b"{}"
        files[f"{folder}/adapter_model.safetensors"] = weights
        files[f"{folder}/trainer_state.pt"] = b"optimizer"
        files[f"{folder}/irodori_lora_metadata.json"] = b'{"base_init": {"checkpoint_path": "x"}}'
        if model is not False:
            config = {"model": {**MODEL_ARCHITECTURE, **(model or {})}, "train": {}}
            files[f"{folder}/config.json"] = json.dumps(config).encode()

    def test_trainer_checkpoints_import_from_nested_folders_keeping_only_what_inference_needs(
        self,
    ):
        import zipfile

        from irodori_tts.desktop.exports import import_voice, voice_candidates
        from irodori_tts.desktop.resources import MODEL_REPO, MODEL_REVISION

        files = {"misuzu/README.txt": b"hello", "misuzu/samples/a.wav": b"RIFF"}
        self.trainer_checkpoint(files, "misuzu/runs/checkpoint_0001100", model=None)
        package = self.root / "misuzu.zip"
        with zipfile.ZipFile(package, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)

        found = voice_candidates(package, self.root / "voices")
        self.assertEqual(
            found,
            [{"folder": "misuzu/runs/checkpoint_0001100", "name": "misuzu", "problem": None}],
        )
        imported = import_voice(package, self.root / "voices")
        self.assertEqual(
            sorted(path.name for path in imported.iterdir()),
            [
                "adapter_config.json",
                "adapter_model.safetensors",
                "config.json",
                "irodori_lora_metadata.json",
                "voice.json",
            ],
        )
        voice = read_json(imported / "voice.json")
        self.assertEqual(voice["name"], "misuzu")
        self.assertEqual(voice["step"], 1100)
        self.assertEqual(voice["base"], {"repo": MODEL_REPO, "revision": MODEL_REVISION})
        self.assertEqual(
            read_json(imported / "irodori_lora_metadata.json"),
            {"base_resource": {"repo": MODEL_REPO, "revision": MODEL_REVISION}},
        )
        self.assertEqual([p.name for p in (self.root / "voices").iterdir()], [imported.name])

    def test_a_package_with_several_sizes_offers_only_the_one_that_fits(self):
        import zipfile

        from irodori_tts.desktop.exports import import_voice, voice_candidates
        from irodori_tts.desktop.messages import BusinessError

        files = {}
        self.trainer_checkpoint(files, "hana/small", model=None)
        self.trainer_checkpoint(files, "hana/large", model={"model_dim": 2048, "num_layers": 24})
        files["hana/large/base_model.json"] = b'{"repo_id": "Aratako/Irodori-TTS-v4-Large"}'
        package = self.root / "hana.zip"
        with zipfile.ZipFile(package, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)

        found = {item["folder"]: item for item in voice_candidates(package, self.root / "v")}
        self.assertIsNone(found["hana/small"]["problem"])
        self.assertEqual(found["hana/small"]["name"], "hana small")
        problem = found["hana/large"]["problem"]
        self.assertEqual(problem.key, "exports.voice_for_another_base_model")
        self.assertIn("Irodori-TTS-v4-Large", str(problem.params["model"].value))

        with self.assertRaises(BusinessError) as raised:
            import_voice(package, self.root / "v")
        self.assertEqual(raised.exception.message.key, "exports.choose_one_voice_in_package")
        with self.assertRaises(BusinessError) as raised:
            import_voice(package, self.root / "v", "hana/large")
        self.assertEqual(raised.exception.message.key, "exports.voice_for_another_base_model")
        imported = import_voice(package, self.root / "v", "hana/small")
        self.assertEqual(read_json(imported / "voice.json")["name"], "hana small")
        self.assertEqual([p.name for p in (self.root / "v").iterdir()], [imported.name])

    def test_an_adapter_without_config_is_judged_by_its_tensor_shapes(self):
        from irodori_tts.desktop.exports import import_voice
        from irodori_tts.desktop.messages import BusinessError

        def safetensors(width, layers):
            header = {
                f"base_model.model.blocks.{i}.attention.wq.lora_A.weight": {
                    "dtype": "F32",
                    "shape": [16, width],
                    "data_offsets": [0, 0],
                }
                for i in range(layers)
            }
            raw = json.dumps(header).encode()
            return len(raw).to_bytes(8, "little") + raw

        for width, layers, fits in ((1280, 12, True), (2048, 24, False)):
            folder = self.root / f"checkpoint_final_{width}"
            files = {}
            self.trainer_checkpoint(files, ".", model=False, weights=safetensors(width, layers))
            for name, data in files.items():
                (folder / name).parent.mkdir(parents=True, exist_ok=True)
                (folder / name).write_bytes(data)
            # Choosing the adapter_config.json in the dialog takes its folder.
            if fits:
                imported = import_voice(folder / "adapter_config.json", self.root / "voices")
                self.assertTrue((imported / "adapter_model.safetensors").is_file())
            else:
                with self.assertRaises(BusinessError) as raised:
                    import_voice(folder, self.root / "voices")
                self.assertEqual(
                    raised.exception.message.key, "exports.voice_for_another_base_model"
                )

    def test_voice_packages_in_7z_and_rar_form_are_read_through_7zip(self):
        import subprocess

        from irodori_tts.desktop.exports import import_voice, voice_candidates
        from irodori_tts.desktop.model_files import seven_zip

        program = seven_zip()
        if program is None:
            self.skipTest("7-Zip is not installed")
        source = self.root / "source"
        files = {"notes.txt": b"x"}
        self.trainer_checkpoint(files, "hana [v2]/checkpoint_final", model=None)
        for name, data in files.items():
            (source / name).parent.mkdir(parents=True, exist_ok=True)
            (source / name).write_bytes(data)
        # Named .zip, as packages renamed by their sharer are.
        package = self.root / "hana.zip"
        subprocess.run(
            [program, "a", "-t7z", str(package), "."], cwd=source, check=True, capture_output=True
        )
        self.assertEqual(
            voice_candidates(package, self.root / "voices"),
            [{"folder": "hana [v2]/checkpoint_final", "name": "hana", "problem": None}],
        )
        imported = import_voice(package, self.root / "voices")
        self.assertEqual((imported / "adapter_model.safetensors").read_bytes(), b"weights")
        self.assertFalse((imported / "trainer_state.pt").exists())

    def test_model_files_import_from_folders_and_zips_and_reject_other_versions(self):
        import hashlib
        import zipfile

        from irodori_tts.desktop import model_files
        from irodori_tts.desktop.messages import BusinessError
        from irodori_tts.desktop.resources import ModelSpec

        weights, config = b"w" * 5000, b'{"a": 1}'
        pins = {
            "org/voice": {
                "revision": "rev1",
                "files": {
                    name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in (("model.safetensors", weights), ("sub/config.json", config))
                },
            }
        }
        source = self.root / "from-a-friend"
        (source / "deep").mkdir(parents=True)
        (source / "deep/model.safetensors").write_bytes(weights)
        (source / "config.json").write_bytes(config)
        (source / "unrelated.txt").write_text("hello", encoding="utf-8")
        spec = ModelSpec("org/voice", "rev1", ("model.safetensors", "sub/config.json"), 5008)
        item = model_files.Download("voice", "Voice", "required", "", (spec,))
        with (
            patch.object(model_files, "load_pins", return_value=pins),
            patch.object(model_files, "downloads", return_value=[item]),
            patch.dict(os.environ, {"HF_HOME": str(self.root / "hf")}),
        ):
            result = model_files.import_model_files(source)
            self.assertEqual(
                sorted(result["imported"]),
                ["org/voice/model.safetensors", "org/voice/sub/config.json"],
            )
            self.assertEqual(result["complete"], ["voice"])
            placed = model_files.snapshot_path("org/voice", "rev1", "sub/config.json")
            self.assertEqual(placed.read_bytes(), config)
            # Importing again finds everything in place.
            again = model_files.import_model_files(source)
            self.assertEqual((again["imported"], len(again["present"])), ([], 2))
            # An export imports into another computer's empty cache.
            package = self.root / "models.zip"
            model_files.export_model_files(["voice"], package)
            with zipfile.ZipFile(package) as archive:
                self.assertIn(model_files.MANIFEST, archive.namelist())
            os.environ["HF_HOME"] = str(self.root / "other")
            moved = model_files.import_model_files(package)
            self.assertEqual(len(moved["imported"]), 2)
            self.assertEqual(moved["complete"], ["voice"])
            # The same name and size with other content is refused, and leaves nothing behind.
            os.environ["HF_HOME"] = str(self.root / "third")
            (source / "deep/model.safetensors").write_bytes(b"x" * 5000)
            refused = model_files.import_model_files(source)
            self.assertEqual(
                [name for name, _ in refused["rejected"]], [str(Path("deep/model.safetensors"))]
            )
            self.assertFalse(
                model_files.snapshot_path("org/voice", "rev1", "model.safetensors").exists()
            )
            self.assertEqual(list((self.root / "third/hub").glob(".import-*")), [])
            stray = self.root / "notes.txt"
            stray.write_text("not models", encoding="utf-8")
            with self.assertRaises(BusinessError):
                model_files.import_model_files(stray)

    def test_model_files_import_reads_a_7z_in_one_pass(self):
        import hashlib

        from irodori_tts.desktop import model_files
        from irodori_tts.desktop.resources import ModelSpec

        program = model_files.seven_zip()
        if program is None:
            self.skipTest("7-Zip is not installed")
        weights, config, extra = b"w" * 300_000, b'{"a": 1}', b"not a model file"
        pins = {
            "org/voice": {
                "revision": "rev1",
                "files": {
                    name: {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                    for name, data in (("model.safetensors", weights), ("sub/config.json", config))
                },
            }
        }
        source = self.root / "pack"
        (source / "sub").mkdir(parents=True)
        # "a-notes.txt" comes first in the archive and must be read past.
        (source / "a-notes.txt").write_bytes(extra)
        (source / "model.safetensors").write_bytes(weights)
        (source / "sub/config.json").write_bytes(config)
        archive = self.root / "models.7z"
        subprocess.run(
            [program, "a", "-t7z", "-ms=on", "-bso0", "-bsp0", str(archive), "*"],
            cwd=source,
            check=True,
        )
        spec = ModelSpec("org/voice", "rev1", ("model.safetensors", "sub/config.json"), 300_008)
        item = model_files.Download("voice", "Voice", "required", "", (spec,))
        with (
            patch.object(model_files, "load_pins", return_value=pins),
            patch.object(model_files, "downloads", return_value=[item]),
            patch.dict(os.environ, {"HF_HOME": str(self.root / "hf")}),
        ):
            # The config is already in place; its bytes are skipped, the weights still land.
            present = model_files.snapshot_path("org/voice", "rev1", "sub/config.json")
            present.parent.mkdir(parents=True)
            present.write_bytes(config)
            result = model_files.import_model_files(archive)
            self.assertEqual(result["imported"], ["org/voice/model.safetensors"])
            self.assertEqual(len(result["present"]), 1)
            self.assertEqual(result["complete"], ["voice"])
            placed = model_files.snapshot_path("org/voice", "rev1", "model.safetensors")
            self.assertEqual(placed.read_bytes(), weights)
            self.assertEqual(list((self.root / "hf/hub").glob(".import-*")), [])

    def test_package_data_lives_beside_the_launcher(self):
        from irodori_tts.desktop import app

        package = self.root / "package"
        (package / "app").mkdir(parents=True)
        with patch.object(app, "REPO", package / "app"):
            self.assertIsNone(app.package_root())
            self.assertEqual(app.default_data_root(), package / "app/data")
            (package / app.LAUNCHER).write_bytes(b"")
            self.assertEqual(app.package_root(), package)
            self.assertEqual(app.default_data_root(), package / "data")

    def test_importing_the_package_does_not_load_torch(self):
        code = "import sys, irodori_tts; print('torch' in sys.modules)"
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "False")

    def test_model_files_import_fills_every_model_sharing_a_file(self):
        import hashlib

        from irodori_tts.desktop import model_files
        from irodori_tts.desktop.resources import ModelSpec

        shared, own = b'{"vocab": 1}', b"w" * 3000
        entry = {"size": len(shared), "sha256": hashlib.sha256(shared).hexdigest()}
        pins = {
            "org/a": {"revision": "ra", "files": {"vocab.json": entry}},
            "org/b": {
                "revision": "rb",
                "files": {
                    "vocab.json": entry,
                    "model.bin": {"size": len(own), "sha256": hashlib.sha256(own).hexdigest()},
                },
            },
        }
        source = self.root / "b-only"
        source.mkdir()
        (source / "vocab.json").write_bytes(shared)
        (source / "model.bin").write_bytes(own)
        items = [
            model_files.Download(
                "a", "A", "optional", "", (ModelSpec("org/a", "ra", ("vocab.json",), 12),)
            ),
            model_files.Download(
                "b",
                "B",
                "optional",
                "",
                (ModelSpec("org/b", "rb", ("vocab.json", "model.bin"), 3012),),
            ),
        ]
        with (
            patch.object(model_files, "load_pins", return_value=pins),
            patch.object(model_files, "downloads", return_value=items),
            patch.dict(os.environ, {"HF_HOME": str(self.root / "hf")}),
        ):
            result = model_files.import_model_files(source)
            self.assertEqual(result["complete"], ["a", "b"])
            for repo, revision in (("org/a", "ra"), ("org/b", "rb")):
                placed = model_files.snapshot_path(repo, revision, "vocab.json")
                self.assertEqual(placed.read_bytes(), shared)
            self.assertEqual(list((self.root / "hf/hub").glob(".import-*")), [])

    def test_compute_processes_skip_cudnn_attention_unless_overridden(self):
        from irodori_tts.desktop.resources import configure_environment

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("IRODORI_SDPA_CUDNN", None)
            configure_environment(self.root)
            self.assertEqual(os.environ["IRODORI_SDPA_CUDNN"], "0")
            os.environ["IRODORI_SDPA_CUDNN"] = "1"
            configure_environment(self.root)
            self.assertEqual(os.environ["IRODORI_SDPA_CUDNN"], "1")

    def test_old_settings_gain_defaults_and_corrupt_settings_are_kept_aside(self):
        from irodori_tts.desktop.storage import AppStore

        self.data.mkdir()
        (self.data / "settings.json").write_text(
            json.dumps({"theme": "neon", "ui": {"page": 2}, "projects": "bad"}), "utf-8"
        )
        store = AppStore(self.data)
        self.assertIsNone(store.notice)
        self.assertEqual(store.settings["theme"], "system")
        self.assertEqual(store.settings["projects"], [])
        self.assertEqual(store.settings["ui"], {"page": 2})
        self.assertEqual(store.project_root(), self.data / "projects")
        (self.data / "settings.json").write_text('{"projects": [', "utf-8")
        recovered = AppStore(self.data)
        self.assertIsNotNone(recovered.notice)
        self.assertEqual(recovered.settings["projects"], [])
        self.assertEqual(len(list(self.data.glob("settings.json.corrupt-*"))), 1)
        self.assertFalse((self.data / "settings.json").exists())
        recovered.save()
        written = (self.data / "settings.json").stat().st_mtime_ns
        recovered.save()
        self.assertEqual((self.data / "settings.json").stat().st_mtime_ns, written)

    def test_missing_message_key_renders_without_raising(self):
        from irodori_tts.desktop.messages import persist, restore, t

        record = persist({"message": t("tasks.stopped")})
        record["_i18n"]["fields"]["message"]["$message"] = "removed.in_a_later_version"
        self.assertEqual(str(restore(record)["message"]), "removed.in_a_later_version")

    def test_moved_data_folder_finds_projects_history_and_references(self):
        from irodori_tts.desktop.storage import AppStore

        store = AppStore(self.data)
        project = create_project(store.project_root(), "portable")
        store.register(project)
        self.assertEqual(store.settings["projects"], [f"projects/{project.name}"])
        record = self.data / "generations" / "g1"
        record.mkdir(parents=True)
        (record / "audio.wav").write_bytes(b"x")
        old_root = self.root / "old-location" / "data"
        write_json(
            record / "generation.json",
            {
                "id": "other",
                "created": "2026-10-27T10:00:00+00:00",
                "audio": str(old_root / "generations/g1/audio.wav"),
                "request": {"text": "a"},
            },
        )
        store.save()
        moved = self.root / "moved" / "data"
        moved.parent.mkdir()
        os.replace(self.data, moved)
        reopened = AppStore(moved)
        self.assertEqual(
            [root for root, _ in reopened.projects()], [moved / "projects" / project.name]
        )
        entry = reopened.history()[0]
        self.assertEqual(entry["id"], "g1")
        self.assertEqual(Path(entry["audio"]), moved / "generations/g1/audio.wav")
        # An absolute entry written before relative storage is re-based onto the new root.
        reopened.settings["projects"] = [str(old_root / "projects" / project.name)]
        self.assertEqual(
            [root for root, _ in reopened.projects()], [moved / "projects" / project.name]
        )
        self.assertEqual(reopened.settings["projects"], [f"projects/{project.name}"])

    def test_deletes_refuse_the_root_itself(self):
        from irodori_tts.desktop.storage import AppStore, removable

        store = AppStore(self.data)
        (self.data / "generations" / "keep").mkdir(parents=True)
        for value in ("", "."):
            with self.assertRaises(ValueError):
                store.delete_history(value)
            with self.assertRaises(ValueError):
                removable(self.data / "generations", value)
        self.assertTrue((self.data / "generations" / "keep").is_dir())

    def test_reimported_project_package_gets_new_ids_and_keeps_the_original(self):
        from irodori_tts.desktop.storage import AppStore

        store = AppStore(self.data)
        original = create_project(store.project_root(), "original")
        run_root = original / "runs" / "run1"
        run_root.mkdir(parents=True)
        write_json(run_root / "run.json", {"id": "run1", "checkpoint": "runs/run1/checkpoints/a"})
        store.register(original)
        archive = self.root / "backup.iroproject"
        export_project(original, archive)
        copy = import_project(
            archive,
            store.project_root(),
            taken_projects=store.project_ids(),
            taken_runs={run["id"] for *_rest, run in store.runs()},
        )
        store.register(copy)
        roots = [root for root, _ in store.projects()]
        self.assertEqual(set(roots), {original.resolve(), copy.resolve()})
        self.assertNotEqual(load_project(copy)["id"], load_project(original)["id"])
        (copied_run,) = list((copy / "runs").glob("*/run.json"))
        run = json.loads(copied_run.read_text("utf-8"))
        self.assertNotEqual(run["id"], "run1")
        # A new ID; the folder keeps its name, and the checkpoint its path.
        self.assertEqual(copied_run.parent.name, "run1")
        self.assertEqual(run["checkpoint"], "runs/run1/checkpoints/a")

    def test_unchanged_recordings_are_not_hashed_again(self):
        project = create_project(self.root / "projects", "hash")
        fixture = Path(__file__).resolve().parents[1] / "examples/desktop"
        import_materials(project, fixture, fixture / "transcripts.csv")
        self.assertFalse(reconcile_sources(project, load_project(project)))
        with patch("irodori_tts.desktop.storage.digest", side_effect=AssertionError("hashed")):
            self.assertFalse(reconcile_sources(project, load_project(project)))


def tone(seconds, rate=16000, level=0.3):
    t = np.arange(round(seconds * rate)) / rate
    return (level * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(
        np.float32
    )


def silence(seconds, rate=16000):
    return np.full(round(seconds * rate), 1e-4, dtype=np.float32)


class RecordingChannel:
    """Stands in for the worker's TaskChannel."""

    task_id = "asr-test"

    def __init__(self):
        self.events = []
        self.cancel = False

    def emit(self, kind, **values):
        self.events.append((kind, values))

    def stopped(self):
        return self.cancel


class AutomaticTranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.folder = self.base / "voices"
        (self.folder / "sub").mkdir(parents=True)
        # 25.3 s with pauses of 0.6 / 0.4 / 0.8 s, a 3 s clip, and one in a subfolder.
        self.long = np.concatenate(
            [tone(5), silence(0.6), tone(5.5), silence(0.4), tone(7), silence(0.8), tone(6)]
        )
        sf.write(self.folder / "long.wav", self.long, 16000)
        sf.write(self.folder / "short.wav", tone(3), 16000)
        sf.write(self.folder / "sub" / "clip.wav", tone(2), 16000)
        self.root = create_project(self.base / "projects", "asr")

    # ----- splitting -----

    def test_cuts_fall_in_the_latest_silences_and_cover_the_recording(self):
        segments = split_points(self.long, 16000)
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0][0], 0.0)
        self.assertAlmostEqual(segments[-1][1], len(self.long) / 16000)
        for (_, end), (start, _) in zip(segments, segments[1:], strict=False):
            self.assertEqual(end, start)
        for start, end in segments:
            self.assertLessEqual(end - start, 12.0)
            self.assertGreaterEqual(end - start, 1.0)
        # 11.1-11.5 s is the last pause inside the first 12 s, 18.5-19.3 s inside the next 12.
        self.assertTrue(11.1 <= segments[0][1] <= 11.5, segments)
        self.assertTrue(18.5 <= segments[1][1] <= 19.3, segments)
        for _, cut in segments[:-1]:
            window = self.long[round((cut - 0.05) * 16000) : round((cut + 0.05) * 16000)]
            self.assertLess(float(np.abs(window).max()), 0.01)

    def test_short_recording_is_not_cut_and_no_silence_cuts_at_the_quietest_point(self):
        self.assertEqual(split_points(tone(11.9), 16000), [(0.0, 11.9 - 0.0)])
        audio = tone(30)
        for dip in (9.0, 20.0):
            audio[round(dip * 16000) : round((dip + 0.1) * 16000)] *= 0.15
        segments = split_points(audio, 16000)
        self.assertEqual(len(segments), 3)
        self.assertTrue(8.95 <= segments[0][1] <= 9.15, segments)
        self.assertTrue(19.95 <= segments[1][1] <= 20.15, segments)
        self.assertTrue(all(end - start <= 12.0 for start, end in segments))

    def test_segments_keep_their_source_and_time_range(self):
        staging = self.root / "temp" / "asr-test"
        units, stats = asr.prepare_units(self.root, self.folder, staging, split=True)
        self.assertEqual(stats, {"files": 3, "split_files": 1, "segments": 3, "skipped": 0})
        expected = split_points(self.long, 16000)
        pieces = [u for u in units if u["segment"]]
        self.assertEqual(len(pieces), 3)
        self.assertEqual(
            [u["name"] for u in pieces], ["long_01.flac", "long_02.flac", "long_03.flac"]
        )
        for unit, (start, end) in zip(pieces, expected, strict=True):
            self.assertEqual(unit["segment"]["file"], "long.wav")
            self.assertAlmostEqual(unit["segment"]["start"], start, places=3)
            self.assertAlmostEqual(unit["segment"]["end"], end, places=3)
            self.assertTrue(unit["move"])
            self.assertAlmostEqual(sf.info(unit["audio"]).duration, end - start, delta=0.002)
            self.assertAlmostEqual(len(unit["clip"]) / 16000, end - start, delta=0.002)
            self.assertEqual(unit["source"], str((self.folder / "long.wav").resolve()))
        whole = {u["name"]: u for u in units if not u["segment"]}
        self.assertEqual(set(whole), {"short.wav", "sub/clip.wav"})
        self.assertEqual(whole["short.wav"]["audio"], self.folder / "short.wav")
        self.assertFalse(whole["short.wav"]["move"])
        shutil.rmtree(staging)
        units, stats = asr.prepare_units(self.root, self.folder, staging, split=False)
        self.assertEqual(stats["segments"], 0)
        self.assertEqual(len(units), 3)
        self.assertTrue(all(u["segment"] is None for u in units))

    # ----- registry and isolation -----

    def test_backend_order_defaults_and_pinned_revisions(self):
        self.assertEqual(
            asr.BACKEND_KEYS, ("whisper-large-v3", "qwen3-asr", "anime-whisper", "fun-asr-nano")
        )
        self.assertEqual([b.recommended for b in asr.BACKENDS], [True, False, False, False])
        self.assertEqual((asr.DEFAULT_MAIN, asr.DEFAULT_CHECK), ("whisper-large-v3", "qwen3-asr"))
        self.assertEqual(
            default_settings()["asr"], {"main": "whisper-large-v3", "check": "qwen3-asr"}
        )
        for backend in asr.BACKENDS:
            self.assertGreater(backend.size, 10**9)
            self.assertGreater(backend.vram, 0)
            for model in backend.models:
                self.assertRegex(model.revision, r"^[0-9a-f]{40}$")
        self.assertEqual(
            {b.key: b.addon for b in asr.BACKENDS},
            {
                "whisper-large-v3": None,
                "qwen3-asr": "qwen3-asr",
                "anime-whisper": None,
                "fun-asr-nano": "funasr",
            },
        )
        # Only the add-on that carries another transformers may replace the application's.
        self.assertTrue(asr.ADDONS["qwen3-asr"].replaces)
        self.assertIn("transformers==4.57.6", asr.ADDONS["qwen3-asr"].packages)
        self.assertFalse(asr.ADDONS["funasr"].replaces)
        self.assertFalse(any(p.startswith("transformers") for p in asr.ADDONS["funasr"].packages))

    def test_saved_choices_fall_back_and_the_check_differs_from_the_main(self):
        self.assertEqual(asr.choice({}), ("whisper-large-v3", "qwen3-asr"))
        self.assertEqual(asr.choice({"asr": {"main": "nope", "check": 3}}), asr.choice({}))
        self.assertEqual(
            asr.choice({"asr": {"main": "whisper-large-v3", "check": "whisper-large-v3"}}),
            ("whisper-large-v3", "none"),
        )
        self.assertEqual([b.key for b in asr.selection("anime-whisper", "none")], ["anime-whisper"])
        with self.assertRaises(ValueError):
            asr.selection("qwen3-asr", "qwen3-asr")

    def test_addon_environment_layers_over_the_application_and_wins(self):
        addons = self.base / "addons"
        self.assertEqual(asr.launch_environment({"kind": "transcribe", "addons": []}, {}), {})
        environment = asr.isolation_environment(
            ["qwen3-asr", "funasr"], {"PYTHONPATH": "x"}, addons
        )
        self.assertEqual(
            environment["PYTHONPATH"],
            os.pathsep.join([str(addons / "qwen3-asr"), str(addons / "funasr"), "x"]),
        )
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")
        # A real worker interpreter: the add-on's tqdm shadows the installed one, torch is untouched.
        (addons / "qwen3-asr" / "tqdm").mkdir(parents=True)
        (addons / "qwen3-asr" / "tqdm" / "__init__.py").write_text("__version__ = 'isolated'")
        probe = "import tqdm, sys; print(tqdm.__version__)"
        with_addon = subprocess.run(
            [sys.executable, "-c", probe],
            env={**os.environ, **asr.isolation_environment(["qwen3-asr"], os.environ, addons)},
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        without = subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True
        ).stdout.strip()
        self.assertEqual(with_addon, "isolated")
        self.assertNotEqual(without, "isolated")

    def installer(self, script):
        return patch.object(asr, "find_installer", return_value=[sys.executable, "-c", script])

    def test_addon_install_records_the_pins_and_cancel_leaves_nothing(self):
        addons = self.base / "addons"
        writer = (
            "import sys, pathlib\n"
            "target = pathlib.Path(sys.argv[sys.argv.index('--target') + 1])\n"
            "for pin in sys.argv[sys.argv.index('--no-deps') + 1:]:\n"
            "    name, version = pin.split('==')\n"
            "    (target / f\"{name.replace('-', '_')}-{version}.dist-info\").mkdir(parents=True)\n"
            "print('installed', flush=True)\n"
        )
        self.assertEqual(asr.missing_addons(asr.BACKENDS, addons), ["qwen3-asr", "funasr"])
        lines = []
        with self.installer(writer):
            asr.install_addon("qwen3-asr", root=addons, progress=lines.append)
        self.assertIn("installed", lines)
        self.assertTrue(asr.addon_state("qwen3-asr", addons)["ready"])
        self.assertEqual(asr.missing_addons(asr.BACKENDS, addons), ["funasr"])
        asr.remove_addon("qwen3-asr", addons)
        self.assertFalse(asr.addon_state("qwen3-asr", addons)["ready"])
        # Cancel after the installer started: it is killed and no partial directory stays.
        stopped = []
        sleeper = "import time; print('started', flush=True); time.sleep(60)"
        started = time.monotonic()
        with self.installer(sleeper), self.assertRaises(TaskCancelled):
            asr.install_addon(
                "funasr", root=addons, progress=stopped.append, cancel=lambda: bool(stopped)
            )
        self.assertLess(time.monotonic() - started, 30)
        self.assertEqual(sorted(p.name for p in addons.iterdir()), [])
        with self.installer("import sys; print('no network'); sys.exit(3)"):
            with self.assertRaises(BusinessError) as raised:
                asr.install_addon("funasr", root=addons)
        self.assertIn("no network", str(raised.exception))

    # ----- cross-check and import -----

    def test_cross_check_flags_only_different_readings(self):
        self.assertFalse(
            asr.readings_differ("今日はいい天気ですね。", "きょうは、いいてんきですね")
        )
        self.assertTrue(asr.readings_differ("今日はいい天気ですね。", "今日は悪い天気ですね。"))
        self.assertTrue(asr.readings_differ("ありがとう", ""))
        with patch("pyopenjtalk.g2p", side_effect=RuntimeError("no dictionary")):
            self.assertFalse(asr.readings_differ("ありがとう！", "ありがとう"))
            self.assertTrue(asr.readings_differ("ありがとう", "ありがと"))

    def fake_backends(self, channel=None, differ=("short.wav",)):
        """Recognizers that answer with the clip length; the check disagrees on `differ`."""
        calls = []

        def make(offset):
            def load(paths, batch):
                def transcribe(clips):
                    calls.append(len(clips))
                    if channel is not None:
                        channel.cancel = True
                    return [f"あ{round(len(clip) / 1600)}" + "い" * offset for clip in clips]

                return asr.Recognizer(transcribe, lambda: None)

            return load

        return calls, patch.dict(
            asr.LOADERS,
            {"qwen3-asr": make(0), "whisper-large-v3": make(0), "anime-whisper": make(1)},
        )

    def run_import(self, check="whisper-large-v3", channel=None, **extra):
        channel = channel or RecordingChannel()
        payload = {
            "project": str(self.root),
            "folder": str(self.folder),
            "main": "qwen3-asr",
            "check": check,
            "split": True,
            **extra,
        }
        models = {key: {} for key in asr.BACKEND_KEYS}
        return asr.run_import(payload, channel, models=models), channel

    def test_transcribing_project_recordings_again_replaces_only_their_text(self):
        calls, patched = self.fake_backends()
        with patched:
            self.run_import()
        items = load_project(self.root)["materials"]
        first, second = items[0], items[1]
        edit_material(self.root, first["id"], text="直しました", caption="説明", included=True)
        before = {i["id"]: i for i in load_project(self.root)["materials"]}
        with patched:
            result, _ = self.run_import(check="anime-whisper", materials=[first["id"]])
        self.assertTrue(result["again"])
        self.assertEqual((result["imported"], result["needs_review"]), (1, 1))
        after = {i["id"]: i for i in load_project(self.root)["materials"]}
        self.assertEqual(len(after), len(before))
        redone = after[first["id"]]
        self.assertTrue(redone["text"].startswith("あ"))
        self.assertEqual(redone["asr"]["check_backend"], "anime-whisper")
        self.assertTrue(redone["asr"]["needs_review"])
        # The name, the description, the audio and where it came from stay.
        for key in ("file_name", "caption", "path", "sha256", "source", "segment"):
            self.assertEqual(redone.get(key), before[first["id"]].get(key), key)
        self.assertEqual(after[second["id"]], before[second["id"]])

    def test_import_transcribes_segments_flags_and_keeps_edits_on_reimport(self):
        calls, patched = self.fake_backends()
        with patched:
            result, channel = self.run_import()
        self.assertEqual((result["imported"], result["segments"], result["split_files"]), (5, 3, 1))
        items = load_project(self.root)["materials"]
        self.assertEqual(len(items), 5)
        self.assertTrue(all(i["included"] and i["text"].startswith("あ") for i in items))
        self.assertTrue(all(i["asr"]["backend"] == "qwen3-asr" for i in items))
        self.assertTrue(all(i["asr"]["check_backend"] == "whisper-large-v3" for i in items))
        # Identical fake readings on both sides: nothing needs review.
        self.assertEqual(result["needs_review"], 0)
        self.assertFalse((self.root / "temp" / "asr-asr-test").exists())
        segments = [i for i in items if i.get("segment")]
        self.assertEqual({i["segment"]["file"] for i in segments}, {"long.wav"})
        self.assertTrue(all(i["duration"] <= 12.0 for i in segments))
        self.assertEqual(
            sorted(i["file_name"] for i in items),
            ["long_01.flac", "long_02.flac", "long_03.flac", "short.wav", "sub/clip.wav"],
        )
        self.assertTrue(any(kind == "progress" for kind, _ in channel.events))
        # A different cross-check model disagrees: every recording is flagged and keeps its text.
        shutil.rmtree(self.root / "source")
        (self.root / "source").mkdir()
        project = load_project(self.root)
        project.update(materials=[])
        save_project(self.root, project)
        with patched:
            result, _ = self.run_import(check="anime-whisper")
        self.assertEqual(result["needs_review"], 5)
        flagged = load_project(self.root)["materials"]
        self.assertTrue(all(i["asr"]["needs_review"] for i in flagged))
        self.assertTrue(all(i["asr"]["check_text"].endswith("い") for i in flagged))
        self.assertTrue(all(not i["text"].endswith("い") for i in flagged))
        # Importing the folder again keeps what is there, edits included.
        edit_material(self.root, flagged[0]["id"], text="直しました", caption="", included=True)
        with patched:
            result, _ = self.run_import(check="anime-whisper")
        self.assertEqual((result["imported"], result["skipped"]), (0, 5))
        again = load_project(self.root)["materials"]
        self.assertEqual(again[0]["text"], "直しました")
        self.assertFalse(again[0]["asr"]["needs_review"])
        self.assertTrue(again[1]["asr"]["needs_review"])
        # A recording whose transcript was emptied is transcribed again, in place.
        edit_material(self.root, again[1]["id"], text="", caption="", included=True)
        with patched:
            result, _ = self.run_import(check="anime-whisper")
        self.assertEqual((result["imported"], result["skipped"]), (1, 4))
        final = load_project(self.root)["materials"]
        self.assertEqual(len(final), 5)
        self.assertEqual(final[1]["id"], again[1]["id"])
        self.assertTrue(final[1]["text"])
        self.assertTrue(final[1]["included"])

    def test_stop_during_recognition_leaves_the_project_untouched(self):
        channel = RecordingChannel()
        calls, patched = self.fake_backends(channel)
        before = (self.root / "project.json").read_bytes()
        with patched, self.assertRaises(TaskCancelled):
            self.run_import(channel=channel)
        self.assertEqual((self.root / "project.json").read_bytes(), before)
        self.assertEqual(list((self.root / "source").iterdir()), [])
        self.assertFalse((self.root / "temp" / "asr-asr-test").exists())
        self.assertEqual(len(calls), 1)

    def test_models_download_stops_when_cancelled_before_the_first_byte(self):
        with self.assertRaises(TaskCancelled):
            asr.download_models([asr.BACKENDS[0]], cancel=lambda: True)

    # ----- review -----

    def test_review_flag_clears_on_edit_or_explicit_review_but_not_on_toggle(self):
        calls, patched = self.fake_backends()
        with patched:
            self.run_import(check="anime-whisper")
        project = load_project(self.root)
        ids = [i["id"] for i in project["materials"]]
        self.assertTrue(all(i["asr"]["needs_review"] for i in project["materials"]))
        first, second, third, fourth = ids[:4]
        by_id = lambda: {i["id"]: i for i in load_project(self.root)["materials"]}  # noqa: E731
        # Excluding and including again is not a review.
        edit_material(self.root, first, text=by_id()[first]["text"], caption="", included=False)
        self.assertTrue(by_id()[first]["asr"]["needs_review"])
        # Changing the transcript is.
        edit_material(self.root, second, text="直した文", caption="", included=True)
        self.assertFalse(by_id()[second]["asr"]["needs_review"])
        self.assertEqual(
            by_id()[second]["asr"]["check_text"], project["materials"][1]["asr"]["check_text"]
        )
        # So is pressing Save on the unchanged text; no new revision, nothing else changes.
        revision = load_project(self.root)["revision"]
        edit_material(
            self.root, third, text=by_id()[third]["text"], caption="", included=True, reviewed=True
        )
        self.assertFalse(by_id()[third]["asr"]["needs_review"])
        self.assertEqual(load_project(self.root)["revision"], revision)
        # Marking reviewed changes no training data.
        self.assertEqual(mark_reviewed(self.root, [fourth, third]), 1)
        self.assertFalse(by_id()[fourth]["asr"]["needs_review"])
        self.assertEqual(load_project(self.root)["revision"], revision)
        # The flag is not training data: the fingerprint ignores it.
        flagged_project = load_project(self.root)
        cleared_project = copy.deepcopy(flagged_project)
        for item in cleared_project["materials"]:
            item["asr"]["needs_review"] = False
        self.assertEqual(
            materials_fingerprint(flagged_project), materials_fingerprint(cleared_project)
        )


class SplitTextTests(unittest.TestCase):
    @staticmethod
    def squeezed(text):
        return "".join(text.split())

    def test_punctuation_stays_and_neighbours_merge_greedily(self):
        from irodori_tts.desktop.long_text import split_text

        text = "あいう。えお！かき？\nくけ…こ"
        self.assertEqual(split_text(text, 100), ["あいう。えお！かき？くけ…こ"])
        self.assertEqual(split_text(text, 7), ["あいう。えお！", "かき？くけ…こ"])
        self.assertEqual(split_text(text, 6), ["あいう。", "えお！かき？", "くけ…こ"])
        self.assertEqual(split_text("はい！？ええ。", 4), ["はい！？", "ええ。"])
        self.assertEqual(split_text("「行く。」と言った。", 5), ["「行く。」", "と言った。"])

    def test_an_oversize_sentence_splits_at_commas_then_anywhere(self):
        from irodori_tts.desktop.long_text import split_text

        self.assertEqual(
            split_text("あいう、えおか、きくけこ。", 6), ["あいう、", "えおか、", "きくけこ。"]
        )
        self.assertEqual(split_text("あいう、えお。", 4), ["あいう、", "えお。"])
        # No comma to cut at: a hard cut at the limit.
        self.assertEqual(split_text("あいうえおかきくけ。", 4), ["あいうえ", "おかきく", "け。"])

    def test_nothing_is_lost_and_no_piece_is_empty_or_too_long(self):
        from irodori_tts.desktop.long_text import SEGMENT_CHARS, split_text

        text = (
            "今日は、とても長い一文が続いて止まらない" * 12
            + "。\n\n  \n次の文です！"
            + "あ" * 300
            + "？"
        ) * 3
        for limit in (SEGMENT_CHARS, 50, 7):
            pieces = split_text(text, limit)
            self.assertTrue(all(piece and len(piece) <= limit for piece in pieces))
            self.assertTrue(all(piece == piece.strip() for piece in pieces))
            self.assertEqual("".join(pieces), self.squeezed(text))
        self.assertEqual(split_text(""), [])
        self.assertEqual(split_text(" \n\n "), [])
        self.assertEqual(split_text("こんにちは。"), ["こんにちは。"])


class FakeChannel:
    def __init__(self):
        self.events = []
        self.stop_after = None

    def stopped(self):
        return self.stop_after is not None and len(self.events) >= self.stop_after

    def emit(self, kind, **values):
        self.events.append((kind, values))


class FakeRuntime:
    """Answers each request with sine clips: one per candidate, 0.5 s each."""

    sample_rate = 48000

    def __init__(self, checkpoint="model"):
        self.requests = []
        self.batches = []
        self.key = SimpleNamespace(checkpoint=checkpoint)
        self.watermarker = SimpleNamespace(ready=True)
        self.unloaded = False

    def unload(self):
        self.unloaded = True

    def synthesize(self, request, step_callback=None, log_fn=None):
        return self.synthesize_batch([request], step_callback=step_callback, log_fn=log_fn)[0]

    def synthesize_batch(self, requests, step_callback=None, log_fn=None):
        self.batches.append(len(requests))
        steps = requests[0].num_steps
        for step in range(1, steps + 1):
            step_callback(step, steps)
        return [self.answer(request) for request in requests]

    def answer(self, request):
        self.requests.append(request)
        clips = [
            torch.full((1, self.sample_rate // 2 + 100 * i), 0.1 * (i + 1))
            for i in range(request.num_candidates)
        ]
        seed = request.seed if request.seed is not None else 1000
        return SimpleNamespace(
            audio=clips[0],
            audios=clips,
            sample_rate=self.sample_rate,
            stage_timings=[],
            used_seed=seed,
        )


class GenerationWorkerTests(unittest.TestCase):
    def run_task(self, **payload):
        from irodori_tts.desktop import worker

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        payload = {
            "kind": "generate",
            "request": {"text": "こんにちは。", "num_steps": 4, **payload.pop("request", {})},
            "output": str(root / "generations" / "g1"),
            **payload,
        }
        channel = payload.pop("channel", None) or FakeChannel()
        runtime = payload.pop("runtime", None) or FakeRuntime()
        with patch.object(worker, "runtime_sources", return_value={}):
            _runtime, record = worker.synthesize(payload, channel, {"model": "model"}, runtime)
        return Path(payload["output"]), record, runtime, channel

    def test_candidates_are_separate_files_in_one_record(self):
        output, record, runtime, _ = self.run_task(request={"num_candidates": 3, "seed": 5})
        self.assertEqual(runtime.requests[0].num_candidates, 3)
        names = ["audio.wav", "audio_2.wav", "audio_3.wav"]
        self.assertEqual(record["candidates"], [str(output / name) for name in names])
        self.assertEqual(len(record["durations"]), 3)
        for name, duration in zip(names, record["durations"], strict=True):
            info = sf.info(str(output / name))
            self.assertEqual(info.samplerate, 48000)
            self.assertAlmostEqual(info.frames / info.samplerate, duration, places=3)
        self.assertEqual(record["audio"], str(output / "audio.wav"))
        self.assertEqual(record["duration"], record["durations"][0])
        stored = read_json(output / "generation.json")
        self.assertEqual(stored["candidates"], names)
        self.assertEqual(stored["audio"], "audio.wav")
        self.assertEqual(list(output.glob("*.partial.wav")), [])

    def test_a_preview_stays_a_single_clip_without_candidates(self):
        from irodori_tts.desktop import worker

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        checkpoint = root / "run" / "checkpoint"
        checkpoint.mkdir(parents=True)
        payload = {
            "kind": "preview",
            "project": str(root),
            "checkpoint_snapshot": str(checkpoint),
            "request": {"text": "a", "num_steps": 2, "num_candidates": 3},
            "output": str(root / "preview"),
        }
        runtime = FakeRuntime()
        with (
            patch.object(worker, "runtime_sources", return_value={}),
            patch.object(worker, "complete_checkpoint", return_value=True),
        ):
            _runtime, record = worker.synthesize(
                payload, FakeChannel(), {"model": "model"}, runtime
            )
        self.assertEqual(runtime.requests[0].num_candidates, 1)
        self.assertNotIn("candidates", record)

    def test_long_text_is_read_in_segments_with_following_seeds(self):
        sentences = ["あ" * 99 + "。", "い" * 99 + "。", "う" * 99 + "。"]
        output, record, runtime, channel = self.run_task(
            request={
                "text": "".join(sentences),
                "seconds": 12.0,
                "num_candidates": 3,
                "seed": 10,
                "num_steps": 4,
            }
        )
        self.assertEqual([r.text for r in runtime.requests], sentences)
        self.assertEqual([r.seed for r in runtime.requests], [10, 11, 12])
        self.assertTrue(all(r.seconds is None and r.num_candidates == 1 for r in runtime.requests))
        self.assertEqual(record["segments"], 3)
        self.assertEqual(record["seed"], 10)
        self.assertEqual(record["candidates"], [str(output / "audio.wav")])
        info = sf.info(str(output / "audio.wav"))
        single = 24000 + 0
        self.assertEqual(info.frames, 3 * single + 2 * round(0.3 * 48000) + (0 + 0 + 0))
        self.assertAlmostEqual(record["duration"], info.frames / 48000)
        self.assertEqual(read_json(output / "generation.json")["segments"], 3)
        # The three segments share one pass of four steps.
        self.assertEqual(runtime.batches, [3])
        progress = [v for kind, v in channel.events if kind == "progress"]
        self.assertEqual([v["step"] for v in progress], list(range(1, 5)))
        self.assertEqual({v["total"] for v in progress}, {4})

    def test_long_text_samples_at_most_a_batch_of_segments_per_pass(self):
        from irodori_tts.desktop import worker

        text = ("あ" * 99 + "。") * (worker.SEGMENT_BATCH + 1)
        _output, record, runtime, channel = self.run_task(request={"text": text, "seed": 3})
        self.assertEqual(runtime.batches, [worker.SEGMENT_BATCH, 1])
        self.assertEqual(
            [r.seed for r in runtime.requests], list(range(3, 3 + len(runtime.requests)))
        )
        progress = [v for kind, v in channel.events if kind == "progress"]
        self.assertEqual([v["step"] for v in progress], list(range(1, 9)))
        self.assertEqual({v["total"] for v in progress}, {8})

    def test_the_low_vram_model_reads_one_segment_per_pass(self):
        from irodori_tts.desktop import model_files

        runtime = FakeRuntime(model_files.low_vram_checkpoint())
        self.run_task(
            request={"text": ("あ" * 99 + "。") * 3}, model=model_files.LOW_VRAM, runtime=runtime
        )
        self.assertEqual(runtime.batches, [1, 1, 1])

    def test_a_pass_out_of_video_memory_is_repeated_one_segment_at_a_time(self):
        runtime = FakeRuntime()
        batch = runtime.synthesize_batch

        def tight(requests, **kwargs):
            if len(requests) > 1:
                raise torch.cuda.OutOfMemoryError("out of memory")
            return batch(requests, **kwargs)

        runtime.synthesize_batch = tight
        _output, record, _, channel = self.run_task(
            request={"text": ("あ" * 99 + "。") * 2, "seed": 7}, runtime=runtime
        )
        self.assertEqual(runtime.batches, [1, 1])
        self.assertEqual([r.seed for r in runtime.requests], [7, 8])
        progress = [v for kind, v in channel.events if kind == "progress"]
        self.assertEqual(progress[-1], {"step": 8, "total": 8})

    def test_a_random_first_seed_leads_the_following_ones(self):
        with patch("secrets.randbits", return_value=1000):
            _output, record, runtime, _ = self.run_task(
                request={"text": "あ" * 99 + "。" + "い" * 99 + "。", "seed": None}
            )
        self.assertEqual(record["seed"], 1000)
        self.assertEqual([r.seed for r in runtime.requests], [1000, 1001])

    def test_stopping_between_segments_leaves_no_audio(self):
        from irodori_tts.task_control import TaskCancelled

        channel = FakeChannel()
        # Five segments make two passes of four steps; the stop comes in the second.
        channel.stop_after = 5
        output = None
        with self.assertRaises(TaskCancelled):
            self.run_task(request={"text": ("あ" * 99 + "。") * 5, "num_steps": 4}, channel=channel)
        for path in Path(tempfile.gettempdir()).glob("tmp*/generations/g1/audio*.wav"):
            self.fail(f"left {path}")
        self.assertIsNone(output)

    def test_a_stop_leaves_no_audio(self):
        channel = FakeChannel()
        channel.stop_after = 3
        from irodori_tts.task_control import TaskCancelled

        with self.assertRaises(TaskCancelled):
            self.run_task(request={"num_candidates": 2}, channel=channel)


class LowVramTests(unittest.TestCase):
    def test_the_setting_reloads_the_runtime_on_the_chosen_model(self):
        from irodori_tts.desktop import model_files, worker

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        loaded = FakeRuntime(model_files.low_vram_checkpoint())
        old = FakeRuntime()
        payload = {
            "kind": "generate",
            "model": model_files.LOW_VRAM,
            "output": str(root / "out"),
            "request": {"text": "こんにちは", "num_steps": 1, "seed": 1},
        }
        channel = MagicMock(stopped=lambda: False)
        resources = {"model": "model", "codec": "codec"}
        with (
            patch(
                "irodori_tts.inference_runtime.InferenceRuntime.from_key", return_value=loaded
            ) as load,
            patch(
                "irodori_tts.inference_runtime.save_wav", side_effect=lambda p, *_: Path(p).touch()
            ),
            patch("torch.cuda.is_bf16_supported", return_value=True),
        ):
            runtime, _ = worker.synthesize(payload, channel, resources, old)
            self.assertIs(runtime, loaded)
            self.assertTrue(old.unloaded)
            self.assertEqual(load.call_args.args[0].checkpoint, model_files.low_vram_checkpoint())
            # The same choice keeps the loaded model.
            again, _ = worker.synthesize(payload, channel, resources, runtime)
            self.assertIs(again, loaded)
            self.assertEqual(load.call_count, 1)

    def test_generating_refuses_a_model_without_the_watermark(self):
        from irodori_tts.desktop import worker

        broken = FakeRuntime()
        broken.watermarker.ready = False
        payload = {"kind": "generate", "output": "unused", "request": {"text": "a"}}
        with (
            patch("irodori_tts.inference_runtime.InferenceRuntime.from_key", return_value=broken),
            patch("torch.cuda.is_bf16_supported", return_value=True),
            self.assertRaises(BusinessError),
        ):
            worker.synthesize(payload, MagicMock(), {"model": "m", "codec": "c"}, None)
        self.assertTrue(broken.unloaded)


class WatermarkCacheTests(unittest.TestCase):
    def test_main_points_at_the_pinned_revision_without_a_network(self):
        from irodori_tts.desktop.resources import WATERMARK_REVISION, pin_watermark_revision

        with tempfile.TemporaryDirectory() as base, patch.dict(os.environ, {"HF_HOME": base}):
            ref = Path(base) / "hub/models--sony--silentcipher/refs/main"
            pin_watermark_revision()
            self.assertEqual(ref.read_text("utf-8"), WATERMARK_REVISION)
            ref.write_text("a newer revision", "utf-8")
            pin_watermark_revision()
            self.assertEqual(ref.read_text("utf-8"), WATERMARK_REVISION)


if __name__ == "__main__":
    from tests.run import guarded_main

    raise SystemExit(guarded_main(lambda: unittest.main()))
