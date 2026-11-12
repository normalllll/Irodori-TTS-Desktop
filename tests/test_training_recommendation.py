"""Fast planning and its no-training boundary."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from irodori_tts.desktop.messages import BusinessError
from irodori_tts.desktop.storage import create_project, load_project, save_project
from irodori_tts.desktop.training_recommendation import (
    material_profile,
    plan_training,
    recommend_training,
)
from irodori_tts.task_control import TaskCancelled


def project_with_lengths(lengths):
    return {
        "materials": [
            {"id": str(i), "included": True, "text": f"sentence {i}", "duration": seconds}
            for i, seconds in enumerate(lengths)
        ]
    }


def gpu(free=12, capability=8.9):
    return {
        "name": "test card",
        "total_mib": 16 * 1024,
        "used_mib": (16 - free) * 1024,
        "compute_cap": capability,
    }


class PlanningTests(unittest.TestCase):
    def test_duration_and_available_memory_change_layout_without_changing_quality_budget(self):
        short = plan_training(project_with_lengths([3] * 50), {}, gpu())
        long = plan_training(project_with_lengths([12] * 50), {}, gpu())
        busy = plan_training(project_with_lengths([3] * 50), {}, gpu(free=7))
        self.assertGreater(short["values"]["batch_size"], long["values"]["batch_size"])
        self.assertGreater(short["values"]["batch_size"], busy["values"]["batch_size"])
        for result in (short, long, busy):
            values = result["values"]
            self.assertEqual(values["batch_size"] * values["gradient_accumulation_steps"], 8)
            self.assertEqual(values["max_steps"], 625)
        self.assertGreater(long["profile"]["total_seconds"], short["profile"]["total_seconds"])

    def test_rare_long_recording_is_not_hidden_by_percentiles(self):
        short = project_with_lengths([1] * 100)
        tail = project_with_lengths([1] * 99 + [12])
        result = plan_training(tail, {}, gpu())
        self.assertEqual(result["profile"]["p90_seconds"], 1)
        self.assertEqual(result["profile"]["maximum_seconds"], 12)
        self.assertLess(
            result["values"]["batch_size"], plan_training(short, {}, gpu())["values"]["batch_size"]
        )

    def test_unknown_low_memory_fp32_and_expanded_training_use_conservative_layouts(self):
        project = project_with_lengths([12] * 50)
        for device, settings in (
            (None, {}),
            (gpu(free=2), {}),
            (gpu(capability=7), {}),
            (gpu(), {"lora_r": 64}),
            (gpu(), {"lora_enabled": False}),
        ):
            values = plan_training(project, settings, device)["values"]
            self.assertEqual(values["batch_size"], 1)
            self.assertEqual(values["gradient_accumulation_steps"], 8)
            self.assertTrue(values["gradient_checkpointing"])

    def test_validation_changes_budget_count_but_statistics_cover_all_included_materials(self):
        project = project_with_lengths([2] * 20 + [12])
        project["materials"][-1]["included"] = False
        result = plan_training(project, {"valid_every": 10, "valid_ratio": 0.25}, gpu())
        self.assertEqual(result["training_count"], 15)
        self.assertEqual(result["profile"]["count"], 20)
        self.assertEqual(result["profile"]["maximum_seconds"], 2)
        self.assertEqual(result["values"]["max_steps"], 188)

    def test_small_projects_do_not_repeat_recordings_to_fill_a_large_batch(self):
        for count in range(1, 20):
            with self.subTest(count=count):
                values = plan_training(project_with_lengths([3] * count), {}, gpu())["values"]
                self.assertLessEqual(
                    values["batch_size"] * values["gradient_accumulation_steps"], count
                )
                self.assertLess(values["warmup_steps"], values["max_steps"])
        for lengths in ([], [0], [float("nan")], [float("inf")]):
            with self.assertRaises(BusinessError):
                material_profile(project_with_lengths(lengths)["materials"])


class RecommendationWorkerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = create_project(Path(temporary.name), "recordings")
        project = load_project(self.root)
        project.update(project_with_lengths([3] * 50))
        save_project(self.root, project)
        self.channel = Mock()
        self.channel.stopped.return_value = False

    def test_worker_needs_no_models_preprocessing_or_trainer(self):
        import sys

        from irodori_tts.desktop.worker import execute

        with (
            patch("irodori_tts.desktop.worker.TaskChannel", return_value=self.channel),
            patch(
                "irodori_tts.desktop.worker.resolve_resources",
                side_effect=AssertionError("model access"),
            ),
            patch(
                "irodori_tts.desktop.worker.initialize", side_effect=AssertionError("preprocessing")
            ),
            patch("irodori_tts.desktop.exports.device_report", return_value={"gpu": gpu()}),
            patch.dict(sys.modules, {"train": None, "torch": None}),
        ):
            execute(
                {
                    "kind": "recommend",
                    "project": str(self.root),
                    "settings": {},
                    "resource_root": "missing",
                }
            )
        results = [
            call.kwargs["result"]
            for call in self.channel.emit.call_args_list
            if call.args == ("result",)
        ]
        self.assertEqual(len(results), 1, self.channel.emit.call_args_list)
        self.assertEqual(results[0]["values"]["max_steps"], 625)
        self.assertFalse((self.root / "initialized").exists())
        self.assertFalse(list((self.root / "runs").iterdir()))
        self.assertFalse(list((self.root / "temp").iterdir()))

    def test_cancel_does_not_publish_a_plan(self):
        self.channel.stopped.return_value = True
        with self.assertRaises(TaskCancelled):
            recommend_training({"project": str(self.root), "settings": {}}, self.channel)


if __name__ == "__main__":
    unittest.main()
