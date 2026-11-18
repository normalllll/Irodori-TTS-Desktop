from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from irodori_tts.desktop.base_models import (
    ARCHITECTURE_KEYS,
    CATALOG,
    InstalledBase,
    architecture_of_config,
    base_record,
    catalog_base,
    import_base,
    load_registry,
    match_architecture,
    match_record,
    normalize_record,
    read_architecture,
    remove_base,
    save_base,
)
from irodori_tts.desktop.messages import BusinessError

SMALL = {
    "model_dim": 1280,
    "num_layers": 12,
    "num_heads": 20,
    "text_tokenizer_repo": "sbintuitions/modernbert-ja-310m",
    "text_dim": 512,
    "speaker_dim": 768,
    "speaker_layers": 8,
    "duration_hidden_dim": 1024,
    "duration_layers": 3,
}
LARGE = {
    "model_dim": 2048,
    "num_layers": 24,
    "num_heads": 32,
    "text_tokenizer_repo": "google/t5gemma-2-1b-1b",
    "text_dim": 1280,
    "speaker_dim": 1280,
    "speaker_layers": 14,
    "duration_hidden_dim": 1280,
    "duration_layers": 4,
}
OTHER = dict(SMALL, model_dim=640, num_layers=6)


def write_checkpoint(
    path: Path, config: dict | None, *, extra: dict | None = None, salt: float = 0.0
) -> Path:
    """Write a tiny real safetensors file whose header carries the given model config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"text_encoder_config_json": "{}"}
    if config is not None:
        metadata["config_json"] = json.dumps(config)
    metadata.update(extra or {})
    save_file({"w": torch.full((2,), salt)}, str(path), metadata=metadata)
    return path


def write_tokenizer(folder: Path, *, with_json: bool = True) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    if with_json:
        (folder / "tokenizer.json").write_text("{}", encoding="utf-8")


def installed(
    root: Path,
    base_id: str,
    architecture: dict,
    *,
    source: str = "catalog",
    revision: str | None = "rev1",
    repo: str | None = "Aratako/x",
    name: str | None = None,
    variants: dict | None = None,
) -> InstalledBase:
    checkpoint = write_checkpoint(root / "bases" / base_id / "model.safetensors", architecture)
    return InstalledBase(
        id=base_id,
        name=name or base_id,
        source=source,
        repo=repo,
        revision=revision,
        checkpoint=checkpoint,
        architecture=dict(architecture),
        variants=variants or {},
    )


class TempRootCase(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()


class ReadArchitectureTests(TempRootCase):
    def test_architecture_keys_are_the_nine_fingerprint_fields(self):
        self.assertEqual(set(ARCHITECTURE_KEYS), set(SMALL))
        self.assertEqual(len(ARCHITECTURE_KEYS), 9)

    def test_small_header(self):
        path = write_checkpoint(self.root / "s.safetensors", SMALL)
        self.assertEqual(read_architecture(path), SMALL)

    def test_large_header(self):
        path = write_checkpoint(self.root / "l.safetensors", LARGE)
        self.assertEqual(read_architecture(path), LARGE)

    def test_extra_config_keys_are_dropped(self):
        config = dict(SMALL, caption_vocab_size=99, unrelated="x")
        path = write_checkpoint(self.root / "e.safetensors", config)
        self.assertEqual(read_architecture(path), SMALL)

    def test_missing_config_json(self):
        path = write_checkpoint(self.root / "n.safetensors", None)
        self.assertIsNone(read_architecture(path))

    def test_missing_one_key(self):
        for key in ARCHITECTURE_KEYS:
            with self.subTest(key=key):
                config = {k: v for k, v in SMALL.items() if k != key}
                path = write_checkpoint(self.root / f"m_{key}.safetensors", config)
                self.assertIsNone(read_architecture(path))

    def test_not_a_safetensors_file(self):
        path = self.root / "junk.safetensors"
        path.write_bytes(b"this is not a safetensors file at all")
        self.assertIsNone(read_architecture(path))

    def test_empty_file(self):
        path = self.root / "empty.safetensors"
        path.write_bytes(b"")
        self.assertIsNone(read_architecture(path))

    def test_header_only_file_without_tensor_data(self):
        header = {
            "__metadata__": {"config_json": json.dumps(LARGE), "text_encoder_config_json": "{}"},
            "w": {"dtype": "F32", "shape": [1024, 1024], "data_offsets": [0, 4 * 1024 * 1024]},
        }
        raw = json.dumps(header).encode()
        path = self.root / "truncated.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw)
        self.assertEqual(read_architecture(path), LARGE)


class ArchitectureOfConfigTests(unittest.TestCase):
    def test_trainer_style_config_ignores_extra_keys(self):
        config = dict(LARGE, use_caption_condition=False, max_text_len=256, dropout=0.1)
        self.assertEqual(architecture_of_config(config), LARGE)

    def test_missing_key_gives_none(self):
        for key in ARCHITECTURE_KEYS:
            with self.subTest(key=key):
                self.assertIsNone(
                    architecture_of_config({k: v for k, v in SMALL.items() if k != key})
                )

    def test_empty_config(self):
        self.assertIsNone(architecture_of_config({}))


class CatalogTests(unittest.TestCase):
    def test_small_entry(self):
        small = catalog_base("v4.1-small")
        self.assertIsNotNone(small)
        self.assertEqual(small.id, "v4.1-small")
        self.assertEqual(small.repo, "Aratako/Irodori-TTS-v4.1-Small")
        self.assertEqual(small.architecture, SMALL)
        self.assertTrue(small.default)
        variants = {v.key: v for v in small.variants}
        self.assertEqual(set(variants), {"low_vram", "fast"})
        self.assertEqual(variants["low_vram"].repo, "Aratako/Irodori-TTS-v4.1-Small-Quantized")
        self.assertEqual(variants["low_vram"].model_file, "int8-weight-only/model.safetensors")
        self.assertIsNone(variants["low_vram"].steps)
        self.assertEqual(variants["fast"].repo, "Aratako/Irodori-TTS-v4.1-Small-MF")
        self.assertEqual(variants["fast"].model_file, "model.safetensors")
        self.assertEqual(variants["fast"].steps, 4)
        self.assertEqual(tuple(small.training_memory), (4.0, 1.0))

    def test_large_entry(self):
        large = catalog_base("v4-large")
        self.assertIsNotNone(large)
        self.assertEqual(large.id, "v4-large")
        self.assertEqual(large.repo, "Aratako/Irodori-TTS-v4-Large")
        self.assertEqual(large.architecture, LARGE)
        self.assertFalse(large.default)
        variants = {v.key: v for v in large.variants}
        self.assertEqual(set(variants), {"low_vram"})
        self.assertEqual(variants["low_vram"].repo, "Aratako/Irodori-TTS-v4-Large-Quantized")
        self.assertEqual(variants["low_vram"].model_file, "int8-weight-only/model.safetensors")
        self.assertEqual(tuple(large.training_memory), (13.5, 3.0))

    def test_catalog_files_include_model_and_tokenizer(self):
        for entry in CATALOG:
            with self.subTest(base=entry.id):
                self.assertIn("model.safetensors", entry.files)
                self.assertTrue(any(f.startswith("tokenizer/") for f in entry.files))

    def test_exactly_one_default_and_it_is_small(self):
        defaults = [entry.id for entry in CATALOG if entry.default]
        self.assertEqual(defaults, ["v4.1-small"])

    def test_ids_unique_and_lookup_roundtrips(self):
        ids = [entry.id for entry in CATALOG]
        self.assertEqual(len(ids), len(set(ids)))
        for entry in CATALOG:
            self.assertIs(catalog_base(entry.id), entry)

    def test_unknown_id(self):
        self.assertIsNone(catalog_base("nope"))
        self.assertIsNone(catalog_base(""))


class RegistryTests(TempRootCase):
    def test_empty_root(self):
        self.assertEqual(load_registry(self.root), [])

    def test_round_trip(self):
        quantized = write_checkpoint(self.root / "bases" / "q" / "model.safetensors", SMALL)
        base = installed(
            self.root,
            "v4.1-small",
            SMALL,
            name="Irodori-TTS v4.1 Small",
            variants={"low_vram": quantized},
        )
        save_base(self.root, base)
        loaded = load_registry(self.root)
        self.assertEqual(len(loaded), 1)
        got = loaded[0]
        self.assertEqual(got.id, base.id)
        self.assertEqual(got.name, base.name)
        self.assertEqual(got.source, "catalog")
        self.assertEqual(got.repo, base.repo)
        self.assertEqual(got.revision, "rev1")
        self.assertEqual(Path(got.checkpoint), base.checkpoint)
        self.assertEqual(got.architecture, SMALL)
        self.assertEqual({k: Path(v) for k, v in got.variants.items()}, {"low_vram": quantized})

    def test_registry_file_lives_in_the_root(self):
        save_base(self.root, installed(self.root, "v4.1-small", SMALL))
        data = json.loads((self.root / "base-models.json").read_text(encoding="utf-8"))
        self.assertEqual(data["format_version"], 1)
        self.assertEqual(list(data["bases"]), ["v4.1-small"])

    def test_imported_base_with_no_repo_or_revision(self):
        base = installed(
            self.root, "imported-abcdef012345", OTHER, source="imported", repo=None, revision=None
        )
        save_base(self.root, base)
        got = load_registry(self.root)[0]
        self.assertEqual(got.source, "imported")
        self.assertIsNone(got.repo)
        self.assertIsNone(got.revision)

    def test_saving_same_id_replaces_entry(self):
        save_base(self.root, installed(self.root, "v4.1-small", SMALL, revision="old"))
        save_base(self.root, installed(self.root, "v4.1-small", SMALL, revision="new"))
        loaded = load_registry(self.root)
        self.assertEqual([b.revision for b in loaded], ["new"])

    def test_multiple_bases(self):
        save_base(self.root, installed(self.root, "v4.1-small", SMALL))
        save_base(self.root, installed(self.root, "v4-large", LARGE))
        self.assertEqual({b.id for b in load_registry(self.root)}, {"v4.1-small", "v4-large"})

    def test_missing_checkpoint_is_omitted(self):
        keep = installed(self.root, "v4.1-small", SMALL)
        gone = installed(self.root, "v4-large", LARGE)
        save_base(self.root, keep)
        save_base(self.root, gone)
        gone.checkpoint.unlink()
        self.assertEqual([b.id for b in load_registry(self.root)], ["v4.1-small"])

    def test_remove_base(self):
        save_base(self.root, installed(self.root, "v4.1-small", SMALL))
        save_base(self.root, installed(self.root, "v4-large", LARGE))
        remove_base(self.root, "v4-large")
        self.assertEqual([b.id for b in load_registry(self.root)], ["v4.1-small"])

    def test_remove_unknown_base_is_harmless(self):
        save_base(self.root, installed(self.root, "v4.1-small", SMALL))
        remove_base(self.root, "does-not-exist")
        self.assertEqual([b.id for b in load_registry(self.root)], ["v4.1-small"])

    def test_remove_from_empty_root_is_harmless(self):
        remove_base(self.root, "anything")
        self.assertEqual(load_registry(self.root), [])


class RecordTests(TempRootCase):
    def test_base_record_shape(self):
        base = installed(self.root, "v4.1-small", SMALL, name="Small", revision="abc")
        record = base_record(base)
        self.assertEqual(
            record,
            {
                "id": "v4.1-small",
                "name": "Small",
                "repo": "Aratako/x",
                "revision": "abc",
                "architecture": SMALL,
            },
        )

    def test_base_record_is_json_serializable(self):
        base = installed(
            self.root, "imported-abcdef012345", OTHER, source="imported", repo=None, revision=None
        )
        record = base_record(base)
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertIsNone(record["repo"])
        self.assertIsNone(record["revision"])

    def test_normalize_legacy_small_record(self):
        record = normalize_record(
            {"repo": "Aratako/Irodori-TTS-v4.1-Small", "revision": "deadbeef"}
        )
        self.assertEqual(record["id"], "v4.1-small")
        self.assertEqual(record["architecture"], SMALL)
        self.assertEqual(record["repo"], "Aratako/Irodori-TTS-v4.1-Small")
        self.assertEqual(record["revision"], "deadbeef")
        self.assertTrue(record["name"])

    def test_normalize_legacy_large_record(self):
        record = normalize_record({"repo": "Aratako/Irodori-TTS-v4-Large", "revision": "r"})
        self.assertEqual(record["id"], "v4-large")
        self.assertEqual(record["architecture"], LARGE)

    def test_normalize_unknown_legacy_repo_stays_unresolved(self):
        record = normalize_record({"repo": "someone/else", "revision": "r"})
        if record is not None:
            self.assertFalse(record.get("architecture"))
            self.assertFalse(record.get("id"))

    def test_normalize_non_dict(self):
        for value in (None, "Aratako/Irodori-TTS-v4.1-Small", 3, ["x"]):
            with self.subTest(value=value):
                self.assertIsNone(normalize_record(value))

    def test_normalize_keeps_record_with_architecture(self):
        given = {
            "id": "imported-abcdef012345",
            "name": "Mine",
            "repo": None,
            "revision": None,
            "architecture": OTHER,
        }
        record = normalize_record(dict(given))
        self.assertEqual(record["id"], given["id"])
        self.assertEqual(record["name"], "Mine")
        self.assertEqual(record["architecture"], OTHER)


class MatchTests(TempRootCase):
    def small(self, **kw):
        return installed(self.root, kw.pop("base_id", "v4.1-small"), SMALL, **kw)

    def record(self, base_id="v4.1-small", revision="rev1", architecture=SMALL):
        return {
            "id": base_id,
            "name": base_id,
            "repo": "Aratako/x",
            "revision": revision,
            "architecture": architecture,
        }

    def test_exact(self):
        base = self.small()
        match = match_record(self.record(), [base])
        self.assertEqual(match.status, "exact")
        self.assertEqual(match.base.id, "v4.1-small")

    def test_other_revision_is_compatible(self):
        base = self.small(revision="newer")
        match = match_record(self.record(revision="older"), [base])
        self.assertEqual(match.status, "compatible")
        self.assertEqual(match.base.id, "v4.1-small")

    def test_other_id_same_architecture_is_compatible(self):
        imported = installed(
            self.root, "imported-aaaaaaaaaaaa", SMALL, source="imported", repo=None, revision=None
        )
        match = match_record(self.record(), [imported])
        self.assertEqual(match.status, "compatible")
        self.assertEqual(match.base.id, "imported-aaaaaaaaaaaa")

    def test_compatible_is_not_decided_by_name_or_repo(self):
        base = self.small(repo="other/repo", name="Renamed")
        self.assertEqual(match_record(self.record(), [base]).status, "exact")

    def test_missing_names_the_catalog_base_that_is_needed(self):
        base = self.small()
        match = match_record(self.record("v4-large", architecture=LARGE), [base])
        self.assertEqual(match.status, "missing")
        self.assertIsNone(match.base)
        self.assertIsNotNone(match.needs)
        self.assertEqual(match.needs.id, "v4-large")

    def test_missing_with_no_bases_installed(self):
        match = match_record(self.record(), [])
        self.assertEqual(match.status, "missing")
        self.assertEqual(match.needs.id, "v4.1-small")

    def test_missing_with_unknown_architecture_needs_nothing(self):
        base = self.small()
        match = match_record(self.record("imported-bbbbbbbbbbbb", architecture=OTHER), [base])
        self.assertEqual(match.status, "missing")
        self.assertIsNone(match.base)
        self.assertIsNone(match.needs)

    def test_unknown_when_no_architecture_and_not_normalizable(self):
        base = self.small()
        for record in ({"repo": "someone/else", "revision": "r"}, {}):
            with self.subTest(record=record):
                match = match_record(record, [base])
                self.assertEqual(match.status, "unknown")
                self.assertIsNone(match.base)

    def test_legacy_record_is_normalized_before_matching(self):
        base = self.small(revision="rev1")
        match = match_record({"repo": "Aratako/Irodori-TTS-v4.1-Small", "revision": "rev1"}, [base])
        self.assertEqual(match.status, "exact")
        legacy_other = match_record(
            {"repo": "Aratako/Irodori-TTS-v4.1-Small", "revision": "zzz"}, [base]
        )
        self.assertEqual(legacy_other.status, "compatible")

    def test_legacy_large_record_with_only_small_installed(self):
        match = match_record(
            {"repo": "Aratako/Irodori-TTS-v4-Large", "revision": "r"}, [self.small()]
        )
        self.assertEqual(match.status, "missing")
        self.assertEqual(match.needs.id, "v4-large")

    def test_exact_preferred_over_compatible(self):
        compatible = installed(
            self.root, "imported-cccccccccccc", SMALL, source="imported", repo=None, revision=None
        )
        exact = self.small(revision="rev1")
        for order in ([compatible, exact], [exact, compatible]):
            match = match_record(self.record(revision="rev1"), order)
            self.assertEqual(match.status, "exact")
            self.assertEqual(match.base.id, "v4.1-small")

    def test_catalog_preferred_over_imported_when_compatible(self):
        imported = installed(
            self.root, "imported-dddddddddddd", SMALL, source="imported", repo=None, revision=None
        )
        catalog = self.small(revision="other")
        for order in ([imported, catalog], [catalog, imported]):
            match = match_record(self.record(revision="rev1"), order)
            self.assertEqual(match.status, "compatible")
            self.assertEqual(match.base.id, "v4.1-small")

    def test_only_the_matching_architecture_is_chosen(self):
        small = self.small()
        large = installed(self.root, "v4-large", LARGE)
        match = match_record(self.record("v4-large", architecture=LARGE), [small, large])
        self.assertEqual(match.status, "exact")
        self.assertEqual(match.base.id, "v4-large")

    def test_match_architecture_found(self):
        small = self.small()
        large = installed(self.root, "v4-large", LARGE)
        match = match_architecture(LARGE, [small, large])
        self.assertIn(match.status, ("exact", "compatible"))
        self.assertEqual(match.base.id, "v4-large")

    def test_match_architecture_prefers_catalog_over_imported(self):
        imported = installed(
            self.root, "imported-eeeeeeeeeeee", SMALL, source="imported", repo=None, revision=None
        )
        catalog = self.small()
        for order in ([imported, catalog], [catalog, imported]):
            self.assertEqual(match_architecture(SMALL, order).base.id, "v4.1-small")

    def test_match_architecture_missing_with_catalog_base(self):
        match = match_architecture(LARGE, [self.small()])
        self.assertEqual(match.status, "missing")
        self.assertIsNone(match.base)
        self.assertEqual(match.needs.id, "v4-large")

    def test_match_architecture_missing_for_unknown_architecture(self):
        match = match_architecture(OTHER, [self.small()])
        self.assertEqual(match.status, "missing")
        self.assertIsNone(match.needs)

    def test_match_architecture_none_is_unknown(self):
        match = match_architecture(None, [self.small()])
        self.assertEqual(match.status, "unknown")
        self.assertIsNone(match.base)


class ImportBaseTests(TempRootCase):
    def make_source(self, name="my-base", config=SMALL, *, tokenizer=True, salt=0.0) -> Path:
        folder = self.root / "src" / name
        write_checkpoint(folder / "model.safetensors", config, salt=salt)
        if tokenizer:
            write_tokenizer(folder / "tokenizer")
        return folder

    def setUp(self):
        super().setUp()
        self.dest = self.root / "resources"
        self.dest.mkdir()

    def test_import_from_folder(self):
        folder = self.make_source()
        base = import_base(folder, self.dest)
        self.assertRegex(base.id, r"^imported-[0-9a-f]{12}$")
        self.assertEqual(base.source, "imported")
        self.assertEqual(base.architecture, SMALL)
        self.assertEqual(base.name, "my-base")
        target = self.dest / "bases" / base.id
        self.assertEqual(Path(base.checkpoint), target / "model.safetensors")
        self.assertTrue((target / "model.safetensors").is_file())
        self.assertTrue((target / "tokenizer" / "tokenizer_config.json").is_file())
        self.assertTrue((target / "tokenizer" / "tokenizer.json").is_file())
        self.assertEqual(read_architecture(target / "model.safetensors"), SMALL)

    def test_import_registers_the_base(self):
        base = import_base(self.make_source(), self.dest)
        loaded = load_registry(self.dest)
        self.assertEqual([b.id for b in loaded], [base.id])
        self.assertEqual(loaded[0].source, "imported")
        self.assertEqual(loaded[0].architecture, SMALL)

    def test_import_from_model_file_path(self):
        folder = self.make_source()
        base = import_base(folder / "model.safetensors", self.dest)
        target = self.dest / "bases" / base.id
        self.assertEqual(base.source, "imported")
        self.assertEqual(base.architecture, SMALL)
        self.assertTrue((target / "model.safetensors").is_file())
        self.assertTrue((target / "tokenizer" / "tokenizer_config.json").is_file())

    def test_import_large_shaped_base_is_its_own_base(self):
        base = import_base(self.make_source("big", LARGE), self.dest)
        self.assertEqual(base.architecture, LARGE)
        self.assertTrue(base.id.startswith("imported-"))
        self.assertNotEqual(base.id, "v4-large")

    def test_import_small_shaped_base_is_not_the_catalog_base(self):
        base = import_base(self.make_source(), self.dest)
        self.assertNotEqual(base.id, "v4.1-small")
        self.assertEqual(base.source, "imported")

    def test_import_accepts_quantized_checkpoint(self):
        folder = self.root / "src" / "quant"
        write_checkpoint(
            folder / "model.safetensors",
            SMALL,
            extra={"quantization": json.dumps({"w": {"scheme": "int8"}})},
        )
        write_tokenizer(folder / "tokenizer", with_json=False)
        base = import_base(folder, self.dest)
        self.assertEqual(base.architecture, SMALL)

    def test_same_checkpoint_gets_same_id(self):
        first = import_base(self.make_source("a"), self.dest)
        second = import_base(self.make_source("b"), self.dest)
        self.assertEqual(first.id, second.id)
        self.assertEqual(len(load_registry(self.dest)), 1)

    def test_different_headers_get_different_ids(self):
        first = import_base(self.make_source("a", SMALL), self.dest)
        second = import_base(self.make_source("b", LARGE), self.dest)
        self.assertNotEqual(first.id, second.id)
        self.assertEqual(len(load_registry(self.dest)), 2)

    def test_import_rejects_file_without_config_json(self):
        folder = self.root / "src" / "noconfig"
        write_checkpoint(folder / "model.safetensors", None)
        write_tokenizer(folder / "tokenizer")
        with self.assertRaises(BusinessError):
            import_base(folder, self.dest)
        with self.assertRaises(BusinessError):
            import_base(folder / "model.safetensors", self.dest)
        self.assertEqual(load_registry(self.dest), [])

    def test_import_rejects_incomplete_architecture(self):
        config = {k: v for k, v in SMALL.items() if k != "speaker_layers"}
        folder = self.make_source("partial", config)
        with self.assertRaises(BusinessError):
            import_base(folder, self.dest)
        self.assertEqual(load_registry(self.dest), [])

    def test_import_rejects_folder_without_tokenizer(self):
        folder = self.make_source(tokenizer=False)
        with self.assertRaises(BusinessError):
            import_base(folder, self.dest)
        with self.assertRaises(BusinessError):
            import_base(folder / "model.safetensors", self.dest)
        self.assertEqual(load_registry(self.dest), [])

    def test_import_rejects_tokenizer_folder_without_config(self):
        folder = self.make_source(tokenizer=False)
        (folder / "tokenizer").mkdir()
        (folder / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(BusinessError):
            import_base(folder, self.dest)

    def test_import_rejects_non_safetensors_file(self):
        folder = self.root / "src" / "junk"
        folder.mkdir(parents=True)
        (folder / "model.safetensors").write_bytes(b"not safetensors")
        write_tokenizer(folder / "tokenizer")
        with self.assertRaises(BusinessError):
            import_base(folder, self.dest)


if __name__ == "__main__":
    unittest.main()
