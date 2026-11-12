"""Validate business catalogs and compile Qt Linguist translations."""

from __future__ import annotations

import argparse
import ast
import re
import string
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LOCALES = REPO / "irodori_tts/desktop/locales"
# These are Japanese speech examples, not application UI wording.
SPEECH_EXAMPLES = {
    "こんにちは。今日はとてもいい天気ですね。ゆっくりお話ししましょう。",
    "こんにちは。今日はとてもいい天気ですね。",
    "おはようございます。今日はとてもいい天気ですね。",
    "こんにちは。今日はいい天気ですね。",
    "自然な声",
}


def validate():
    catalogs = {}
    formatter = string.Formatter()
    for language in ("en", "zh_CN"):
        tree = ET.parse(LOCALES / f"{language}.ts")
        values = {}
        for message in tree.findall(".//message"):
            key = message.findtext("source")
            translation = message.find("translation")
            if key in values or translation is None or translation.get("type") == "unfinished":
                raise ValueError(f"{language}: duplicate, empty or unfinished message: {key}")
            plural_forms = translation.findall("numerusform")
            if message.get("numerus") == "yes" and len(plural_forms) != (
                2 if language == "en" else 1
            ):
                raise ValueError(f"{language}: invalid plural-form count: {key}")
            forms = [v.text for v in plural_forms] or [translation.text]
            if any(not form for form in forms):
                raise ValueError(f"{language}: empty translation: {key}")
            values[key] = forms
        catalogs[language] = values
    if catalogs["en"].keys() != catalogs["zh_CN"].keys():
        raise ValueError("Catalog key sets differ")

    def fields(value):
        result = {name for _, name, _, _ in formatter.parse(value) if name is not None}
        if "%n" in value or "%Ln" in value:
            result.add("n")
        return result

    for key, forms in catalogs["en"].items():
        expected = fields(forms[0])
        for text in forms + catalogs["zh_CN"][key]:
            if fields(text) != expected:
                raise ValueError(f"Placeholder mismatch: {key}")
    references = {}
    files = list((REPO / "irodori_tts/desktop").glob("*.py"))
    files += [REPO / "irodori_tts/task_control.py"]
    for path in files:
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                visible = None
                if isinstance(node.func, ast.Name):
                    if (
                        node.func.id
                        in {
                            "label",
                            "button",
                            "ghost_button",
                            "field",
                            "search_box",
                            "Page",
                            "Expand",
                            "CheckBox",
                        }
                        and node.args
                    ):
                        visible = node.args[0]
                    elif node.func.id in {"set_text", "bind", "icon_button"} and len(node.args) > 1:
                        visible = node.args[1]
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr
                    in {
                        "setText",
                        "setPlaceholderText",
                        "setToolTip",
                        "setWindowTitle",
                        "setAccessibleName",
                        "setSuffix",
                        "setSpecialValueText",
                        "addRow",
                    }
                    and node.args
                ):
                    visible = node.args[0]
                if isinstance(visible, ast.Constant) and isinstance(visible.value, str):
                    value = visible.value
                    if (
                        value
                        and value not in SPEECH_EXAMPLES
                        and not re.fullmatch(r"[0-9:]+", value)
                    ):
                        raise ValueError(
                            f"Visible literal outside catalog: {path}:{node.lineno}: {value!r}"
                        )
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "t"
                and path.name != "messages.py"
            ):
                if not node.args or not isinstance(node.args[0], ast.Constant):
                    raise ValueError(f"Translation key must be literal: {path}:{node.lineno}")
                key = node.args[0].value
                if key not in catalogs["en"]:
                    raise ValueError(f"Missing key {key}: {path}:{node.lineno}")
                references[key] = references.get(key, 0) + 1
                required = fields(catalogs["en"][key][0])
                if {kw.arg for kw in node.keywords} != required:
                    raise ValueError(f"Argument mismatch {key}: {path}:{node.lineno}")
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if re.search("[\\u4e00-\\u9fff]", node.value) and node.value not in SPEECH_EXAMPLES:
                    raise ValueError(
                        f"Business wording outside catalog: {path}:{node.lineno}: {node.value!r}"
                    )
    return {
        "languages": list(catalogs),
        "messages_per_language": len(catalogs["en"]),
        "referenced_messages": len(references),
        "references": sum(references.values()),
    }


def compile_catalogs():
    import PySide6

    executable = Path(PySide6.__file__).parent / (
        "lrelease.exe" if sys.platform == "win32" else "lrelease"
    )
    for language in ("en", "zh_CN"):
        subprocess.run(
            [
                str(executable),
                str(LOCALES / f"{language}.ts"),
                "-qm",
                str(LOCALES / f"{language}.qm"),
            ],
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Validate without regenerating QM files"
    )
    args = parser.parse_args()
    result = validate()
    if not args.check:
        compile_catalogs()
    print(result)


if __name__ == "__main__":
    main()
