"""Locale-neutral business messages; workers do not import Qt."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

LANGUAGES = ("en", "zh_CN")
_language = "zh_CN"
_renderer = None
_formatter = None
_date_formatter = None


@lru_cache(maxsize=2)
def catalog(language):
    tree = ET.parse(Path(__file__).with_name("locales") / f"{language}.ts")
    return {
        m.findtext("source"): (
            [v.text for v in m.findall("translation/numerusform")]
            if m.get("numerus") == "yes"
            else m.findtext("translation")
        )
        for m in tree.findall(".//message")
    }


def render(message):
    # Persisted records can outlive a catalog key or its placeholders (an older or newer
    # app wrote them). Rendering never raises: it falls back to the key or raw template.
    if message.key not in catalog("en"):
        return message.key
    # The plural count may arrive formatted (fmt(n)); Qt's translate takes a plain int.
    count = message.params.get("n", -1)
    count = count.value if isinstance(count, Formatted) else count
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = -1
    if _renderer:
        template = _renderer(message.key, count)
    else:
        template = catalog(_language).get(message.key, catalog("en").get(message.key))
    if isinstance(template, list):
        count = max(count, 0)
        template = template[0 if _language == "zh_CN" or count == 1 else 1]
        template = template.replace("%Ln", str(count)).replace("%n", str(count))
    if template is None:
        return message.key
    params = {
        key: str(value) if isinstance(value, (Message, Formatted)) else value
        for key, value in message.params.items()
    }
    try:
        if message.key == "common.date" and _date_formatter:
            params["value"] = _date_formatter(params["value"])
        return template.format(**params)
    except (KeyError, IndexError, ValueError, TypeError, AttributeError):
        return template


@dataclass(frozen=True)
class Formatted:
    value: object
    spec: str = ""
    conversion: int = -1

    def __str__(self):
        if self.conversion == 114:
            return format(repr(self.value), self.spec)
        if self.conversion == 97:
            return format(ascii(self.value), self.spec)
        if _formatter and isinstance(self.value, (int, float)):
            return _formatter(self.value, self.spec)
        return format(str(self.value) if isinstance(self.value, Message) else self.value, self.spec)


def fmt(value, spec="", conversion=-1):
    if isinstance(value, BaseException):
        value = error_message(value)
    if not isinstance(value, (int, float, str, Message)):
        value = str(value)
    return Formatted(value, spec, conversion)


class Message(str):
    """A string-compatible message that retains its key until the display boundary."""

    def __new__(cls, key, **params):
        obj = super().__new__(cls, "")
        obj.key, obj.params = key, params
        # A real initial string keeps JSON/Qt consumers interoperable; persistence
        # and widget bindings retain the key separately for subsequent rendering.
        obj = super().__new__(cls, render(obj))
        obj.key, obj.params = key, params
        return obj

    def __str__(self):
        return render(self)

    def __add__(self, other):
        return t("common.concatenate", p0=self, p1=other)

    def __radd__(self, other):
        return t("common.concatenate", p0=other, p1=self)


def t(key, **params):
    return Message(key, **params)


class BusinessError(ValueError):
    def __init__(self, message):
        self.message = message
        super().__init__(str(message))

    def __str__(self):
        return str(self.message)


def error_message(error):
    return getattr(error, "message", str(error))


def pack(value):
    if isinstance(value, Formatted):
        return {"$format": pack(value.value), "spec": value.spec, "conversion": value.conversion}
    if isinstance(value, Message):
        return {"$message": value.key, "params": {k: pack(v) for k, v in value.params.items()}}
    if isinstance(value, dict):
        return {k: pack(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [pack(v) for v in value]
    return value


def unpack(value):
    if isinstance(value, dict):
        if "$format" in value:
            return fmt(unpack(value["$format"]), value["spec"], value["conversion"])
        if "$message" in value:
            return t(
                value["$message"], **{k: unpack(v) for k, v in value.get("params", {}).items()}
            )
        return {k: unpack(v) for k, v in value.items()}
    if isinstance(value, list):
        return [unpack(v) for v in value]
    return value


def persist(value):
    """Keep existing readable fields and attach message identity as metadata."""
    if isinstance(value, dict):
        result = {k: persist(v) for k, v in value.items()}
        messages = {}
        for key, item in value.items():
            if isinstance(item, Message):
                result[key] = str(item)
                # Names and model inputs become user data at their creation boundary.
                if key not in {"name", "voice", "text", "caption", "file_name", "path"}:
                    messages[key] = pack(item)
        if messages:
            result["_i18n"] = {"format": "irodori.messages.v1", "fields": messages}
        return result
    if isinstance(value, (list, tuple)):
        return [persist(v) for v in value]
    return str(value) if isinstance(value, (Message, Formatted)) else value


def restore(value):
    if isinstance(value, dict):
        metadata = value.get("_i18n")
        localized = isinstance(metadata, dict) and metadata.get("format") == "irodori.messages.v1"
        result = {k: restore(v) for k, v in value.items() if not (localized and k == "_i18n")}
        if localized:
            for key, message in (metadata.get("fields") or {}).items():
                try:
                    result[key] = unpack(message)
                except (KeyError, TypeError, AttributeError):
                    continue
        return result
    if isinstance(value, list):
        return [restore(v) for v in value]
    return value
