"""Qt Linguist catalogs and in-place property bindings for the existing widget tree."""

from __future__ import annotations

import weakref
from pathlib import Path

from PySide6.QtCore import (
    QCoreApplication,
    QDateTime,
    QLocale,
    QObject,
    QSignalBlocker,
    Qt,
    QTimer,
    QTranslator,
    Signal,
)
from PySide6.QtWidgets import QApplication, QLineEdit
from qfluentwidgets import FluentTranslator
from shiboken6 import isValid

from . import messages
from .messages import Message

_bindings = weakref.WeakKeyDictionary()


def format_date(text):
    value = QDateTime.fromString(text, Qt.DateFormat.ISODateWithMs)
    if not value.isValid():
        value = QDateTime.fromString(text, Qt.DateFormat.ISODate)
    return (
        QLocale().toString(value.toLocalTime(), QLocale.FormatType.ShortFormat)
        if value.isValid()
        else text[:16]
    )


def format_number(value, spec):
    if isinstance(value, int) and not spec:
        return QLocale().toString(value)
    if spec.endswith("f") and spec.startswith("."):
        return QLocale().toString(float(value), "f", int(spec[1:-1]))
    return format(value, spec)


def forget(widget, setter="setText"):
    """Drop a widget's translated binding, so a language switch cannot bring old text back."""
    bindings = _bindings.get(widget)
    if bindings:
        bindings.pop(setter, None)


def clear_text(widget):
    _bindings.pop(widget, None)
    widget.clear()


def system_language():
    """The catalog closest to the system locale: Chinese for zh*, otherwise English."""
    return "zh_CN" if QLocale.system().name().lower().startswith("zh") else "en"


def bind(widget, value, setter="setText"):
    bindings = _bindings.setdefault(widget, {})
    # Editable text is a draft, including a translated default name. Changing the
    # locale must never replace the user's input or move its cursor.
    editable = setter == "setText" and isinstance(widget, QLineEdit)
    if isinstance(value, Message) and not editable:
        bindings[setter] = value
    else:
        bindings.pop(setter, None)
    getattr(widget, setter)(str(value))
    return widget


def set_text(widget, value):
    return bind(widget, value)


def translated_widget(kind, value, *args, **kwargs):
    return bind(kind(str(value), *args, **kwargs), value)


def add_item(combo, text, **kwargs):
    combo.addItem(str(text), **kwargs)
    if isinstance(text, Message):
        entries = getattr(combo, "_message_items", None)
        if entries is None:
            entries = {}
            combo._message_items = entries
        entries[combo.count() - 1] = (text, kwargs.get("userData"))
    return combo.count() - 1


def clear_items(combo):
    combo._message_items = {}
    combo.clear()


class LocaleManager(QObject):
    changed = Signal(str)

    def __init__(self, app):
        super().__init__(app)
        self.app = app
        self.language = None
        self.translators = []
        self.english = QTranslator(self)
        self._load(self.english, "en")
        app.installTranslator(self.english)
        messages._renderer = lambda key, n: QCoreApplication.translate("Desktop", key, None, n)
        messages._formatter = format_number
        messages._date_formatter = format_date
        self.set_language("zh_CN")

    @staticmethod
    def _load(translator, language):
        path = Path(__file__).with_name("locales") / f"{language}.qm"
        if not translator.load(str(path)):
            raise RuntimeError(f"Missing compiled language catalog: {path}")

    def set_language(self, language):
        if language not in messages.LANGUAGES:
            raise ValueError(f"Unsupported application language: {language}")
        if language == self.language:
            return
        app = self.app
        focus = app.focusWidget()
        scrolls = [
            (weakref.ref(w), w.value()) for w in app.allWidgets() if w.inherits("QScrollBar")
        ]
        for translator in reversed(self.translators):
            app.removeTranslator(translator)
            translator.deleteLater()
        self.translators = []
        locale = QLocale("en_US" if language == "en" else language)
        QLocale.setDefault(locale)
        fluent = FluentTranslator(locale, self)
        app.installTranslator(fluent)
        self.translators.append(fluent)
        if language != "en":
            translator = QTranslator(self)
            self._load(translator, language)
            app.installTranslator(translator)
            self.translators.append(translator)
        self.language = messages._language = language
        for widget in app.allWidgets():
            widget.setLocale(locale)
        for widget, bindings in list(_bindings.items()):
            if not isValid(widget):
                continue
            for setter, value in list(bindings.items()):
                getattr(widget, setter)(str(value))
        for widget in app.allWidgets():
            entries = getattr(widget, "_message_items", {})
            if entries:
                blocker = QSignalBlocker(widget)
                for index, (text, data) in entries.items():
                    if index < widget.count() and widget.itemData(index) == data:
                        widget.setItemText(index, str(text))
                del blocker
        self.changed.emit(language)

        # Layout changes from longer captions must not reset an existing view.
        def restore_views():
            for reference, value in scrolls:
                widget = reference()
                if widget is not None:
                    try:
                        widget.setValue(value)
                    except RuntimeError:
                        pass

        restore_views()
        QTimer.singleShot(0, restore_views)
        if focus is not None:
            focus.setFocus()


def locale_manager():
    app = QApplication.instance()
    if not hasattr(app, "_locale_manager"):
        app._locale_manager = LocaleManager(app)
    return app._locale_manager
