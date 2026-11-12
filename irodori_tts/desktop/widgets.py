from __future__ import annotations

import re
import threading
import weakref
from enum import Enum
from pathlib import Path

from PySide6.QtCore import (
    Q_ARG,
    QAbstractListModel,
    QAbstractTableModel,
    QEasingCurve,
    QEvent,
    QMetaObject,
    QModelIndex,
    QObject,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QRect,
    QRectF,
    QSize,
    QStandardPaths,
    Qt,
    QThreadPool,
    QTimer,
    QUrl,
    QVariantAnimation,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDesktopServices,
    QFont,
    QFontDatabase,
    QIcon,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QResizeEvent,
    QValidator,
)
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QApplication,
    QBoxLayout,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QStyleOptionViewItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import (
    Action,
    BodyLabel,
    CaptionLabel,
    DoubleSpinBox,
    DropDownPushButton,
    FluentIcon,
    FluentIconBase,
    InfoBar,
    InfoBarPosition,
    LineEdit,
    ListView,
    MessageBox,
    MessageBoxBase,
    PlainTextEdit,
    PrimaryPushButton,
    PushButton,
    RadioButton,
    ScrollArea,
    SearchLineEdit,
    SimpleCardWidget,
    SpinBox,
    StrongBodyLabel,
    SubtitleLabel,
    SwitchButton,
    TableItemDelegate,
    TableView,
    Theme,
    TitleLabel,
    ToolTipFilter,
    TransparentPushButton,
    TransparentTogglePushButton,
    TransparentToolButton,
    getFont,
    isDarkTheme,
    setCustomStyleSheet,
    setFont,
    setFontFamilies,
    setTheme,
    setThemeColor,
)
from qfluentwidgets.common import style_sheet as _style_sheet
from qfluentwidgets.common.auto_wrap import TextWrap
from qfluentwidgets.common.icon import getIconColor
from qfluentwidgets.components.material import AcrylicComboBox
from qfluentwidgets.components.material.acrylic_combo_box import (
    AcrylicComboMenuActionListWidget,
)
from qfluentwidgets.components.material.acrylic_menu import (
    AcrylicMenuActionListWidget,
    AcrylicMenuBase,
)
from qfluentwidgets.components.widgets.check_box import CheckBoxIcon
from qfluentwidgets.components.widgets.menu import (
    IndicatorMenuItemDelegate,
    MenuAnimationType,
    RoundMenu,
)
from qfluentwidgets.multimedia import StandardMediaPlayBar
from shiboken6 import isValid

from irodori_tts.desktop.messages import t

from .i18n import bind, forget, locale_manager, set_text, translated_widget

QWIDGETSIZE_MAX = 16777215
_read_style_sheet = _style_sheet.getStyleSheetFromFile


def _style_sheet_with_white_selection(file):
    """The library's style sheets, with selected text white on the theme colour: its dark
    sheets drew it black and its light ones left it to the palette."""
    qss = re.sub(r"selection-color:\s*black;", "", _read_style_sheet(file))
    return re.sub(r"(selection-background-color:[^;]*;)", r"\g<1> selection-color: white;", qss)


_style_sheet.getStyleSheetFromFile = _style_sheet_with_white_selection


def apply_theme(app: QApplication, preference: str):
    if not QFontDatabase.families():
        # Qt's offscreen platform needs explicit system-font registration.
        from pathlib import Path

        for file in ("msyh.ttc", "msyhbd.ttc"):
            font = Path("C:/Windows/Fonts") / file
            if font.exists():
                QFontDatabase.addApplicationFont(str(font))
    # The system UI fonts, hinted for the screen at these sizes. Noto Sans SC was tried: it has
    # no hinting (soft at 14 px) and a taller line, which clipped text sized for these.
    setFontFamilies(["Microsoft YaHei UI", "Yu Gothic UI", "Segoe UI"])
    app.setFont(getFont(14))
    setTheme(
        {"system": Theme.AUTO, "light": Theme.LIGHT, "dark": Theme.DARK}[preference], save=False
    )
    setThemeColor(ACCENTS[current_accent()]["color"], save=False)
    # Paint the native window background before Qt's first frame, so a dark start
    # does not flash white.
    palette = app.palette()
    palette.setColor(
        palette.ColorRole.Window, QColor(32, 32, 32) if isDarkTheme() else QColor(240, 244, 249)
    )
    # Selected text in fields the style sheets leave to the palette (a spin box's number).
    palette.setColor(palette.ColorRole.HighlightedText, ON_ACCENT)
    app.setPalette(palette)


# Theme colours to choose from in Settings: the icon's aqua and pink, a little more saturated
# and a touch darker so a button reads as one (the icon's own were too pale for a fill), used
# everywhere (fills, lines, curves, text), with white text and glyphs on fills. The library's
# dark theme would lighten the colour and draw black on it. Hover and pressed are a shade
# lighter and darker.
ACCENTS = {
    "aqua": {"color": "#27b7ed", "hover": "#3cc3f5", "pressed": "#1ba7db"},
    "pink": {"color": "#e66985", "hover": "#ed8098", "pressed": "#d45874"},
}
ON_ACCENT = QColor("#ffffff")
# The current fill. Modules keep this object, so a change of colour updates it in place.
ACCENT = QColor(ACCENTS["aqua"]["color"])
_accent = "aqua"
# Widgets whose look was set from the accent: {widget: how to set it again}.
_accented = weakref.WeakKeyDictionary()
# Pictures drawn once with the accent (step markers): {owner: how to draw it again}.
_accent_drawings = weakref.WeakKeyDictionary()


def accent_drawn(owner, draw):
    """Run draw() now, and again whenever the theme colour changes: for a picture made with
    the accent and kept (a step marker's pixmap), which a repaint would not renew."""
    draw()
    _accent_drawings[owner] = draw


def swatch(name) -> QIcon:
    """A round sample of a theme colour, for the choice in Settings."""
    pixmap = QPixmap(32, 32)
    pixmap.setDevicePixelRatio(2)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QPen(QColor(0, 0, 0, 40), 1))
    painter.setBrush(QColor(ACCENTS[name]["color"]))
    painter.drawEllipse(QRectF(1.5, 1.5, 13, 13))
    painter.end()
    return QIcon(pixmap)


def current_accent() -> str:
    return _accent


def accent_line(dark=None) -> QColor:
    """The accent for lines, curves and text (the same colour on light and dark; dark is
    accepted for callers that name the theme)."""
    return QColor(ACCENTS[_accent]["color"])


def set_accent(name):
    """Use another theme colour everywhere, open windows included."""
    global _accent
    _accent = name if name in ACCENTS else "aqua"
    ACCENT.setRgb(*QColor(ACCENTS[_accent]["color"]).getRgb()[:3])
    # Lazily: widgets out of sight take the new style sheet when they are next shown, so the
    # switch does not stall on restyling every page at once.
    setThemeColor(ACCENTS[_accent]["color"], save=False, lazy=True)
    for widget, restyle in list(_accented.items()):
        if isValid(widget):
            restyle(widget)
            widget.update()
    for owner, draw in list(_accent_drawings.items()):
        if isValid(owner):
            draw()
    for widget in QApplication.topLevelWidgets():
        widget.update()


def _accent_qss(kind, dark):
    c = ACCENTS[_accent]
    ink = "white"
    disabled = (
        "color: rgba(255, 255, 255, 0.43); background-color: rgb(52, 52, 52);"
        " border: 1px solid rgb(52, 52, 52);"
        if dark
        else "color: rgba(255, 255, 255, 0.9); background-color: rgb(205, 205, 205);"
        " border: 1px solid rgb(205, 205, 205);"
    )
    if kind == "toggle":
        return f"""
TransparentTogglePushButton:checked {{
    color: {ink}; background-color: {c["color"]}; border: none;
}}
TransparentTogglePushButton:checked:hover {{ background-color: {c["hover"]}; }}
TransparentTogglePushButton:checked:pressed {{
    color: rgba(255, 255, 255, 0.78); background-color: {c["pressed"]};
}}
"""
    return f"""
PrimaryPushButton {{
    color: {ink}; background-color: {c["color"]};
    border: 1px solid {c["hover"]}; border-bottom: 1px solid {c["pressed"]};
}}
PrimaryPushButton:hover {{
    background-color: {c["hover"]}; border: 1px solid {c["hover"]};
    border-bottom: 1px solid {c["pressed"]};
}}
PrimaryPushButton:pressed {{
    color: rgba(255, 255, 255, 0.78); background-color: {c["pressed"]};
    border: 1px solid {c["pressed"]};
}}
PrimaryPushButton:disabled {{ {disabled} }}
"""


def _style(kind):
    def restyle(widget):
        setCustomStyleSheet(widget, _accent_qss(kind, False), _accent_qss(kind, True))

    return restyle


def accent(widget):
    """The accent fill with white text in both themes, also for a library dialog's confirm
    button; it follows a change of theme colour."""
    restyle = _style("button")
    restyle(widget)
    _accented[widget] = restyle
    return widget


class TextButton(PushButton):
    """Never narrower than its text (80 px at least): a crowded row gives way through its
    labels instead of cutting a button's text off."""

    def minimumSizeHint(self):
        hint = super().sizeHint()
        return QSize(max(80, hint.width()), hint.height())


class AccentButton(PrimaryPushButton):
    def minimumSizeHint(self):
        hint = super().sizeHint()
        return QSize(max(80, hint.width()), hint.height())

    def _drawIcon(self, icon, painter, rect, state=QIcon.State.Off):
        if isinstance(icon, FluentIconBase) and self.isEnabled():
            # Theme.DARK variants are the white glyphs.
            PushButton._drawIcon(self, icon.icon(Theme.DARK), painter, rect, state)
        else:
            super()._drawIcon(icon, painter, rect, state)


class AccentToggle(TransparentTogglePushButton):
    def _drawIcon(self, icon, painter, rect):
        if self.isChecked() and isinstance(icon, FluentIconBase) and self.isEnabled():
            PushButton._drawIcon(self, icon.icon(Theme.DARK), painter, rect)
        else:
            super()._drawIcon(icon, painter, rect)


class AccentRadio(RadioButton):
    """The accent ring with a white centre in both themes; the library's dark theme draws a
    black centre on a lightened accent."""

    def _drawIndicator(self, painter):
        if self.isChecked() and self.isEnabled():
            width = 4 if self.isHover and not self.isDown() else 5
            self._drawCircle(painter, self.indicatorPos, 10, width, ACCENT, QColor("white"))
        else:
            super()._drawIndicator(painter)


def toggle_button():
    widget = AccentToggle()
    restyle = _style("toggle")
    restyle(widget)
    _accented[widget] = restyle
    return widget


# Whether a cell's icon is a button (Table.set_icons) rather than a status mark.
ACTION_ROLE = Qt.ItemDataRole.UserRole + 7


class _TableDelegate(TableItemDelegate):
    """The library's row delegate; icons set with Table.set_icons are drawn centred in their
    column as buttons, with the hover and pressed background of a transparent tool button."""

    def __init__(self, parent):
        super().__init__(parent)
        self.hover_cell = None  # (row, column) of the icon button under the mouse
        self.pressed_cell = None

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        if index.data(ACTION_ROLE):
            option.features &= ~QStyleOptionViewItem.ViewItemFeature.HasDecoration
            option.icon = QIcon()

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        icon = index.data(Qt.ItemDataRole.DecorationRole) if index.data(ACTION_ROLE) else None
        if icon is None:
            return
        cell = (index.row(), index.column())
        center = QRectF(option.rect).center()
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if cell in (self.hover_cell, self.pressed_cell):
            dark = isDarkTheme()
            pressed = cell == self.pressed_cell
            alpha = (11 if pressed else 21) if dark else (6 if pressed else 12)
            c = 255 if dark else 0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(c, c, c, alpha))
            painter.drawRoundedRect(QRectF(0, 0, 32, 30).translated(center - QPointF(16, 15)), 5, 5)
        size = 15 if cell == self.pressed_cell else 16
        target = QRectF(0, 0, size, size).translated(center - QPointF(size / 2, size / 2))
        icon.paint(painter, target.toRect())
        painter.restore()

    def _drawCheckBox(self, painter, option, index):
        state = Qt.CheckState(index.data(Qt.ItemDataRole.CheckStateRole))
        if state == Qt.CheckState.Unchecked:
            return super()._drawCheckBox(painter, option, index)
        # The library's check box, with the accent fill and a white mark in both themes.
        painter.save()
        rect = QRectF(option.rect.x() + 15, option.rect.center().y() - 9.5, 19, 19)
        painter.setPen(ACCENT)
        painter.setBrush(ACCENT)
        painter.drawRoundedRect(rect, 4.5, 4.5)
        icon = (
            CheckBoxIcon.ACCEPT if state == Qt.CheckState.Checked else CheckBoxIcon.PARTIAL_ACCEPT
        )
        icon.render(painter, rect, theme=Theme.LIGHT)  # reversed: the white mark
        painter.restore()


# Field colours from the library's line-edit / spin-box style sheets:
# (background, hover, focused, disabled, border, bottom edge, disabled border).
FIELD_COLORS = {
    True: (
        QColor(255, 255, 255, 15),
        QColor(255, 255, 255, 21),
        # The library's solid #1f1f1f read as a black hole on the wallpaper-tinted cards.
        QColor(0, 0, 0, 56),
        QColor(255, 255, 255, 11),
        QColor(255, 255, 255, 20),
        QColor(255, 255, 255, 139),
        QColor(255, 255, 255, 18),
    ),
    False: (
        QColor(255, 255, 255, 179),
        QColor(249, 249, 249, 128),
        QColor(255, 255, 255),
        QColor(249, 249, 249, 77),
        QColor(0, 0, 0, 13),
        QColor(0, 0, 0, 100),
        QColor(0, 0, 0, 13),
    ),
}
FIELD_STATES = (
    "",
    ":hover",
    ":focus",
    ":disabled",
    ":read-only",
    "[transparent=true]:focus",
    "[transparent=false]:focus",
    ":focus[transparent=true]",
    "[symbolVisible=false]",
)


class _SmoothField(QObject):
    """Paints a field's rounded background and border with antialiasing before the field
    draws its text. Qt's style-sheet renderer draws a radius whose bottom edge has another
    colour with stepped corners at fractional scale factors such as 150%."""

    def eventFilter(self, field, event):
        if event.type() == QEvent.Type.Paint:
            background, hover, focused, disabled, border, bottom, disabled_border = FIELD_COLORS[
                isDarkTheme()
            ]
            if not field.isEnabled():
                fill, border, bottom = disabled, disabled_border, disabled_border
            elif field.hasFocus():
                fill = focused
            elif field.underMouse():
                fill = hover
            else:
                fill = background
            rect = QRectF(field.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
            path = QPainterPath()
            path.addRoundedRect(rect, 5, 5)
            painter = QPainter(field)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.fillPath(path, fill)
            painter.setPen(QPen(border, 1))
            painter.drawPath(path)
            # The darker (light theme) or brighter (dark theme) bottom edge curls into
            # the corners, as in the library's own look.
            painter.setClipRect(QRectF(0, rect.bottom() - 2, field.width(), 3))
            focus_line = field.hasFocus() and isinstance(field, (QTextEdit, QPlainTextEdit))
            painter.setPen(QPen(accent_line() if focus_line else bottom, 1))
            painter.drawPath(path)
            painter.end()
        elif event.type() in (QEvent.Type.Enter, QEvent.Type.Leave):
            field.update()
        return False


def smooth(field):
    """Draw a line edit, spin box or text edit's frame antialiased (see _SmoothField)."""
    name = next(
        kind.__name__
        for kind in type(field).__mro__
        if kind.__name__ in {"LineEdit", "SpinBox", "DoubleSpinBox", "TextEdit", "PlainTextEdit"}
    )
    qss = ", ".join(f"{name}{state}" for state in FIELD_STATES) + (
        " { background: transparent; border-color: transparent; }"
    )
    setCustomStyleSheet(field, qss, qss)
    if isinstance(field, QTextEdit):
        # Pasted text keeps no fonts or colours from where it came from (a web page's black
        # highlight stayed behind the characters).
        field.setAcceptRichText(False)
    field._smooth = _SmoothField(field)
    field.installEventFilter(field._smooth)
    return field


def label(text: str, role="", *, wrap=True):
    kind = {
        "title": TitleLabel,
        "section": SubtitleLabel,
        "page": SubtitleLabel,
        "card": StrongBodyLabel,
        "muted": CaptionLabel,
    }.get(role, BodyLabel)
    widget = bind(kind(str(text)), text)
    widget.setWordWrap(wrap and role not in {"page", "card"})
    if role == "muted":
        widget.setTextColor(QColor("#5b6068"), QColor("#bfc5ce"))
        setFont(widget, 13)
    elif role == "error":
        widget.setTextColor(QColor("#a4262c"), QColor("#ffb4ab"))
    return widget


def toast(anchor, message, *, kind="error", duration=None):
    """A message over the window anchor belongs to, gone after a while: the outcome of an
    action, which must not stay on the page or be cut short there."""
    window = anchor.window()
    if window is anchor or not str(message):
        return
    show = {"error": InfoBar.error, "success": InfoBar.success, "warning": InfoBar.warning}
    show[kind](
        "",
        str(message),
        duration=duration or (8000 if kind == "error" else 5000),
        position=InfoBarPosition.TOP_RIGHT,
        parent=window,
    )


class ErrorLabel(BodyLabel):
    """An error that never changes the height of the row or field it belongs to.

    The toast form (error_label()) draws no text: a message it is given appears as a toast,
    and it keeps the text (full_text) for whoever asks. It stays in its row as the stretch
    that keeps the row's buttons at their own size. The wrapping form is for state shown in detail
    cards (why a run failed, what is wrong with a recording); it pushes the content below it
    down. The inline form is one elided line with the full text as its tooltip, for a
    widget's own state (a player that cannot play)."""

    def __init__(self, parent=None, *, wrap=False, toast=False):
        # qfluentwidgets dispatches BodyLabel(text, parent) back into self.__init__(parent),
        # so a subclass must keep the (parent) signature.
        super().__init__(parent)
        self.full_text = ""
        self.wrap = wrap
        self.toast = toast
        self.setWordWrap(wrap)
        self.setTextColor(QColor("#a4262c"), QColor("#ffb4ab"))
        if not wrap:
            self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        else:
            self.hide()

    def setText(self, text):
        previous, self.full_text = self.full_text, str(text)
        if self.toast:
            # A new message shows once; the same state reported again does not repeat it,
            # and a language switch does not bring it back.
            if self.full_text and self.full_text != previous:
                forget(self)
                toast(self, self.full_text)
            return
        self._fit()

    def minimumSizeHint(self):
        # The elided form shrinks to nothing; its text must not set the row's minimum width.
        hint = super().minimumSizeHint()
        return hint if self.wrap else QSize(0, hint.height())

    def clear(self):
        forget(self)
        self.setText("")

    def _fit(self):
        if self.wrap:
            super().setText(self.full_text)
            self.setVisible(bool(self.full_text))
            return
        elided = self.fontMetrics().elidedText(
            self.full_text, Qt.TextElideMode.ElideRight, max(0, self.width())
        )
        super().setText(elided)
        self.setToolTip(self.full_text if elided != self.full_text else "")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.wrap:
            self._fit()


class ElidedLabel(CaptionLabel):
    """Muted single line that elides to the width it is given (full text in the tooltip), so a
    long path or name can never widen the layout it sits in."""

    def __init__(self, parent=None, *, mode=Qt.TextElideMode.ElideRight):
        super().__init__(parent)
        self.full_text = ""
        self.mode = mode
        self.setTextColor(QColor("#5b6068"), QColor("#bfc5ce"))
        setFont(self, 13)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)

    def emphasize(self):
        """Normal text colour instead of the muted one (for the value side of a key/value row)."""
        self.setTextColor(QColor("#1f1f1f"), QColor("#f0f0f0"))
        return self

    def strengthen(self, size=14):
        """The look of a card heading (StrongBodyLabel), for a name that heads a detail card;
        size 20 is a page title's look (SubtitleLabel)."""
        self.emphasize()
        setFont(self, size, QFont.Weight.DemiBold)
        return self

    def setText(self, text):
        self.full_text = str(text)
        self._fit()

    def minimumSizeHint(self):
        # It elides to the width it gets; its text must not set the row's minimum width.
        return QSize(0, super().minimumSizeHint().height())

    def _fit(self):
        elided = self.fontMetrics().elidedText(self.full_text, self.mode, max(0, self.width()))
        super().setText(elided)
        self.setToolTip(self.full_text if elided != self.full_text else "")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()


class RowPlayback(QObject):
    """A play/pause icon in a table column for rows that each have an audio file, played by
    the page's player. Painted by the model, so long tables need no widget per row."""

    def __init__(self, table, player, path_of, key="play"):
        super().__init__(table)
        self.table, self.player, self.path_of, self.key = table, player, path_of, key
        table.icon_clicked.connect(self._clicked)
        player.player.playbackStateChanged.connect(lambda _state: self.sync())

    def _clicked(self, row, key):
        if key == self.key:
            self.player.toggle(self.path_of(row))

    def sync(self):
        """Call after the table's rows change."""
        icons, tips = {}, {}
        for row in self.table.rows_model.rows:
            playing = self.player.playing(self.path_of(row))
            icons[row["id"]] = FluentIcon.PAUSE if playing else FluentIcon.PLAY
            tips[row["id"]] = t("widgets.pause") if playing else t("widgets.listen")
        self.table.set_icons(self.key, icons, tips)


ICON = Path(__file__).with_name("assets") / "icon.ico"


class AppIcon(FluentIconBase, Enum):
    """Glyphs drawn for this application (assets/icons), in the library's black/white pair
    form: generating speech, training (a rising curve), models (a cube) and emoji (a smile)."""

    INFERENCE = "inference"
    TRAINING = "training"
    MODELS = "models"
    EMOJI = "emoji"

    def path(self, theme=Theme.AUTO):
        return str(ICON.parent / "icons" / f"{self.value}_{getIconColor(theme)}.svg")


def app_icon() -> QIcon:
    """The application's icon: a speech bubble cut into pink and aqua, with a waveform (assets/icon.svg;
    icon-small.svg for 32 px and below; scripts/build_icon.py renders icon.ico)."""
    return QIcon(str(ICON))


def elided_label(
    text="", *, emphasis=False, strong=False, heading=False, mode=Qt.TextElideMode.ElideRight
):
    widget = bind(ElidedLabel(mode=mode), text)
    if heading:
        return widget.strengthen(20)
    if strong:
        return widget.strengthen()
    return widget.emphasize() if emphasis else widget


def open_folder(path):
    """Open a folder (or the folder holding a file) in the system file manager."""
    path = Path(path)
    while not path.exists() and path.parent != path:
        path = path.parent
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path if path.is_dir() else path.parent)))


class PathField(QWidget):
    """Read-only path: elided in the middle so both the drive and the last folder stay
    visible, full path in the tooltip, with open-folder and optional change buttons."""

    def __init__(self, path="", *, change=None):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        self.value = elided_label(emphasis=True, mode=Qt.TextElideMode.ElideMiddle)
        layout.addWidget(self.value, 1)
        layout.addWidget(
            icon_button(
                FluentIcon.FOLDER, t("common.open_folder"), lambda: open_folder(self.text())
            )
        )
        if change is not None:
            layout.addWidget(icon_button(FluentIcon.EDIT, t("common.change_folder"), change))
        self.setText(path)

    def text(self):
        return self.value.full_text

    def setText(self, path):
        self.value.setText(str(path))


class KeyValues(QWidget):
    """Aligned key / value lines; values elide and keep the full text as a tooltip."""

    def __init__(self, titles, *, key_width=96, paths=()):
        super().__init__()
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(8)
        layout.setColumnStretch(1, 1)
        self.values = {}
        self.captions = {}
        for position, (key, title) in enumerate(titles):
            caption = label(title, "muted", wrap=False)
            caption.setMinimumWidth(key_width)
            # Paths elide in the middle so the drive and the file name both stay visible.
            mode = Qt.TextElideMode.ElideMiddle if key in paths else Qt.TextElideMode.ElideRight
            value = elided_label(emphasis=True, mode=mode)
            layout.addWidget(caption, position, 0)
            layout.addWidget(value, position, 1)
            self.values[key] = value
            self.captions[key] = caption

    def set(self, key, value):
        set_text(self.values[key], value)

    def as_text(self):
        return "\n".join(
            f"{self.captions[key].text()}: {value.full_text}" for key, value in self.values.items()
        )


def error_label(*, wrap=False):
    """Where an action reports failure: a toast, or with wrap=True state shown in a card."""
    return ErrorLabel(wrap=wrap, toast=not wrap)


def button(text, callback, *, primary=False, icon=None):
    kind = AccentButton if primary else TextButton
    widget = bind(kind(str(text)), text)
    if primary:
        accent(widget)
    if icon is not None:
        widget.setIcon(icon)
    widget.setMinimumHeight(32)
    widget.clicked.connect(callback)
    return widget


def ghost_button(text, callback, *, icon=None):
    """Secondary, borderless text button for actions that sit beside a primary one."""
    widget = bind(GhostButton(str(text)), text)
    if icon is not None:
        widget.setIcon(icon)
    widget.setMinimumHeight(30)
    widget.clicked.connect(callback)
    return widget


def icon_button(icon, tooltip, callback, *, size=30):
    """Row-level action: icon only, named by its tooltip (also its accessible name)."""
    widget = TransparentToolButton(icon)
    widget.setFixedSize(size, size)
    bind(widget, tooltip, "setToolTip")
    bind(widget, tooltip, "setAccessibleName")
    widget.installEventFilter(ToolTipFilter(widget, 300))
    widget.clicked.connect(callback)
    return widget


def search_box(placeholder):
    widget = smooth(SearchLineEdit())
    bind(widget, placeholder, "setPlaceholderText")
    bind(widget, placeholder, "setAccessibleName")
    widget.setClearButtonEnabled(True)
    widget.setMinimumWidth(220)
    widget.setMaximumWidth(260)
    return widget


def card(parent_layout=None, *, stretch=0):
    widget = SimpleCardWidget()
    widget.setBorderRadius(8)
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(16, 14, 16, 14)
    layout.setSpacing(10)
    layout.setAlignment(Qt.AlignmentFlag.AlignTop)
    if parent_layout is not None:
        parent_layout.addWidget(widget, stretch)
    return widget, layout


def row(*items, spacing=8):
    """A single horizontal line. None adds a stretch, (widget, n) gives a widget stretch n,
    anything else keeps its own size."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(spacing)
    for item in items:
        if item is None:
            layout.addStretch(1)
        elif isinstance(item, tuple):
            layout.addWidget(*item)
        else:
            layout.addWidget(item)
    return widget


class _InWindow(QObject):
    """Shows a library dialog as a layer inside its window instead of as a window of its own.

    A window of its own takes the activation from the main window, and Windows then swaps the
    main window's Mica backdrop for the flat inactive colour: the whole window flashed when a
    dialog opened and again when it closed. As a child the dialog cannot be modal, so while it
    is shown this keeps shortcuts, keys and focus from reaching the page under it."""

    def __init__(self, dialog):
        super().__init__(dialog)
        self.dialog = dialog
        parent = dialog.parentWidget()
        dialog.setWindowFlags(Qt.WindowType.Widget)
        dialog.setGeometry(0, 0, parent.width(), parent.height())
        # The library watched dialog.window() for resizes, which was the dialog itself.
        parent.installEventFilter(dialog)
        dialog.installEventFilter(self)

    def outside(self, widget):
        """Whether widget is in the dialog's window but not in the dialog."""
        return (
            widget.window() is self.dialog.window()
            and widget is not self.dialog
            and not self.dialog.isAncestorOf(widget)
        )

    def eventFilter(self, obj, event):
        kind = event.type()
        if obj is self.dialog:
            if kind == QEvent.Type.Show:
                self.dialog.raise_()
                QApplication.instance().installEventFilter(self)
            elif kind == QEvent.Type.Hide:
                QApplication.instance().removeEventFilter(self)
            return False
        if kind == QEvent.Type.Shortcut:
            # Delivered to the QShortcut or QAction; its owner decides where it belongs.
            owner = obj.parent()
            while owner is not None and not isinstance(owner, QWidget):
                owner = owner.parent()
            return owner is not None and self.outside(owner)
        if not isinstance(obj, QWidget):
            return False
        if kind in (QEvent.Type.ShortcutOverride, QEvent.Type.KeyPress):
            return self.outside(obj)
        if kind == QEvent.Type.FocusIn and self.outside(obj):
            if obj.window() is self.dialog.window():
                QTimer.singleShot(0, self.dialog.widget.setFocus)
        return False


class _Layer:
    """A library dialog shown inside its window (_InWindow), dimming the page under it.

    No fade in or out: the library faded the whole layer through an opacity effect and dropped
    it at the end, and removed the box's shadow at once on closing; a fade of the dimming
    alone still read as a flicker. The layer appears and goes in one step."""

    def _init_layer(self):
        self._in_window = _InWindow(self)
        # The library's mask for message boxes: black at 30 %, in both themes.
        self.setMaskColor(QColor(0, 0, 0, 76))

    def showEvent(self, event):
        QDialog.showEvent(self, event)

    def done(self, code):
        QDialog.done(self, code)


class Confirmation(_Layer, MessageBox):
    """A message box whose text wraps at a readable width. The library wraps it to the window's
    width, so a long message made the box span the window."""

    WRAP = 64  # width units: a Chinese character counts two

    def __init__(self, title, content, parent):
        super().__init__(title, content, parent)
        self._init_layer()

    def _adjustText(self):
        self.contentLabel.setText(TextWrap.wrap(self.content, self.WRAP, False)[0])


class GhostButton(TransparentPushButton):
    def minimumSizeHint(self):
        return super().sizeHint()


class Dialog(_Layer, MessageBoxBase):
    """A message box whose confirm button keeps the accent and white text in the dark theme."""

    def __init__(self, parent):
        super().__init__(parent)
        self._init_layer()
        accent(self.yesButton)


class TextPrompt(Dialog):
    def __init__(self, parent, title, caption, value):
        super().__init__(parent)
        self.viewLayout.addWidget(translated_widget(SubtitleLabel, title, self))
        self.viewLayout.addWidget(translated_widget(BodyLabel, caption, self))
        self.input = smooth(LineEdit(self))
        set_text(self.input, value)
        self.viewLayout.addWidget(self.input)
        set_text(self.yesButton, t("widgets.save"))
        set_text(self.cancelButton, t("app.cancel"))
        self.yesButton.setDefault(False)
        self.cancelButton.setDefault(True)
        self.input.selectAll()
        self.widget.setMinimumWidth(380)


def prompt_text(parent, title, caption, value=""):
    dialog = TextPrompt(parent.window(), title, caption, value)
    accepted = bool(dialog.exec())
    return dialog.input.text(), accepted


class ChoicePrompt(Dialog):
    def __init__(self, parent, title, caption, items):
        super().__init__(parent)
        self.viewLayout.addWidget(translated_widget(SubtitleLabel, title, self))
        text = translated_widget(BodyLabel, caption, self)
        text.setWordWrap(True)
        self.viewLayout.addWidget(text)
        self.choice = ChoiceBox()
        for label_text, value in items:
            self.choice.addItem(str(label_text), userData=value)
        self.choice.setCurrentIndex(0)
        self.viewLayout.addWidget(self.choice)
        set_text(self.yesButton, t("widgets.choose"))
        set_text(self.cancelButton, t("app.cancel"))
        self.widget.setMinimumWidth(380)


def prompt_choice(parent, title, caption, items):
    """One of items ((label, value) pairs) chosen in a dialog: its value, or None if cancelled."""
    dialog = ChoicePrompt(parent.window(), title, caption, items)
    return dialog.choice.value() if dialog.exec() else None


def files_start(settings: dict) -> Path:
    """Where export and import pickers open: the folder last used, else the desktop (not the
    working directory, which is the application's own folder)."""
    last = settings.get("files_folder")
    if last and Path(last).is_dir():
        return Path(last)
    return Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation))


def save_file(parent, settings: dict, title, name, filters) -> str:
    path, _ = QFileDialog.getSaveFileName(
        parent, str(title), str(files_start(settings) / str(name)), str(filters)
    )
    if path:
        settings["files_folder"] = str(Path(path).parent)
    return path


def open_file(parent, settings: dict, title, filters, *, many=False):
    """One path ("" when cancelled), or a list of them with many=True."""
    pick = QFileDialog.getOpenFileNames if many else QFileDialog.getOpenFileName
    chosen, _ = pick(parent, str(title), str(files_start(settings)), str(filters))
    if chosen:
        settings["files_folder"] = str(Path(chosen[0] if many else chosen).parent)
    return chosen


def choose_folder(parent, settings: dict, title) -> str:
    path = QFileDialog.getExistingDirectory(parent, str(title), str(files_start(settings)))
    if path:
        settings["files_folder"] = str(Path(path))
    return path


def field(title, widget, *, buddy=None):
    """A caption above its control, for compact parameter grids."""
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(4)
    caption = translated_widget(CaptionLabel, title, box)
    caption.setTextColor(QColor("#323a45"), QColor("#d6dbe3"))
    caption.setBuddy(buddy or widget)
    bind(buddy or widget, title, "setAccessibleName")
    layout.addWidget(caption)
    layout.addWidget(widget)
    return box


def grid(*fields, columns=2):
    box = QWidget()
    layout = QGridLayout(box)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setHorizontalSpacing(12)
    layout.setVerticalSpacing(10)
    for position, item in enumerate(fields):
        layout.addWidget(item, position // columns, position % columns)
    for column in range(columns):
        layout.setColumnStretch(column, 1)
    return box


def switch():
    """A bare on/off indicator; its caption is a separate label so it can wrap and translate."""
    widget = SwitchButton()
    widget.setOnText("")
    widget.setOffText("")
    indicator = widget.indicator
    indicator.setCheckedColor(ACCENT, ACCENT)
    _accented[indicator] = lambda target: target.setCheckedColor(ACCENT, ACCENT)
    library_knob = indicator._sliderColor
    indicator._sliderColor = lambda: (
        QColor(ON_ACCENT) if indicator.isChecked() and indicator.isEnabled() else library_knob()
    )
    return widget


def minimum_height(layout, width):
    """The least height layout needs at width: wrapped text at its height for its width,
    everything else at its minimum. Qt's own height-for-width of a box layout is the
    preferred height, which would make a page scroll while its lists could still shrink."""
    margins = layout.contentsMargins()
    inner = width - margins.left() - margins.right()
    every = [layout.itemAt(i) for i in range(layout.count())]
    items = [item for item in every if not item.isEmpty()]
    # Fixed spacing (addSpacing) is an empty item that still takes its height.
    spacers = sum(item.minimumSize().height() for item in every if item.spacerItem())

    def least(item, width):
        widget = item.widget()
        if widget is None:
            nested = item.layout()
            return minimum_height(nested, width) if nested else item.minimumSize().height()
        if not widget.hasHeightForWidth() or isinstance(widget, QAbstractScrollArea):
            return item.minimumSize().height()
        if widget.layout() is not None:
            return max(widget.minimumHeight(), minimum_height(widget.layout(), width))
        return widget.heightForWidth(width)

    def across(item):
        return item.geometry().width() or inner

    if isinstance(layout, QStackedLayout):
        current = layout.currentWidget()
        total = least(layout.itemAt(layout.currentIndex()), inner) if current else 0
    elif isinstance(layout, QGridLayout):
        rows = {}
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if not item.isEmpty():
                row_index = layout.getItemPosition(index)[0]
                rows[row_index] = max(rows.get(row_index, 0), least(item, across(item)))
        total = sum(rows.values()) + max(0, layout.verticalSpacing()) * max(0, len(rows) - 1)
    elif isinstance(layout, QBoxLayout) and layout.direction() in (
        QBoxLayout.Direction.LeftToRight,
        QBoxLayout.Direction.RightToLeft,
    ):
        total = max((least(item, across(item)) for item in items), default=0)
    elif isinstance(layout, QBoxLayout):
        total = sum(least(item, inner) for item in items) + spacers
        total += max(0, layout.spacing()) * max(0, len(items) - 1)
    else:
        total = layout.minimumSize().height() - margins.top() - margins.bottom()
    return total + margins.top() + margins.bottom()


class _PageContent(QWidget):
    def heightForWidth(self, width):
        # The scroll area asks this of content holding wrapped text: the least it needs, so
        # the page scrolls only once its lists and inner areas cannot give up more height.
        return minimum_height(self.layout(), width)


class Page(ScrollArea):
    """Scroll shell. Work pages fit the window from 1280x688 up and scroll only below that;
    pages that read top to bottom (settings, getting started) may scroll at any size. None
    scrolls sideways (scripts/capture_pages checks all of this)."""

    def __init__(self, title=None):
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.Shape.NoFrame)
        self.content = _PageContent()
        self.body = QVBoxLayout(self.content)
        self.body.setContentsMargins(24, 20, 24, 20)
        self.body.setSpacing(14)
        if title:
            self.body.addWidget(label(title, "page", wrap=False))
        self.setWidget(self.content)
        self.content.setAutoFillBackground(False)
        self.enableTransparentBackground()


TONES = {
    "ok": ("#107c10", "#6ccb5f"),
    "warn": ("#9d5d00", "#f5d58a"),
    "bad": ("#a4262c", "#ff99a4"),
    "accent": None,  # accent_line(): follows the chosen theme colour
    "muted": ("#5b6068", "#9a9a9a"),
}


def tone_color(tone):
    if tone == "accent":
        return accent_line()
    light, dark = TONES[tone]
    return QColor(dark if isDarkTheme() else light)


class Rows(QAbstractTableModel):
    toggled = Signal(str, bool)

    def __init__(self, columns, parent=None, checks=None):
        """checks maps a column key to the row field its checkbox shows and edits."""
        super().__init__(parent)
        self.columns = columns
        self.checks = dict(checks or {})
        self.rows = []
        locale_manager().changed.connect(self.retranslate)

    def retranslate(self, _language):
        self.headerDataChanged.emit(Qt.Orientation.Horizontal, 0, len(self.columns) - 1)
        if self.rows:
            self.dataChanged.emit(
                self.index(0, 0), self.index(len(self.rows) - 1, len(self.columns) - 1)
            )

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self.rows[index.row()]
        key = self.columns[index.column()][0]
        if key in self.checks and role == Qt.ItemDataRole.CheckStateRole:
            state = Qt.CheckState.Checked if row.get(self.checks[key]) else Qt.CheckState.Unchecked
            return state
        if role == Qt.ItemDataRole.DisplayRole:
            return str(row.get(key, ""))
        if role == Qt.ItemDataRole.ToolTipRole:
            if key in row.get("icons", {}):
                return str(row.get("icon_tips", {}).get(key, ""))
            return str(row.get("tooltip") or row.get(key, ""))
        if role == Qt.ItemDataRole.ForegroundRole and key in row.get("tones", {}):
            return QBrush(tone_color(row["tones"][key]))
        if role == Qt.ItemDataRole.DecorationRole and key in row.get("icons", {}):
            return row["icons"][key].icon()
        if role == ACTION_ROLE:
            return key in row.get("icons", {})
        if role == Qt.ItemDataRole.DecorationRole and index.column() == 0 and row.get("icon"):
            return row["icon"].icon()
        return None

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and self.columns[index.column()][0] in self.checks:
            flags |= Qt.ItemFlag.ItemIsUserCheckable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        key = self.columns[index.column()][0] if index.isValid() else None
        if role != Qt.ItemDataRole.CheckStateRole or key not in self.checks:
            return False
        checked = int(getattr(value, "value", value)) == Qt.CheckState.Checked.value
        row = self.rows[index.row()]
        row[self.checks[key]] = checked
        self.dataChanged.emit(index, index)
        self.toggled.emit(row["id"], checked)
        return True

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.DisplayRole:
            return str(self.columns[section][1])
        if orientation == Qt.Orientation.Horizontal and role == Qt.ItemDataRole.TextAlignmentRole:
            # Titles start where the cells' text starts, not centered over it.
            return int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        return None

    def update(self, rows):
        wanted = {row["id"] for row in rows}
        for index in range(len(self.rows) - 1, -1, -1):
            if self.rows[index]["id"] not in wanted:
                self.beginRemoveRows(QModelIndex(), index, index)
                self.rows.pop(index)
                self.endRemoveRows()
        for index, value in enumerate(rows):
            current = (
                index
                if index < len(self.rows) and self.rows[index]["id"] == value["id"]
                else next((i for i, row in enumerate(self.rows) if row["id"] == value["id"]), None)
            )
            if current is None:
                self.beginInsertRows(QModelIndex(), index, index)
                self.rows.insert(index, value)
                self.endInsertRows()
            elif current != index:
                self.beginMoveRows(
                    QModelIndex(),
                    current,
                    current,
                    QModelIndex(),
                    index if index < current else index + 1,
                )
                self.rows.insert(index, self.rows.pop(current))
                self.endMoveRows()
            if self.rows[index] != value:
                self.rows[index] = value
                self.dataChanged.emit(
                    self.index(index, 0), self.index(index, len(self.columns) - 1)
                )


class Table(TableView):
    selected = Signal(dict)
    # (row, column key) when the icon a row shows in that column is clicked.
    icon_clicked = Signal(dict, str)

    def __init__(
        self,
        columns,
        *,
        height=210,
        fill=False,
        header=True,
        stretch=None,
        checks=None,
        main_minimum=160,
        optional=(),
    ):
        """fill: take whatever height the layout gives and scroll inside it.
        checks: {column key: row field} for a checkbox in front of that column's text.
        stretch: index of the column that absorbs spare width (default: the last one).
        main_minimum: the narrowest that column may get; when the table is narrower than all
        preferred widths, the other columns give up their spare width first.
        optional: keys of columns to hide, in this order, while the main column would still
        get less than its minimum even with the other columns at their narrowest."""
        super().__init__()
        self.setItemDelegate(_TableDelegate(self))
        self.rows_model = Rows(columns, self, checks)
        self.setModel(self.rows_model)
        self.setAlternatingRowColors(True)
        self.setShowGrid(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setWordWrap(False)
        self.verticalHeader().hide()
        self.verticalHeader().setDefaultSectionSize(40)
        self.verticalHeader().setMinimumSectionSize(1)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        if stretch is None:
            self.horizontalHeader().setStretchLastSection(True)
        else:
            self.horizontalHeader().setStretchLastSection(False)
            self.horizontalHeader().setSectionResizeMode(stretch, QHeaderView.ResizeMode.Stretch)
        if not header:
            self.horizontalHeader().hide()
        # The cells' text is indented 16px (the library's item padding); the titles match it.
        qss = "QHeaderView::section:horizontal { padding-left: 16px; }"
        setCustomStyleSheet(self, qss, qss)
        self.main_column = len(columns) - 1 if stretch is None else stretch
        self.main_minimum = main_minimum
        keys = [column[0] for column in columns]
        self.optional = [keys.index(key) for key in optional]
        for index, column in enumerate(columns):
            if index != stretch:
                self.setColumnWidth(index, column[2])
        self.preferred_height = height
        self.manage_empty_height = not fill
        self.row_actions = None
        self.actions_for = "selected"
        if fill:
            self.setMinimumHeight(88)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        else:
            self.setFixedHeight(96)
        self.empty_label = translated_widget(
            BodyLabel, t("widgets.no_records_yet"), self.viewport()
        )
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setTextColor(QColor("#5b6068"), QColor("#bfc5ce"))
        self.height_animation = QVariantAnimation(self)
        self.height_animation.setDuration(220)
        self.height_animation.valueChanged.connect(lambda value: self.setFixedHeight(int(value)))
        self.requested_rows = []
        self.requested_ids = set()
        self.row_heights = {}
        self.row_animation = QVariantAnimation(self)
        self.row_animation.setDuration(220)
        self.row_animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.row_animation.valueChanged.connect(self._animate_rows)
        self.row_animation.finished.connect(self._finish_rows)
        self.selectionModel().currentRowChanged.connect(self._selected)
        self.clicked.connect(self._clicked)
        bind(self, t("widgets.selectable_records"), "setAccessibleName")

    def _action_cell(self, pos):
        index = self.indexAt(pos)
        return (index.row(), index.column()) if index.data(ACTION_ROLE) else None

    def _set_cells(self, hover, pressed):
        delegate = self.itemDelegate()
        old = {delegate.hover_cell, delegate.pressed_cell}
        delegate.hover_cell, delegate.pressed_cell = hover, pressed
        for cell in (old | {hover, pressed}) - {None}:
            self.update(self.rows_model.index(*cell))
        self.viewport().setCursor(
            Qt.CursorShape.PointingHandCursor if hover else Qt.CursorShape.ArrowCursor
        )

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        cell = self._action_cell(event.position().toPoint())
        delegate = self.itemDelegate()
        if cell != delegate.hover_cell:
            pressed = delegate.pressed_cell if delegate.pressed_cell == cell else None
            self._set_cells(cell, pressed)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        cell = self._action_cell(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and cell:
            self._set_cells(cell, cell)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self._set_cells(self._action_cell(event.position().toPoint()), None)

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._set_cells(None, None)

    def _clicked(self, index):
        if not index.isValid() or index.row() >= len(self.rows_model.rows):
            return
        row = self.rows_model.rows[index.row()]
        key = self.rows_model.columns[index.column()][0]
        if key in row.get("icons", {}) and row["id"] in self.requested_ids:
            self.icon_clicked.emit(row, key)

    def set_icons(self, key, icons, tips=None):
        """Show icons[row id] in column key of each row (none for a missing id)."""
        column = next(i for i, c in enumerate(self.rows_model.columns) if c[0] == key)
        for position, row in enumerate(self.rows_model.rows):
            icon = icons.get(row["id"])
            if row.get("icons", {}).get(key) is icon:
                continue
            row.setdefault("icons", {}).pop(key, None)
            if icon is not None:
                row["icons"][key] = icon
                row.setdefault("icon_tips", {})[key] = (tips or {}).get(row["id"], "")
            index = self.rows_model.index(position, column)
            self.rows_model.dataChanged.emit(index, index)

    def set_row_actions(self, factory, *, for_rows="selected"):
        """Icon buttons in the last column; factory(row) returns the widget for one row.
        Widgets are created for the current row only, or for every row with for_rows="all"."""
        self.row_actions = factory
        self.actions_for = for_rows
        self._place_actions()

    def _place_actions(self):
        if not self.row_actions or not self.rows_model.columns:
            return
        last = self.rows_model.columnCount() - 1
        current = self.currentIndex().row()
        for position, row in enumerate(self.rows_model.rows):
            index = self.rows_model.index(position, last)
            wanted = row["id"] in self.requested_ids and (
                self.actions_for == "all" or position == current
            )
            existing = self.indexWidget(index)
            if not wanted:
                if existing is not None:
                    self.setIndexWidget(index, None)
            elif existing is None or existing.property("row_id") != row["id"]:
                widget = self.row_actions(row)
                widget.setProperty("row_id", row["id"])
                self.setIndexWidget(index, widget)

    def _selected(self, index, old):
        self._place_actions()
        if (
            index.isValid()
            and index.row() < len(self.rows_model.rows)
            and self.rows_model.rows[index.row()]["id"] in self.requested_ids
        ):
            self.selected.emit(self.rows_model.rows[index.row()])

    def item(self):
        index = self.currentIndex()
        return (
            self.rows_model.rows[index.row()]
            if index.isValid()
            and index.row() < len(self.rows_model.rows)
            and self.rows_model.rows[index.row()]["id"] in self.requested_ids
            else None
        )

    def update_rows(self, rows):
        self.empty_label.setVisible(not rows)
        if self.manage_empty_height:
            target = self.preferred_height if rows else 96
            if target != self.height_animation.endValue():
                self.height_animation.stop()
                if QApplication.instance().property("reduce_motion"):
                    self.setFixedHeight(target)
                else:
                    self.height_animation.setStartValue(self.height())
                    self.height_animation.setEndValue(target)
                    self.height_animation.start()
        selected_id = self.item()["id"] if self.item() else None
        anchor = self.rowAt(0)
        anchor_id = self.rows_model.rows[anchor]["id"] if anchor >= 0 else None
        offset = self.verticalScrollBar().value()
        self.row_animation.stop()
        old_rows = list(self.rows_model.rows)
        old_heights = {row["id"]: self.rowHeight(index) for index, row in enumerate(old_rows)}
        self.requested_rows = list(rows)
        self.requested_ids = {row["id"] for row in rows}
        changed_ids = self.requested_ids.symmetric_difference(old_heights)
        animated = (
            bool(changed_ids) or any(height != 40 for height in old_heights.values())
        ) and not QApplication.instance().property("reduce_motion")
        working = list(rows)
        if animated:
            for index, row in enumerate(old_rows):
                if row["id"] not in self.requested_ids:
                    working.insert(min(index, len(working)), row)
        self.rows_model.update(working)
        if animated:
            self.row_heights = {
                row["id"]: (
                    old_heights.get(row["id"], 1),
                    40 if row["id"] in self.requested_ids else 1,
                )
                for row in working
                if old_heights.get(row["id"], 1) != (40 if row["id"] in self.requested_ids else 1)
            }
            self.row_animation.setStartValue(0.0)
            self.row_animation.setEndValue(1.0)
            self.row_animation.start()
        else:
            self._finish_rows()
        for index, row in enumerate(working):
            if row["id"] == selected_id:
                self.setCurrentIndex(self.rows_model.index(index, 0))
            if row["id"] == anchor_id:
                self.scrollTo(
                    self.rows_model.index(index, 0), QAbstractItemView.ScrollHint.PositionAtTop
                )
        if anchor_id is None:
            self.verticalScrollBar().setValue(offset)

    def _animate_rows(self, amount):
        for index, row in enumerate(self.rows_model.rows):
            if row["id"] in self.row_heights:
                start, end = self.row_heights[row["id"]]
                self.setRowHeight(index, max(1, round(start + (end - start) * amount)))

    def _finish_rows(self):
        self.rows_model.update(self.requested_rows)
        for index in range(len(self.requested_rows)):
            self.setRowHeight(index, 40)
        self.row_heights = {}
        self._place_actions()
        self.fit_columns()

    def fit_columns(self):
        """Every column at least its preferred width, wider (up to half as much again) when
        its text needs it and there is room, so a short value is not cut off. When
        the table is narrower, the other columns give that back and then shrink towards what
        their header and visible text need, so the main column keeps its minimum. Text that
        is elided even at the preferred width (a name) may shrink to its header; an icon
        column (no header) keeps its width."""
        columns = self.rows_model.columns
        room = self.viewport().width() - self.main_minimum
        metrics = self.horizontalHeader().fontMetrics()
        shown = not self.horizontalHeader().isHidden()
        text = {i: self.sizeHintForColumn(i) for i in range(len(columns))}

        def floor(i):
            if not columns[i][1]:
                return columns[i][2]
            header = metrics.horizontalAdvance(str(columns[i][1])) + 32 if shown else 0
            if text[i] > columns[i][2]:
                return min(columns[i][2], max(header, 72))
            return min(columns[i][2], max(header, text[i]))

        def wanted(i):
            # A short value that only just misses (a duration) gets its width; a name far
            # wider than the column elides there anyway and leaves the room to the main column.
            if columns[i][1] and columns[i][2] < text[i] <= columns[i][2] * 3 // 2:
                return text[i]
            return columns[i][2]

        hidden = []
        for index in [None, *self.optional]:
            if index is not None:
                hidden.append(index)
            others = [i for i in range(len(columns)) if i != self.main_column and i not in hidden]
            floors = {i: floor(i) for i in others}
            if room >= sum(floors.values()):
                break
        for index in range(len(columns)):
            if self.isColumnHidden(index) != (index in hidden):
                self.setColumnHidden(index, index in hidden)
        preferred = {i: wanted(i) for i in others}
        if room >= sum(preferred.values()):
            widths = preferred
        else:
            spare = sum(preferred[i] - floors[i] for i in others)
            share = min(1.0, (sum(preferred.values()) - room) / spare) if spare else 0.0
            widths = {i: round(preferred[i] - (preferred[i] - floors[i]) * share) for i in others}
        for index, width in widths.items():
            if self.columnWidth(index) != width:
                self.setColumnWidth(index, width)

    def squeezed(self):
        """The main column's title when the table is too narrow to give it its minimum."""
        if self.columnWidth(self.main_column) < self.main_minimum - 1:
            return str(self.rows_model.columns[self.main_column][1]) or self.main_column
        return None

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "empty_label"):
            self.empty_label.setGeometry(self.viewport().rect())
            self.fit_columns()


class _Clip(QWidget):
    """Reveals a fixed-size child. Animating this viewport instead of the content keeps
    the content laid out at its final height, so text never reflows mid-animation."""

    def __init__(self, content):
        super().__init__()
        self.content = content
        content.setParent(self)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def content_height(self, width=None):
        width = self.width() if width is None else width
        if self.content.hasHeightForWidth() and width > 0:
            return self.content.heightForWidth(width)
        return self.content.sizeHint().height()

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return min(self.content_height(width), self.maximumHeight())

    def sizeHint(self):
        return QSize(
            self.content.sizeHint().width(), min(self.content_height(), self.maximumHeight())
        )

    def minimumSizeHint(self):
        return QSize(self.content.minimumSizeHint().width(), 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.content.setGeometry(0, 0, self.width(), self.content_height())


class _ExpandHeader(TransparentPushButton):
    """The header that opens and closes an Expand. It keeps checked state like a toggle but
    looks the same open or closed (only its chevron turns): a toggle button fills with the
    accent when checked, so every click swapped its whole look and hover state, and the
    header twitched, most of all when the pointer left it right after the click."""

    toggled = Signal(bool)

    def __init__(self):
        super().__init__()
        self.opened = False
        self.clicked.connect(lambda: self.setChecked(not self.opened, emit=True))

    def isChecked(self):
        return self.opened

    def setChecked(self, opened, *, emit=False):
        changed = opened != self.opened
        self.opened = opened
        self.setIcon(FluentIcon.CHEVRON_DOWN_MED if opened else FluentIcon.CHEVRON_RIGHT_MED)
        if changed and emit and not self.signalsBlocked():
            self.toggled.emit(opened)


class Expand(QWidget):
    def __init__(self, title, content, reduced=lambda: False):
        super().__init__()
        self.reduced = reduced
        self.toggle = _ExpandHeader()
        set_text(self.toggle, title)
        self.toggle.setChecked(False)
        self.content = content
        self.clip = _Clip(content)
        self.clip.setMaximumHeight(0)
        self.clip.setVisible(False)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self.clip)
        self.animation = QPropertyAnimation(self.clip, b"maximumHeight", self)
        self.animation.setEasingCurve(QEasingCurve.Type.InOutCubic)
        self.animation.finished.connect(self._settle)
        self.toggle.toggled.connect(self.set_open)
        self.locale_timer = QTimer(self)
        self.locale_timer.setSingleShot(True)
        self.locale_timer.timeout.connect(self._fit_translation)
        locale_manager().changed.connect(self._queue_translation_fit)

    def _target(self):
        return self.clip.content_height(self.width() or None)

    def _settle(self):
        opened = self.toggle.isChecked()
        self.clip.setVisible(opened)
        # Once open, follow the content (window width or wording may change its height).
        self.clip.setMaximumHeight(QWIDGETSIZE_MAX if opened else 0)

    def _queue_translation_fit(self, _language):
        self.locale_timer.start(0)

    def _fit_translation(self):
        if self.toggle.isChecked() and self.animation.state() == self.animation.State.Running:
            self.animation.setEndValue(self._target())
        self.clip.resizeEvent(QResizeEvent(self.clip.size(), self.clip.size()))
        self.clip.updateGeometry()

    def set_open(self, opened):
        self.animation.stop()
        current = self.clip.height() if self.clip.isVisible() else 0
        self.clip.setMaximumHeight(current)
        self.clip.setVisible(True)
        self.toggle.setChecked(opened)
        target = self._target() if opened else 0
        if self.reduced():
            self._settle()
        else:
            self.animation.setStartValue(current)
            self.animation.setEndValue(target)
            self.animation.setDuration(220)
            self.animation.start()


_peaks: dict = {}
_peaks_lock = threading.Lock()


def audio_peaks(path, signature, buckets=480):
    """The loudest sample of each slice of the recording, 0..1, for its waveform; None when it
    cannot be read. Remembered per file version."""
    if signature in _peaks:
        return _peaks[signature]
    import numpy as np
    import soundfile as sf

    try:
        data, _rate = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        return None
    level = np.abs(data).max(axis=1)
    if not len(level):
        return None
    edges = np.linspace(0, len(level), min(buckets, len(level)) + 1).astype(int)[:-1]
    peaks = np.maximum.reduceat(level, edges)
    loudest = float(peaks.max())
    # The square root keeps quiet passages visible next to loud ones.
    peaks = np.sqrt(peaks / loudest) if loudest > 0 else peaks
    with _peaks_lock:
        if len(_peaks) >= 128:
            _peaks.pop(next(iter(_peaks)))
        _peaks[signature] = peaks
    return peaks


class Waveform(QWidget):
    """The recording's shape with the played part in the accent; click, drag or arrow keys
    seek."""

    seek = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.peaks = None
        self.position = 0.0
        self.setMinimumHeight(30)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_peaks(self, peaks):
        self.peaks = peaks
        self.update()

    def set_position(self, fraction):
        self.position = min(1.0, max(0.0, fraction))
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        dark = isDarkTheme()
        rest = QColor(255, 255, 255, 70) if dark else QColor(0, 0, 0, 60)
        played = accent_line(dark) if self.isEnabled() else rest
        width, height = self.width(), self.height()
        if self.peaks is None or not len(self.peaks):
            painter.fillRect(QRectF(0, height / 2 - 1, width, 2), rest)
            return
        step = 3
        count = max(1, width // step)
        bounds = [round(i * len(self.peaks) / count) for i in range(count + 1)]
        for index in range(count):
            start, end = bounds[index], max(bounds[index + 1], bounds[index] + 1)
            value = float(self.peaks[start:end].max()) if start < len(self.peaks) else 0.0
            bar = max(2.0, value * (height - 4))
            painter.setBrush(played if (index + 0.5) / count <= self.position else rest)
            painter.drawRoundedRect(QRectF(index * step, (height - bar) / 2, 2, bar), 1, 1)

    def _seek_to(self, x):
        if self.isEnabled() and self.width() > 0:
            self.seek.emit(min(1.0, max(0.0, x / self.width())))

    def mousePressEvent(self, event):
        self._seek_to(event.position().x())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._seek_to(event.position().x())

    def keyPressEvent(self, event):
        steps = {Qt.Key.Key_Left: -0.05, Qt.Key.Key_Right: 0.05}
        if event.key() in steps and self.isEnabled():
            self.seek.emit(min(1.0, max(0.0, self.position + steps[event.key()])))
        else:
            super().keyPressEvent(event)


class Player(StandardMediaPlayBar):
    peaks_read = Signal(object, object)
    stopped = Signal()

    def __init__(self):
        super().__init__()
        self.path = None
        self.source_signature = None
        self.setEnabled(False)
        self._compact()
        bind(self, t("widgets.audio_player"), "setAccessibleName")
        bind(self.playButton, t("widgets.play_or_pause"), "setAccessibleName")
        bind(self.waveform, t("widgets.audio_position"), "setAccessibleName")
        bind(self.volumeButton, t("widgets.volume"), "setAccessibleName")
        self.player.durationChanged.connect(self._duration_changed)
        self.player.errorOccurred.connect(self._failed)
        # The library starts at 30 %, which is barely audible; the system volume governs.
        self.setVolume(100)
        self.waveform.seek.connect(
            lambda fraction: self.player.setPosition(int(fraction * self.player.duration()))
        )
        self.peaks_read.connect(self.show_peaks)

    def _compact(self):
        """One row: play, elapsed, scrubber, remaining, volume (the stock bar is two rows)."""
        layout = self.vBoxLayout
        while layout.count():
            layout.takeAt(0)
        for hidden in (
            self.leftButtonContainer,
            self.centerButtonContainer,
            self.rightButtonContainer,
            self.skipBackButton,
            self.skipForwardButton,
        ):
            hidden.hide()
        line = QHBoxLayout()
        line.setSpacing(8)
        # The waveform takes the slider's place; the slider stays as the library's position model.
        # Created with its parent: shown before the row is installed, a parentless one would
        # flash on screen as a window of its own.
        self.waveform = Waveform(self)
        for widget, stretch in (
            (self.playButton, 0),
            (self.currentTimeLabel, 0),
            (self.waveform, 1),
            (self.remainTimeLabel, 0),
            (self.volumeButton, 0),
        ):
            line.addWidget(widget, stretch)
            widget.show()
        self.progressSlider.hide()
        # A failure replaces the scrubber with its reason, where the user is looking.
        self.error = ErrorLabel()
        self.error.hide()
        line.insertWidget(2, self.error, 1)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.addLayout(line)
        self.setFixedHeight(52)

    @Slot(int)
    def _duration_changed(self, duration):
        # The library updates time labels on position changes only. Initial
        # metadata can arrive while position remains zero, before any playback.
        self._onPositionChanged(self.player.position())

    def _show_error(self, message=None):
        failed = message is not None
        for widget in (self.currentTimeLabel, self.waveform, self.remainTimeLabel):
            widget.setVisible(not failed)
        self.error.setVisible(failed)
        if failed:
            set_text(self.error, message)
            bind(self, message, "setToolTip")
        else:
            self.error.clear()
            bind(self, "", "setToolTip")

    @Slot(object, str)
    def _failed(self, _error, message):
        if self.path is not None:
            self._show_error(t("widgets.playback_failed") + message)

    def load(self, path, *, autoplay=False):
        file = Path(path)
        if not file.is_file():
            self.clear()
            self._show_error(t("widgets.audio_file_missing", p1=file.name))
            return
        signature = (
            (str(file), file.stat().st_size, file.stat().st_mtime_ns)
            if file.is_file()
            else (str(file), None, None)
        )
        if signature == self.source_signature and self.player.error().value == 0:
            if autoplay:
                self.start()
            return
        self.source_signature = signature
        self.path = str(path)
        self.setEnabled(True)
        self._show_error()
        self.read_peaks(file, signature)
        self.waveform.set_position(0)
        self._queue_source(QUrl.fromLocalFile(str(path)))
        if autoplay:
            self.start()

    def read_peaks(self, file, signature):
        """Decoding a long recording takes a while: the waveform follows when it is done."""
        if signature in _peaks:
            self.waveform.set_peaks(_peaks[signature])
            return
        self.waveform.set_peaks(None)

        def read():
            try:
                peaks = audio_peaks(file, signature)
            except Exception:
                peaks = None
            try:
                self.peaks_read.emit(signature, peaks)
            except RuntimeError:
                # The player closed while this ran.
                pass

        QThreadPool.globalInstance().start(read)

    @Slot(object, object)
    def show_peaks(self, signature, peaks):
        if signature == self.source_signature:
            self.waveform.set_peaks(peaks)

    def start(self):
        QMetaObject.invokeMethod(self.player, "play", Qt.ConnectionType.QueuedConnection)

    def togglePlayState(self):
        if self.player.isPlaying():
            QMetaObject.invokeMethod(self.player, "pause", Qt.ConnectionType.QueuedConnection)
        else:
            self.start()

    def _onPositionChanged(self, position):
        super()._onPositionChanged(position)
        duration = self.player.duration()
        self.waveform.set_position(position / duration if duration > 0 else 0)

    def _queue_source(self, source):
        # Native queued slots leave Python/media signal dispatch before the backend
        # joins its decoder. A synchronous stop after seek in a Python callback can
        # deadlock PySide/Fluent (see the isolated playback dispatch regression).
        QMetaObject.invokeMethod(self.player, "stop", Qt.ConnectionType.QueuedConnection)
        QMetaObject.invokeMethod(
            self.player, "setSource", Qt.ConnectionType.QueuedConnection, Q_ARG(QUrl, source)
        )

    def stop(self):
        """Stop outside media callbacks, then notify owners it is safe to close."""
        QMetaObject.invokeMethod(self.player, "stop", Qt.ConnectionType.QueuedConnection)
        QMetaObject.invokeMethod(self, "_after_stop", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _after_stop(self):
        self.stopped.emit()

    def playing(self, path) -> bool:
        return (
            self.path is not None
            and Path(self.path) == Path(path)
            and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        )

    def toggle(self, path):
        """Play path from where it was, or pause it when it is the one playing."""
        if self.playing(path):
            QMetaObject.invokeMethod(self.player, "pause", Qt.ConnectionType.QueuedConnection)
        else:
            self.load(path, autoplay=True)

    def clear(self):
        self._queue_source(QUrl())
        self.path = None
        self.source_signature = None
        self.setEnabled(False)
        self.progressSlider.setValue(0)
        self.waveform.set_peaks(None)
        self.waveform.set_position(0)
        set_text(self.currentTimeLabel, "0:00:00")
        set_text(self.remainTimeLabel, "0:00:00")
        self._show_error()


STATE_FILLS = {"done": "#107c10", "failed": "#c42b1c", "warn": "#c27c0e"}


def step_icon(number, state, *, dot=False, size=16):
    """Round marker: the number while pending or active, a check mark when done, a cross when
    failed and an exclamation mark for a warning. number=None draws a plain state dot."""
    scale = 2
    pixmap = QPixmap(size * scale, size * scale)
    pixmap.setDevicePixelRatio(scale)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    dark = isDarkTheme()
    area = QRectF(1, 1, size - 2, size - 2)
    if state in STATE_FILLS:
        fill, ink = QColor(STATE_FILLS[state]), QColor("#ffffff")
    elif state == "active":
        fill, ink = ACCENT, QColor(ON_ACCENT)
    else:
        fill, ink = QColor(0, 0, 0, 0), QColor("#9a9a9a" if dark else "#5b6068")
    painter.setBrush(fill)
    painter.setPen(QPen(ink if fill.alpha() == 0 else fill, 1.2))
    painter.drawEllipse(area)
    pen = QPen(ink, max(1.6, size / 10))
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    if state == "done":
        path = QPainterPath()
        path.moveTo(size * 0.3, size * 0.52)
        path.lineTo(size * 0.45, size * 0.66)
        path.lineTo(size * 0.72, size * 0.36)
        painter.drawPath(path)
    elif state == "failed":
        low, high = size * 0.35, size * 0.65
        painter.drawLine(QPointF(low, low), QPointF(high, high))
        painter.drawLine(QPointF(high, low), QPointF(low, high))
    elif state == "warn":
        painter.drawLine(QPointF(size / 2, size * 0.28), QPointF(size / 2, size * 0.56))
        painter.drawPoint(QPointF(size / 2, size * 0.72))
    elif number is not None:
        font = painter.font()
        font.setPixelSize(int(size * 0.62))
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(area, Qt.AlignmentFlag.AlignCenter, str(number))
    elif state == "active":
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QPointF(size / 2, size / 2), size * 0.14, size * 0.14)
    if dot:
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#f2c14e"))
        painter.drawEllipse(QRectF(size - 6, 0, 6, 6))
    painter.end()
    return QIcon(pixmap)


def _solid_tint(view):
    # The library's tint (200/255 dark, 160/255 light) let the page show through wherever
    # the blur of the grabbed screen was weak: the list read over the fields under it.
    if isDarkTheme():
        view.acrylicBrush.tintColor = QColor(40, 40, 40, 244)
        view.acrylicBrush.luminosityColor = QColor(0, 0, 0, 0)
    else:
        view.acrylicBrush.tintColor = QColor(250, 250, 250, 240)
        view.acrylicBrush.luminosityColor = QColor(255, 255, 255, 0)


class _SolidMenuList(AcrylicMenuActionListWidget):
    def _updateAcrylicColor(self):
        _solid_tint(self)


class _SolidComboList(AcrylicComboMenuActionListWidget):
    def _updateAcrylicColor(self):
        _solid_tint(self)


class AcrylicMenu(AcrylicMenuBase, RoundMenu):
    """A menu on frosted acrylic that is nearly opaque, so its items never read over the page."""

    def __init__(self, title="", parent=None):
        super().__init__(title, parent)
        self.setUpMenu(_SolidMenuList(self))

    # Without this override a call on the instance reaches QMenu.exec, which rejects the
    # aniType a drop-down button passes: the click raised and no menu opened.
    def exec(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        return super().exec(pos, ani, aniType)


class _SolidComboMenu(AcrylicMenuBase, RoundMenu):
    # The library's AcrylicComboBoxMenu with the nearly opaque list.
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setUpMenu(_SolidComboList(self))
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setItemDelegate(IndicatorMenuItemDelegate())
        self.view.setObjectName("comboListWidget")
        self.setItemHeight(33)

    def exec(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        return super().exec(pos, ani, aniType)


class ElidedComboBox(AcrylicComboBox):
    """A drop-down whose shown item elides to the width the layout gives it, full text in the
    tooltip: a long project or model name never widens the row it sits in."""

    full_text = ""
    # The library's padding left of the text and the room it keeps for the arrow.
    PADDING = 10 + 29

    def setText(self, text):
        self.full_text = str(text)
        self._fit()

    def _fit(self):
        shown = self.fontMetrics().elidedText(
            self.full_text, Qt.TextElideMode.ElideRight, max(0, self.width() - self.PADDING)
        )
        QPushButton.setText(self, shown)
        self.setToolTip(self.full_text if shown != self.full_text else "")

    def sizeHint(self):
        hint = super().sizeHint()
        width = self.fontMetrics().horizontalAdvance(self.full_text) + self.PADDING + 12
        return QSize(max(hint.width(), min(width, 360)), hint.height())

    def minimumSizeHint(self):
        return QSize(120, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def _createComboMenu(self):
        return _SolidComboMenu(self)


def combo_box():
    """A drop-down whose list opens on frosted acrylic, like the menus."""
    return ElidedComboBox()


class ChoiceBox(ElidedComboBox):
    """A drop-down that may be left unset (no item chosen, an empty field), read and written
    like a spin box: value() is the chosen item's data or None."""

    valueChanged = Signal(object)

    def __init__(self):
        super().__init__()
        self.currentIndexChanged.connect(lambda _index: self.valueChanged.emit(self.value()))

    def value(self):
        return self.currentData()

    def setPlaceholderText(self, text):
        # The library also writes the placeholder over the shown item when the first item is
        # chosen; keep it for when nothing is.
        if self.currentIndex() >= 0:
            self._placeholderText = text
        else:
            super().setPlaceholderText(text)

    def setValue(self, value):
        index = next((i for i in range(self.count()) if self.itemData(i) == value), -1)
        if index >= 0:
            self.setCurrentIndex(index)
        elif self.currentIndex() >= 0:
            # The library sets -1 without telling anyone.
            self.setCurrentIndex(-1)
            self.valueChanged.emit(None)


# The narrowest a detail panel beside a list gets; the list gives up its width first.
DETAIL_MINIMUM = 300


class SplitRow(QWidget):
    """A leading and a trailing part side by side (a toolbar's two groups, a page's two
    columns); when the line is narrower than both need, the trailing part moves under the
    leading one instead of pushing the page wider. stretch shares spare width between them."""

    def __init__(self, leading, trailing, *, spacing=8, stretch=(1, 0)):
        super().__init__()
        self.leading, self.trailing = leading, trailing
        self.stretch = stretch
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(spacing)
        # The row's minimum is the wider group, not both: below both it stacks them.
        self.grid.setSizeConstraint(QGridLayout.SizeConstraint.SetNoConstraint)
        self.stacked = None
        self._place(False)

    def _needed(self):
        return (
            self.leading.minimumSizeHint().width()
            + self.grid.horizontalSpacing()
            + self.trailing.minimumSizeHint().width()
        )

    def _place(self, stacked):
        if stacked == self.stacked:
            return
        self.stacked = stacked
        self.grid.removeWidget(self.leading)
        self.grid.removeWidget(self.trailing)
        self.grid.addWidget(self.leading, 0, 0)
        if stacked:
            self.grid.addWidget(self.trailing, 1, 0)
            self.grid.setColumnStretch(0, 1)
            self.grid.setColumnStretch(1, 0)
        else:
            self.grid.addWidget(self.trailing, 0, 1)
            self.grid.setColumnStretch(0, self.stretch[0])
            self.grid.setColumnStretch(1, self.stretch[1])
        self.updateGeometry()

    def minimumSizeHint(self):
        width = max(self.leading.minimumSizeHint().width(), self.trailing.minimumSizeHint().width())
        return QSize(width, self.grid.minimumSize().height())

    def sizeHint(self):
        return QSize(self._needed(), self.grid.sizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._place(event.size().width() < self._needed())

    def event(self, event):
        # A part's needs change without a resize, e.g. its buttons' text in another language.
        if event.type() == QEvent.Type.LayoutRequest:
            self._place(self.width() < self._needed())
        return super().event(event)


class _EmptyState(QWidget):
    def hasHeightForWidth(self):
        # Its wrapped hint would otherwise hold the card it fills at its preferred height.
        return False


def empty_state(text):
    """A centred, wrapping hint that fills a detail panel while nothing is selected."""
    widget = _EmptyState()
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    hint = label(text, "muted")
    hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    layout.addWidget(hint)
    return widget


class _InnerScroll(ScrollArea):
    def sizeHint(self):
        # A card holding wrapped text keeps its preferred height in a column (Qt's layouts
        # never shrink a height-for-width item below it), so the content's full height here
        # would make the page scroll instead of this area. Spare height still comes: it expands.
        return QSize(super().sizeHint().width(), 80)


def inner_scroll(content):
    """content in a frameless, transparent area that scrolls vertically when the card it sits
    in is shorter than it, so the card's own footer (its main action) stays in view."""
    area = _InnerScroll()
    area.setWidgetResizable(True)
    area.setFrameShape(QScrollArea.Shape.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    area.setWidget(content)
    content.setAutoFillBackground(False)
    area.enableTransparentBackground()
    area.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
    return area


def menu_button(text, entries, *, icon=None):
    """A dropdown button; entries are (text, callback) pairs."""
    widget = bind(DropDownPushButton(str(text)), text)
    if icon is not None:
        widget.setIcon(icon)
    widget.setMinimumHeight(32)
    menu = AcrylicMenu(parent=widget)
    for title, callback in entries:
        action = Action(str(title))
        bind(action, title)
        action.triggered.connect(callback)
        menu.addAction(action)
    widget.setMenu(menu)
    return widget


def show_menu(anchor, entries, *, at=None):
    """Pop a menu under anchor (or at a global point); entries are (text, callback, enabled)."""
    menu = AcrylicMenu(parent=anchor)
    # exec does not block, and every call builds a new menu.
    menu.closedSignal.connect(menu.deleteLater)
    for title, callback, enabled in entries:
        action = Action(str(title))
        action.setEnabled(enabled)
        action.triggered.connect(callback)
        menu.addAction(action)
    menu.exec(at or anchor.mapToGlobal(QPoint(0, anchor.height())))


class LogDialog(Dialog):
    def __init__(self, parent, title, text):
        super().__init__(parent)
        self.viewLayout.addWidget(translated_widget(SubtitleLabel, title, self))
        self.log = smooth(PlainTextEdit(self))
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(1500)
        self.log.setPlainText(str(text))
        self.log.setMinimumSize(640, 320)
        self.viewLayout.addWidget(self.log)
        self.yesButton.hide()
        set_text(self.cancelButton, t("widgets.close"))
        self.widget.setMinimumWidth(720)


def show_log(parent, title, text):
    LogDialog(parent.window(), title, text).exec()


class ModelNotice(QWidget):
    """Where an action needs the speech model and it is not there: why the action is off, and
    the way to get the model."""

    def __init__(self, callback):
        super().__init__()
        from . import model_files
        from .asr_dialog import human_size

        size = human_size(model_files.download_for(model_files.SPEECH).size)
        text = label(t("models.speech_needed", size=size))
        text.setTextColor(QColor("#9d5d00"), QColor("#f5d58a"))
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(text, 1)
        layout.addWidget(button(t("models.get_speech_model"), callback, icon=FluentIcon.DOWNLOAD))
        self.hide()


class Callout(QWidget):
    """A line set apart on a tint of the theme colour while it asks for something, and plain
    once it only informs. The padding stays either way, so nothing moves when it changes."""

    def __init__(self, content):
        super().__init__()
        self.tinted = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(content)

    def set_tinted(self, tinted):
        self.tinted = bool(tinted)
        self.update()

    def paintEvent(self, _event):
        if not self.tinted:
            return
        dark = isDarkTheme()
        fill, edge = QColor(accent_line()), QColor(accent_line())
        fill.setAlpha(46 if dark else 26)
        edge.setAlpha(120 if dark else 100)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(edge, 1))
        painter.setBrush(fill)
        painter.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 6, 6)


def card_header(title, *actions, meta=None):
    """A card's first line: its title, an optional short fact right after it, and its actions
    at the right end. Every card starts with this line, so titles and actions line up."""
    widget = QWidget()
    layout = QHBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    widget.title = label(title, "card", wrap=False)
    layout.addWidget(widget.title)
    widget.meta = elided_label(meta or "")
    widget.meta.setTextColor(QColor("#5b6068"), QColor("#bfc5ce"))
    layout.addWidget(widget.meta, 1)
    for action in actions:
        layout.addWidget(action)
    return widget


class _ModelRows(QAbstractListModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows = []
        locale_manager().changed.connect(self._retranslate)

    def _retranslate(self, _language):
        if self.rows:
            self.dataChanged.emit(self.index(0), self.index(len(self.rows) - 1))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return str(row["name"])
        if role == Qt.ItemDataRole.ToolTipRole and not row.get("section"):
            return f"{row['name']}\n{row.get('detail', '')}".strip()
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None

    def flags(self, index):
        if index.isValid() and self.rows[index.row()].get("section"):
            return Qt.ItemFlag.ItemIsEnabled
        return super().flags(index)


class _ModelDelegate(TableItemDelegate):
    """Two lines per model (name, then a muted state line); a section title is one small line."""

    def sizeHint(self, option, index):
        row = index.data(Qt.ItemDataRole.UserRole) or {}
        return QSize(option.rect.width(), 30 if row.get("section") else 56)

    def _drawBackground(self, painter, option, index):
        painter.drawRoundedRect(option.rect.adjusted(2, 1, -2, -1), 5, 5)

    def _drawIndicator(self, painter, option, index):
        y, h = option.rect.y(), option.rect.height()
        painter.setBrush(ACCENT)
        painter.drawRoundedRect(QRectF(3, y + h * 0.3, 3, h * 0.4), 1.5, 1.5)

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        option.text = ""

    def paint(self, painter, option, index):
        row = index.data(Qt.ItemDataRole.UserRole) or {}
        dark = isDarkTheme()
        muted = QColor("#9a9a9a" if dark else "#5b6068")
        rect = option.rect.adjusted(14, 0, -10, 0)
        left = Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        if row.get("section"):
            painter.save()
            painter.setFont(getFont(12))
            painter.setPen(muted)
            metrics = painter.fontMetrics()
            text = metrics.elidedText(str(row["name"]), Qt.TextElideMode.ElideRight, rect.width())
            painter.drawText(rect.adjusted(-6, 6, 0, 0), left, text)
            painter.restore()
            return
        super().paint(painter, option, index)
        painter.save()
        top = QRect(rect.x(), rect.y() + 9, rect.width(), 20)
        bottom = QRect(rect.x(), rect.y() + 29, rect.width(), 18)
        painter.setFont(getFont(14, QFont.Weight.DemiBold))
        painter.setPen(QColor("#ffffff" if dark else "#1f1f1f"))
        metrics = painter.fontMetrics()
        name = metrics.elidedText(str(row["name"]), Qt.TextElideMode.ElideRight, top.width())
        painter.drawText(top, left, name)
        painter.setFont(getFont(12))
        painter.setPen(tone_color(row["tone"]) if row.get("tone") else muted)
        metrics = painter.fontMetrics()
        detail = metrics.elidedText(
            str(row.get("detail", "")), Qt.TextElideMode.ElideRight, bottom.width()
        )
        painter.drawText(bottom, left, detail)
        painter.restore()


class ModelList(ListView):
    """The models page's list: rows are dicts with id, name, detail (a state line) and an
    optional tone; {"section": True, "name": ...} rows are titles between groups."""

    selected = Signal(dict)

    def __init__(self):
        super().__init__()
        self.rows_model = _ModelRows(self)
        self.setModel(self.rows_model)
        self.setItemDelegate(_ModelDelegate(self))
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setUniformItemSizes(False)
        self.selectionModel().currentChanged.connect(self._current)
        bind(self, t("widgets.selectable_records"), "setAccessibleName")

    def _current(self, index, _old):
        row = index.data(Qt.ItemDataRole.UserRole) if index.isValid() else None
        if row and not row.get("section"):
            self.selected.emit(row)

    def item(self):
        row = self.currentIndex().data(Qt.ItemDataRole.UserRole)
        return row if row and not row.get("section") else None

    def update_rows(self, rows):
        """Replace the rows, keeping the current model selected when it is still listed."""
        current = self.item()
        old = [row["id"] for row in self.rows_model.rows]
        if old == [row["id"] for row in rows]:
            # The same models with new state lines: repaint them, keep everything else.
            self.rows_model.rows = list(rows)
            if rows:
                self.rows_model.dataChanged.emit(
                    self.rows_model.index(0), self.rows_model.index(len(rows) - 1)
                )
            return
        selection = self.selectionModel()
        selection.blockSignals(True)
        self.rows_model.beginResetModel()
        self.rows_model.rows = list(rows)
        self.rows_model.endResetModel()
        selection.blockSignals(False)
        if current is not None:
            self.select_id(current["id"], emit=False)
        self.updateSelectedRows()

    def select_id(self, row_id, *, emit=True):
        for position, row in enumerate(self.rows_model.rows):
            if not row.get("section") and row["id"] == row_id:
                selection = self.selectionModel()
                selection.blockSignals(not emit)
                self.setCurrentIndex(self.rows_model.index(position))
                selection.blockSignals(False)
                return True
        return False


def scientific(value):
    """1e-4, 2.5e-5: the one way the application writes a learning rate."""
    mantissa, exponent = f"{value:.2e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".")
    return f"{mantissa}e{int(exponent)}"


class LearningRateBox(DoubleSpinBox):
    """A learning rate written the way it is read everywhere else: 1e-4, 5e-5."""

    def __init__(self):
        super().__init__()
        self.setDecimals(8)

    def textFromValue(self, value):
        return scientific(value)

    def valueFromText(self, text):
        try:
            return float(text)
        except ValueError:
            return self.value()

    def validate(self, text, position):
        try:
            float(text)
        except ValueError:
            return QValidator.State.Intermediate, text, position
        return QValidator.State.Acceptable, text, position


class OptionalSpinBox(SpinBox):
    """A whole number that may be left unset: 0 shows as an empty field."""

    def textFromValue(self, value):
        return "" if value == 0 else str(value)

    def valueFromText(self, text):
        return int(text) if text.strip() else 0

    def validate(self, text, position):
        if not text.strip():
            return QValidator.State.Acceptable, text, position
        return super().validate(text, position)
