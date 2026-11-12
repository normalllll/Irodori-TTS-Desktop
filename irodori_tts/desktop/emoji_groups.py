"""The palette's emoji by what they do, with a hint for each in the interface language."""

from __future__ import annotations

import re

from irodori_tts.desktop.messages import t
from irodori_tts.emoji_palette import EMOJI_PALETTE_ITEMS

EMOJI_GROUPS = (
    (
        t("emoji.group_emotion"),
        {
            "😆": t("emoji.joy"),
            "😊": t("emoji.cheerful"),
            "😌": t("emoji.relieved"),
            "😎": t("emoji.proud"),
            "😏": t("emoji.teasing"),
            "🫶": t("emoji.tender"),
            "🫣": t("emoji.shy"),
            "🙏": t("emoji.pleading"),
            "🤔": t("emoji.questioning"),
            "😲": t("emoji.surprised"),
            "😠": t("emoji.angry"),
            "🙄": t("emoji.exasperated"),
            "😟": t("emoji.worried"),
            "😰": t("emoji.flustered"),
            "🥺": t("emoji.trembling"),
            "😭": t("emoji.crying"),
            "😱": t("emoji.screaming"),
            "😖": t("emoji.pained"),
            "😪": t("emoji.sleepy"),
            "🥴": t("emoji.drunk"),
        },
    ),
    (
        t("emoji.group_pace"),
        {
            "⏸️": t("emoji.pause"),
            "⏩": t("emoji.fast"),
            "🐢": t("emoji.slow"),
            "💥": t("emoji.forceful"),
            "💪": t("emoji.powerful"),
            "📖": t("emoji.narration"),
        },
    ),
    (
        t("emoji.group_sounds"),
        {
            "🤭": t("emoji.giggle"),
            "😮‍💨": t("emoji.sigh"),
            "🌬️": t("emoji.panting"),
            "😮": t("emoji.gasp"),
            "🥱": t("emoji.yawn"),
            "🤧": t("emoji.cough"),
            "😒": t("emoji.tut"),
            "🎵": t("emoji.humming"),
            "👌": t("emoji.backchannel"),
            "🥵": t("emoji.moan"),
            "😴": t("emoji.sleep_talk"),
            "🥤": t("emoji.swallow"),
            "👅": t("emoji.licking"),
            "💋": t("emoji.lip_smack"),
            "👃": t("emoji.sniff"),
        },
    ),
    (
        t("emoji.group_space"),
        {
            "👂": t("emoji.whisper"),
            "📢": t("emoji.echo"),
            "📞": t("emoji.phone"),
            "🤐": t("emoji.muffled"),
        },
    ),
)


def emoji_name(hint) -> str:
    """The short name a hint starts with: "Crying: sobbing, sad" names it "Crying"."""
    return re.split("[:：]", str(hint), maxsplit=1)[0]


def grouped_emoji():
    """(title, [(emoji, hint), ...]) in display order. An emoji a newer upstream adds still
    shows, under Other with upstream's own words."""
    known = {emoji for _title, hints in EMOJI_GROUPS for emoji in hints}
    other = [
        (item.emoji, f"{item.label}: {item.description}")
        for item in EMOJI_PALETTE_ITEMS
        if item.emoji not in known
    ]
    groups = [(title, list(hints.items())) for title, hints in EMOJI_GROUPS]
    return groups + ([(t("emoji.group_other"), other)] if other else [])
