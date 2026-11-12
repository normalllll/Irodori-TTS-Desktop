"""Application regression and opt-in real-device acceptance checks."""

import irodori_tts  # noqa: F401  (makes the fork's root scripts importable)

from .processes import configure_error_reporting

configure_error_reporting()
