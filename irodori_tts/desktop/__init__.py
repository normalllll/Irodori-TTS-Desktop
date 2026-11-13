"""Chinese desktop application. Compute dependencies are imported by workers only."""

APP_VERSION = "0.1.1"
PROJECT_VERSION = 1
INITIALIZATION_RULES_VERSION = 1
# Automatic transcription: the backend that writes the transcript and the one that checks it.
ASR_DEFAULT_MAIN = "whisper-large-v3"
ASR_DEFAULT_CHECK = "qwen3-asr"
