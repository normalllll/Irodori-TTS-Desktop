"""Reject incomplete wheels; supported deployment is checkout or assembled directory."""


def _unsupported(*args, **kwargs):
    raise RuntimeError(
        "Irodori-TTS-Desktop supports a recursive source checkout and the independent "
        "directory build only. Wheel/editable/sdist installation is intentionally unsupported; "
        "use python -m scripts.build_desktop after initializing the pinned submodule."
    )


build_wheel = _unsupported
build_editable = _unsupported
build_sdist = _unsupported
get_requires_for_build_wheel = _unsupported
get_requires_for_build_editable = _unsupported
get_requires_for_build_sdist = _unsupported
prepare_metadata_for_build_wheel = _unsupported
prepare_metadata_for_build_editable = _unsupported
