"""Assemble the pinned upstream baseline plus the desktop modules into a directory app."""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath

from irodori_tts.desktop import asr
from irodori_tts.desktop.exports import environment_report
from irodori_tts.desktop.resources import RESOURCE_SPEC
from irodori_tts.desktop.storage import digest, now, write_json
from irodori_tts.upstream import UPSTREAM_COMMIT, UPSTREAM_ROOT, UPSTREAM_URL

REPO = Path(__file__).resolve().parents[1]
FFMPEG = REPO / "runtime/ffmpeg"


def git_at(directory, *arguments, check=True):
    return subprocess.run(
        ["git", "-c", f"safe.directory={directory.as_posix()}", "-C", str(directory), *arguments],
        capture_output=True,
        check=check,
    )


def source_provenance():
    head = git_at(UPSTREAM_ROOT, "rev-parse", "HEAD").stdout.decode().strip()
    dirty = git_at(
        UPSTREAM_ROOT, "status", "--porcelain=v1", "--untracked-files=all"
    ).stdout.decode()
    link = git_at(REPO, "ls-files", "--stage", "--", "upstream/Irodori-TTS").stdout.decode().split()
    if head != UPSTREAM_COMMIT or dirty or link[:2] != ["160000", UPSTREAM_COMMIT]:
        raise RuntimeError("Build requires the exact clean pinned upstream submodule and gitlink")
    commit = git_at(REPO, "rev-parse", "--verify", "HEAD", check=False)
    status = git_at(REPO, "status", "--porcelain=v1", "--untracked-files=all").stdout.decode()
    return {
        "upstream": {"url": UPSTREAM_URL, "commit": head, "gitlink": link[1], "clean": True},
        "desktop": {
            "project": "Irodori-TTS-Desktop",
            "commit": commit.stdout.decode().strip() if commit.returncode == 0 else None,
            "state": "uncommitted" if status or commit.returncode != 0 else "committed",
            "unborn": commit.returncode != 0,
            "working_tree_status": status.splitlines(),
        },
    }


def assemble_app(app, provenance):
    # Export actual fixed Git objects; the dirty outer checkout is never copied
    # into upstream. Runtime apps are flat complete packages, with no Git lookup.
    archive = git_at(UPSTREAM_ROOT, "archive", "--format=zip", UPSTREAM_COMMIT).stdout
    with zipfile.ZipFile(io.BytesIO(archive)) as baseline:
        for name in baseline.namelist():
            if not (app / name).resolve().is_relative_to(app.resolve()):
                raise ValueError("Unexpected upstream archive path")
        baseline.extractall(app)
    for path in REPO.glob("*.py"):
        shutil.copy2(path, app / path.name)
    for name in ("irodori_tts", "examples"):
        shutil.copytree(REPO / name, app / name, dirs_exist_ok=True, ignore=ignore_artifacts)
    write_json(app / "UPSTREAM_SOURCE.json", provenance["upstream"])


def bundle_addons(destination):
    """Speech recognition add-ons are installed here, at build time, into resources/addons
    (where a launcher's IRODORI_CACHE makes the application look). Users never install Python
    packages. An add-on already current is kept, and one current in the checkout's own cache
    is copied, so a build needs no network."""
    root = destination / "resources/addons"
    local = REPO / ".cache/addons"
    for key in asr.ADDONS:
        if asr.addon_state(key, root)["ready"]:
            continue
        if asr.addon_state(key, local)["ready"]:
            print(f"Copying the {key} add-on into the package", flush=True)
            shutil.rmtree(root / key, ignore_errors=True)
            shutil.copytree(local / key, root / key, ignore=ignore_artifacts)
            continue
        print(f"Installing the {key} add-on into the package", flush=True)
        asr.install_addon(key, root=root, progress=lambda line: print(line, flush=True))


LAUNCHER = "Irodori-TTS-Desktop.exe"


def msvc_environment():
    """vcvars64.bat of the newest Visual Studio with the C++ tools, found through vswhere."""
    vswhere = (
        Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    found = ""
    if vswhere.is_file():
        found = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
    vcvars = Path(found) / "VC/Auxiliary/Build/vcvars64.bat" if found else None
    if not vcvars or not vcvars.is_file():
        raise RuntimeError(
            "The launcher needs Visual Studio's C++ build tools (MSVC and the Windows SDK)"
        )
    return vcvars


def build_launcher(destination):
    """Compile scripts/launcher/launcher.c into the package's exe, with the app icon and
    version details. The C runtime is linked in, so the exe needs nothing installed."""
    work = REPO / "build/launcher"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    version = re.search(
        r'^version = "([^"]+)"', (REPO / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )[1]
    numbers = ",".join((version.split(".") + ["0"] * 4)[:4])
    icon = (REPO / "irodori_tts/desktop/assets/icon.ico").as_posix()
    (work / "launcher.rc").write_text(
        f'1 ICON "{icon}"\n'
        "1 VERSIONINFO\n"
        f"FILEVERSION {numbers}\nPRODUCTVERSION {numbers}\n"
        "FILEOS 0x40004\nFILETYPE 0x1\n"
        'BEGIN\n BLOCK "StringFileInfo"\n BEGIN\n  BLOCK "040904B0"\n  BEGIN\n'
        '   VALUE "CompanyName", "Irodori-TTS-Desktop"\n'
        '   VALUE "FileDescription", "Irodori TTS Desktop"\n'
        f'   VALUE "FileVersion", "{version}"\n'
        '   VALUE "ProductName", "Irodori TTS Desktop"\n'
        f'   VALUE "ProductVersion", "{version}"\n'
        f'   VALUE "OriginalFilename", "{LAUNCHER}"\n'
        '  END\n END\n BLOCK "VarFileInfo"\n BEGIN\n  VALUE "Translation", 0x409, 1200\n END\n'
        "END\n",
        encoding="utf-8",
    )
    source = REPO / "scripts/launcher/launcher.c"
    (work / "build.bat").write_text(
        "@echo off\r\n"
        f'call "{msvc_environment()}" >nul || exit /b 1\r\n'
        "rc /nologo /c 65001 /fo launcher.res launcher.rc || exit /b 1\r\n"
        # /Brepro: the same source gives the same exe, so a rebuild changes nothing.
        f'cl /nologo /utf-8 /O1 /MT /W4 /WX /Brepro /DUNICODE /D_UNICODE "{source}" launcher.res'
        f" /Fe:{LAUNCHER} /link /Brepro /SUBSYSTEM:WINDOWS user32.lib || exit /b 1\r\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["cmd", "/c", str(work / "build.bat")],
        cwd=work,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode:
        raise RuntimeError(f"Building the launcher failed:\n{result.stdout}{result.stderr}")
    shutil.copy2(work / LAUNCHER, destination / LAUNCHER)


def seven_zip():
    for candidate in (
        shutil.which("7z"),
        Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "7-Zip/7z.exe",
    ):
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise RuntimeError("The build needs 7-Zip (7z.exe on PATH or in Program Files)")


def bundle_seven_zip(destination):
    """The package carries 7-Zip's console program, which imports 7z model packages. Its
    license travels with it, as the LGPL asks."""
    source = Path(seven_zip()).parent
    target = destination / "runtime/7zip"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("7z.exe", "7z.dll", "License.txt"):
        shutil.copy2(source / name, target / name)


def unpack_ffmpeg(archive, target=FFMPEG):
    """Keep what the application uses of an FFmpeg 'shared' build zip (BtbN's FFmpeg-Builds
    layout): the libraries torchaudio loads, the programs, and the license."""
    with zipfile.ZipFile(archive) as source:
        wanted = {}
        for info in source.infolist():
            path = PurePosixPath(info.filename)
            if path.name == "LICENSE.txt" and len(path.parts) == 2:
                wanted[path.name] = info
            elif path.parent.name == "bin" and (
                path.suffix.lower() == ".dll" or path.name in ("ffmpeg.exe", "ffprobe.exe")
            ):
                wanted[path.name] = info
        if not any(name.startswith("avcodec-") for name in wanted):
            raise RuntimeError(f"{archive} is not an FFmpeg 'shared' build: it has no libraries")
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        for name, info in wanted.items():
            with source.open(info) as stream, (target / name).open("wb") as out:
                shutil.copyfileobj(stream, out)


def bundle_ffmpeg(destination, source=FFMPEG):
    """The package carries FFmpeg: speech cannot be read or written without it. Its license
    travels with it."""
    if not any(source.glob("avcodec-*.dll")):
        raise RuntimeError(
            "The build needs FFmpeg: pass --ffmpeg with a win64 'shared' build zip once"
        )
    target = destination / "runtime/ffmpeg"
    shutil.rmtree(target, ignore_errors=True)
    shutil.copytree(source, target)


# What a package collects while it is used: the user's data and downloads. Neither belongs in
# a build's manifest or archive.
USER_STATE = ("data", "resources/huggingface", "resources/torch")


def precompile(destination):
    """Ship compiled bytecode, so the first start does not compile thousands of modules, nor
    every start when the folder is read-only. The runtime and add-ons never change after the
    build, so their bytecode is not checked against the sources; the app's is."""
    import compileall
    import py_compile

    print("Compiling bytecode", flush=True)
    mode = py_compile.PycInvalidationMode
    for folder, invalidation in (
        (destination / "runtime/python/Lib", mode.UNCHECKED_HASH),
        (destination / "resources/addons", mode.UNCHECKED_HASH),
        (destination / "app", mode.CHECKED_HASH),
    ):
        # A few third-party files only parse on other Python versions; they are never
        # imported here, so their failure is not the build's.
        compileall.compile_dir(folder, quiet=2, workers=0, invalidation_mode=invalidation)


def write_archive(destination):
    """dist/<package>.7z with the package folder at its top, for handing to users. 7-Zip
    compresses on every core; the data and downloads of a package that was tried stay out."""
    archive = destination.with_suffix(".7z")
    partial = archive.with_name(f"{archive.name}.partial")
    partial.unlink(missing_ok=True)
    print(f"Writing {archive.name}", flush=True)
    subprocess.run(
        [
            seven_zip(),
            "a",
            "-t7z",
            "-mx=5",
            "-mmt=on",
            "-bso0",
            "-bsp0",
            str(partial),
            destination.name,
            *(f"-x!{destination.name}\\{Path(state)}" for state in USER_STATE),
        ],
        cwd=destination.parent,
        check=True,
    )
    partial.replace(archive)
    return archive


def ignore_artifacts(directory, names):
    return [
        name
        for name in names
        if name in {"__pycache__", ".locks", ".git"} or name.endswith((".pyc", ".lock"))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO / "dist/Irodori-TTS-Desktop")
    parser.add_argument("--refresh-app", action="store_true")
    parser.add_argument(
        "--archive", action="store_true", help="also write dist/<package>.7z with 7-Zip"
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        help="an FFmpeg win64 'shared' build zip, unpacked into runtime/ffmpeg for this and later builds",
    )
    args = parser.parse_args()
    # First, like the compiler check below: nothing is worth copying without it.
    if args.ffmpeg:
        unpack_ffmpeg(args.ffmpeg)
    if not any(FFMPEG.glob("avcodec-*.dll")):
        raise RuntimeError(
            "The build needs FFmpeg: pass --ffmpeg with a win64 'shared' build zip once"
        )
    from scripts.translations import compile_catalogs, validate

    validate()
    compile_catalogs()
    provenance = source_provenance()
    destination = args.output.resolve()
    allowed = (REPO / "dist").resolve()
    if not destination.is_relative_to(allowed) or destination == allowed:
        raise ValueError("Build output must be a dedicated directory inside this project's dist")
    destination.mkdir(parents=True, exist_ok=True)
    # First, so a missing compiler stops the build before gigabytes are copied.
    build_launcher(destination)
    app = destination / "app"
    if app.exists():
        if not app.resolve().is_relative_to(allowed):
            raise ValueError("Application directory resolves outside dist")
        shutil.rmtree(app)
    app.mkdir()
    assemble_app(app, provenance)
    runtime = destination / "runtime/python"
    if not args.refresh_app:
        print("Copying independent Python standard library and CUDA/Qt environment", flush=True)
        shutil.copytree(Path(sys.base_prefix), runtime, dirs_exist_ok=True, ignore=ignore_artifacts)
        site = runtime / "Lib/site-packages"
        shutil.copytree(
            Path(sys.prefix) / "Lib/site-packages",
            site,
            dirs_exist_ok=True,
            ignore=ignore_artifacts,
        )
        for name in ("_virtualenv.pth", "_virtualenv.py"):
            (site / name).unlink(missing_ok=True)
        # No venv metadata, machine-specific Python path, or credentials are shipped.
    # Models are not shipped: every one of them can be downloaded (or imported from another
    # user's files) in the application, which verifies them against model_pins.json.
    packaged_hub = destination / "resources/huggingface/hub"
    if packaged_hub.exists():
        if not packaged_hub.resolve().is_relative_to(allowed):
            raise ValueError("Resource directory resolves outside dist")
        shutil.rmtree(packaged_hub)
    bundle_addons(destination)
    bundle_seven_zip(destination)
    bundle_ffmpeg(destination)
    shutil.copy2(REPO / "LICENSE", destination / "LICENSE")
    # Files an earlier build left at the top level.
    for name in (
        "Launch.vbs",
        "Irodori-TTS.exe",
        "DO-NOT-DISTRIBUTE.txt",
        "THIRD_PARTY_NOTICES.md",
        "dependencies.json",
        "requirements-snapshot.txt",
        "package-manifest.json",
    ):
        (destination / name).unlink(missing_ok=True)
    precompile(destination)
    print("Hashing package contents", flush=True)
    manifest = app / "package-manifest.json"
    user_state = [destination / state for state in USER_STATE]
    files = {}
    for path in destination.rglob("*"):
        if (
            path.is_file()
            and path != manifest
            and not any(path.is_relative_to(state) for state in user_state)
            and "__pycache__" not in path.parts
            and not path.name.endswith(".pyc")
            and (not path.name.endswith(".lock") or path.name == "uv.lock")
        ):
            files[path.relative_to(destination).as_posix()] = {
                "size": path.stat().st_size,
                "sha256": digest(path),
            }
    write_json(
        manifest,
        {
            "format_version": 3,
            "created": now(),
            "independent_runtime": True,
            **provenance,
            "resources": RESOURCE_SPEC,
            "environment": environment_report(),
            "hardware": "Windows x64, NVIDIA CUDA-capable GPU, compatible installed NVIDIA driver; development RTX 5080/CUDA 12.8",
            "files": files,
            "total_bytes": sum(value["size"] for value in files.values()),
        },
    )
    print(f"Built {destination}; {len(files)} files", flush=True)
    if args.archive:
        print(f"Wrote {write_archive(destination)}", flush=True)


if __name__ == "__main__":
    main()
