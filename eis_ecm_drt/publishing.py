"""Publish task-local EIS result files to the current user's desktop."""

import os
import re
import shutil
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from . import paths


def _safe_prefix(folder_name):
    value = re.sub(r'[^\w .-]+', "_", str(folder_name or "analysis"), flags=re.UNICODE)
    value = value.strip(" ._-")
    return value or "analysis"


def _validated_zip_targets(archive, destination):
    root = Path(destination).resolve()
    targets = []
    for info in archive.infolist():
        posix = PurePosixPath(info.filename)
        windows = PureWindowsPath(info.filename)
        if (
            posix.is_absolute()
            or windows.is_absolute()
            or windows.drive
            or ".." in posix.parts
            or ".." in windows.parts
        ):
            raise ValueError("unsafe ZIP member: %s" % info.filename)

        target = (root / Path(*posix.parts)).resolve()
        try:
            contained = os.path.commonpath([str(root), str(target)]) == str(root)
        except ValueError:
            contained = False
        if not contained:
            raise ValueError("unsafe ZIP member: %s" % info.filename)
        targets.append((info, target))
    return targets


def publish_analysis_results(source_path, folder_name=None):
    source = Path(source_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError("Analysis result source does not exist: %s" % source)

    if source.is_dir():
        source_type = "directory"
    elif source.is_file() and zipfile.is_zipfile(source):
        source_type = "zip"
    else:
        raise ValueError("source_path must be a result directory or ZIP file")

    output_dir = Path(paths.new_analysis_output_dir(prefix=_safe_prefix(folder_name)))
    try:
        if source_type == "directory":
            for child in source.iterdir():
                target = output_dir / child.name
                if child.is_dir():
                    shutil.copytree(child, target)
                else:
                    shutil.copy2(child, target)
        else:
            with zipfile.ZipFile(source) as archive:
                targets = _validated_zip_targets(archive, output_dir)
                for info, target in targets:
                    if info.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with archive.open(info) as reader, open(target, "wb") as writer:
                            shutil.copyfileobj(reader, writer)
    except Exception:
        shutil.rmtree(output_dir, ignore_errors=True)
        raise

    files = sorted(
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    return {
        "output_dir": str(output_dir.resolve()),
        "source_type": source_type,
        "files": files,
    }
