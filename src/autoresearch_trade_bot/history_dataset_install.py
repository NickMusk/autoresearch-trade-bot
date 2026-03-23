from __future__ import annotations

import json
import os
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .data import default_history_readiness_state_path


@dataclass(frozen=True)
class DatasetInstallConfig:
    archive_url: str
    storage_root: str
    manifest_relative_path: str
    readiness_state_path: str


@dataclass(frozen=True)
class DatasetInstallResult:
    manifest_path: Path | None
    installed: bool


def history_dataset_install_config_from_env() -> DatasetInstallConfig | None:
    archive_url = (
        os.environ.get("AUTORESEARCH_LLM_DATASET_INSTALL_URL")
        or os.environ.get("AUTORESEARCH_DATASET_INSTALL_URL")
        or ""
    ).strip()
    if not archive_url:
        return None
    storage_root = (
        os.environ.get("AUTORESEARCH_LLM_SHARED_DATA_ROOT")
        or os.environ.get("AUTORESEARCH_SHARED_DATA_ROOT")
        or os.environ.get("AUTORESEARCH_LLM_DATA_ROOT")
        or os.environ.get("AUTORESEARCH_DATA_ROOT")
        or "data"
    )
    manifest_relative_path = (
        os.environ.get("AUTORESEARCH_LLM_DATASET_MANIFEST_RELATIVE_PATH")
        or os.environ.get("AUTORESEARCH_DATASET_MANIFEST_RELATIVE_PATH")
        or ""
    ).strip()
    if not manifest_relative_path:
        raise ValueError(
            "dataset install URL is configured, but manifest relative path is missing"
        )
    readiness_state_path = (
        os.environ.get("AUTORESEARCH_LLM_HISTORY_READINESS_STATE_PATH")
        or os.environ.get("AUTORESEARCH_HISTORY_READINESS_STATE_PATH")
        or str(default_history_readiness_state_path(storage_root))
    )
    return DatasetInstallConfig(
        archive_url=archive_url,
        storage_root=storage_root,
        manifest_relative_path=manifest_relative_path,
        readiness_state_path=readiness_state_path,
    )


def maybe_install_history_dataset_from_env() -> DatasetInstallResult:
    config = history_dataset_install_config_from_env()
    if config is None:
        return DatasetInstallResult(manifest_path=None, installed=False)
    return ensure_history_dataset_installed(config)


def ensure_history_dataset_installed(config: DatasetInstallConfig) -> DatasetInstallResult:
    storage_root = Path(config.storage_root)
    manifest_path = storage_root / config.manifest_relative_path
    if manifest_path.exists():
        _write_readiness_state(
            readiness_state_path=Path(config.readiness_state_path),
            manifest_path=manifest_path,
            storage_root=storage_root,
        )
        return DatasetInstallResult(manifest_path=manifest_path, installed=False)

    storage_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="history-install-", dir=storage_root.parent) as tempdir:
        temp_root = Path(tempdir)
        archive_path = temp_root / "dataset.tar.gz"
        extracted_root = temp_root / "extract"
        extracted_root.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(config.archive_url, archive_path)
        _safe_extract_tarball(archive_path, extracted_root)
        _copy_extracted_tree(extracted_root, storage_root)

    if not manifest_path.exists():
        raise FileNotFoundError(
            "dataset archive did not produce the expected manifest at "
            f"{manifest_path}"
        )

    _write_readiness_state(
        readiness_state_path=Path(config.readiness_state_path),
        manifest_path=manifest_path,
        storage_root=storage_root,
    )
    return DatasetInstallResult(manifest_path=manifest_path, installed=True)


def _safe_extract_tarball(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = destination / member.name
            if not member_path.resolve().is_relative_to(destination.resolve()):
                raise ValueError(f"refusing to extract unsafe path from archive: {member.name}")
        archive.extractall(destination)


def _copy_extracted_tree(source_root: Path, destination_root: Path) -> None:
    for item in source_root.iterdir():
        destination = destination_root / item.name
        if item.is_dir():
            shutil.copytree(item, destination, dirs_exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, destination)


def _write_readiness_state(
    *,
    readiness_state_path: Path,
    manifest_path: Path,
    storage_root: Path,
) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(manifest["start"].replace("Z", "+00:00"))
    end = datetime.fromisoformat(manifest["end"].replace("Z", "+00:00"))
    lookback_days = int((end - start).total_seconds() // 86400)
    payload = {
        "manifest_path": str(manifest_path),
        "storage_root": str(storage_root),
        "lookback_days": lookback_days,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "source": "dataset_install",
    }
    readiness_state_path.parent.mkdir(parents=True, exist_ok=True)
    readiness_state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

