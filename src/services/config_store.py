import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict

import yaml

_LOCKS_GUARD = threading.Lock()
_CONFIG_LOCKS: Dict[str, threading.RLock] = {}


def _get_lock(path: Path) -> threading.RLock:
    key = str(path.resolve())
    with _LOCKS_GUARD:
        lock = _CONFIG_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _CONFIG_LOCKS[key] = lock
    return lock


class ConfigStore:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.path = Path(config_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = _get_lock(self.path)

    def read(self) -> Dict[str, Any]:
        with self._lock:
            return self._read_unlocked()

    def update(self, updater: Callable[[Dict[str, Any]], None]) -> Dict[str, Any]:
        with self._lock:
            config = self._read_unlocked()
            updater(config)
            self._write_unlocked(config)
            return config

    def write(self, config: Dict[str, Any]) -> None:
        with self._lock:
            self._write_unlocked(config)

    def _read_unlocked(self) -> Dict[str, Any]:
        if not self.path.exists():
            raise FileNotFoundError(f"Config file not found: {self.path}")
        with self.path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Invalid config format in {self.path}: expected a mapping at root")
        return config

    def _write_unlocked(self, config: Dict[str, Any]) -> None:
        temp_fd, temp_path = tempfile.mkstemp(
            prefix=f"{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as temp_file:
                yaml.safe_dump(config, temp_file, default_flow_style=False, sort_keys=False)
                temp_file.flush()
                os.fsync(temp_file.fileno())
            os.replace(temp_path, self.path)
        except Exception:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            raise
