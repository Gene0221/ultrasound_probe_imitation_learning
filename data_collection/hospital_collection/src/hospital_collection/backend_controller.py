from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal

APP_ROOT = Path(__file__).resolve().parents[2]
DATA_COLLECTION_ROOT = APP_ROOT.parent
COLLECTION_RUNTIME_SRC = DATA_COLLECTION_ROOT / "collection_runtime" / "src"
if str(COLLECTION_RUNTIME_SRC) not in sys.path:
    sys.path.insert(0, str(COLLECTION_RUNTIME_SRC))

from collection_runtime.launcher import CollectionLauncher  # noqa: E402


class BackendController(QObject):
    """Direct wrapper around CollectionLauncher for UI integration.

    Replaces the old subprocess + JSON-file protocol.  The controller
    lives in the Qt main thread; the adapter subprocesses it spawns run
    independently and do not block the UI.
    """

    status_updated = Signal(dict)   # payload from CollectionLauncher._status_payload()
    log_message = Signal(str)       # status / error text for the UI log

    def __init__(self, config_path: str | Path, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._launcher: CollectionLauncher | None = None
        self._config_path = str(Path(config_path).resolve())

        self._timer = QTimer(self)
        self._timer.setInterval(300)
        self._timer.timeout.connect(self._broadcast_status)

        # Track last-seen messages per adapter so we only emit on change.
        self._last_messages: dict[str, str | None] = {}

    # ── Lifecycle ──────────────────────────────────────────────────────

    def initialize(self) -> tuple[bool, str]:
        """Create and initialise the launcher.

        Returns ``(success, message)``.  On success the controller begins
        broadcasting status via the ``status_updated`` signal.
        """
        try:
            self._launcher = CollectionLauncher.from_config_path(self._config_path)
            self._launcher.initialize()
            self._timer.start()
            return True, "OK"
        except Exception as exc:
            self._launcher = None
            return False, str(exc)

    def shutdown(self) -> None:
        """Stop the launcher and release resources."""
        self._timer.stop()
        if self._launcher is not None:
            self._launcher.stop()
            self._launcher = None

    # ── Session control ────────────────────────────────────────────────

    def start_or_resume(self) -> None:
        if self._launcher is not None:
            self._launcher.start_or_resume()

    def pause(self) -> None:
        if self._launcher is not None:
            self._launcher.pause()

    # ── Internal ───────────────────────────────────────────────────────

    def _broadcast_status(self) -> None:
        if self._launcher is not None:
            self.status_updated.emit(self._launcher._status_payload())
            for adapter in self._launcher.adapters:
                adapter.health_check()
                msg = adapter.status.message
                if msg and msg != self._last_messages.get(adapter.name):
                    self._last_messages[adapter.name] = msg
                    self.log_message.emit(f"[{adapter.name}] {msg}")
