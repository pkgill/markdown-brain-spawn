"""Scheduler: file watcher + realtime / batch dispatch.

Realtime mode: every detected file is processed immediately.
Batch mode:    files accumulate in memory; the `schedule` library fires
               _flush_batch() at the configured time.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from queue import Empty, Queue

import schedule

from watchdog.events import FileCreatedEvent, FileMovedEvent, FileSystemEventHandler
from watchdog.observers import Observer

from brain.config import AppConfig
from brain.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Watchdog event handler
# ---------------------------------------------------------------------------

class InboxEventHandler(FileSystemEventHandler):
    """Detects file creation and moves into the watched folder.

    Only forwards files whose extensions match the configured list.
    Vault subdirectory events are excluded because the observer runs
    non-recursively, so files moved *into* the vault won't re-trigger.
    """

    def __init__(self, extensions: set[str], queue: "Queue[Path]"):
        super().__init__()
        self._extensions = extensions
        self._queue = queue
        self._seen: dict[str, float] = {}    # path → timestamp of last event
        self._seen_lock = threading.Lock()
        self._DEDUP_WINDOW = 5  # seconds — ignore duplicate events within this window

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._handle(Path(event.src_path))

    def on_moved(self, event: FileMovedEvent) -> None:
        # A file copied or moved INTO the watch folder
        if not event.is_directory:
            self._handle(Path(event.dest_path))

    def _handle(self, path: Path) -> None:
        if path.suffix.lower() in self._extensions:
            now = time.monotonic()
            key = str(path)
            with self._seen_lock:
                last = self._seen.get(key, 0)
                if now - last < self._DEDUP_WINDOW:
                    logger.debug("Duplicate event suppressed: %s", path.name)
                    return
                self._seen[key] = now
            logger.info("Detected: %s", path.name)
            self._queue.put(path)
        else:
            logger.debug("Ignored (wrong extension): %s", path.name)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class Scheduler:
    """Runs the watchdog observer and dispatches files to the pipeline."""

    def __init__(self, config: AppConfig, pipeline: Pipeline):
        self._config = config
        self._pipeline = pipeline
        self._queue: Queue[Path] = Queue()
        self._batch_accumulator: list[Path] = []
        self._batch_lock = threading.Lock()
        self._observer = Observer()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Block until KeyboardInterrupt. Sets up watcher and optional schedule."""
        self._start_watcher()

        if self._config.scheduler.mode == "batch":
            self._setup_batch_schedule()
            logger.info(
                "Batch mode — files will be processed at %s (%s)",
                self._config.scheduler.batch_time,
                self._config.scheduler.batch_schedule,
            )
        else:
            logger.info("Realtime mode — files will be processed immediately.")

        logger.info(
            "Watching: %s (extensions: %s)",
            self._config.watch.inbox_path,
            ", ".join(sorted(self._config.watch.extensions)),
        )

        try:
            self._event_loop()
        except KeyboardInterrupt:
            logger.info("Shutdown requested.")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Internal setup
    # ------------------------------------------------------------------

    def _start_watcher(self) -> None:
        handler = InboxEventHandler(
            extensions=set(self._config.watch.extensions),
            queue=self._queue,
        )
        watch_path = str(self._config.watch.inbox_path)
        self._observer.schedule(handler, watch_path, recursive=False)
        self._observer.start()
        logger.debug("Watchdog observer started on: %s", watch_path)

    def _setup_batch_schedule(self) -> None:
        sched_cfg = self._config.scheduler
        t = sched_cfg.batch_time

        if sched_cfg.batch_schedule == "daily":
            schedule.every().day.at(t).do(self._flush_batch)
            logger.debug("Batch schedule: daily at %s", t)
        else:  # weekly
            day_fn = getattr(schedule.every(), sched_cfg.batch_day)
            day_fn.at(t).do(self._flush_batch)
            logger.debug("Batch schedule: every %s at %s", sched_cfg.batch_day, t)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------

    def _event_loop(self) -> None:
        while True:
            schedule.run_pending()

            try:
                file_path = self._queue.get(timeout=1)
            except Empty:
                continue

            if self._config.scheduler.mode == "realtime":
                self._process_one(file_path)
            else:
                with self._batch_lock:
                    self._batch_accumulator.append(file_path)
                logger.info("Queued for batch processing: %s", file_path.name)

    # ------------------------------------------------------------------
    # Processing helpers
    # ------------------------------------------------------------------

    def _process_one(self, file_path: Path) -> None:
        if not file_path.exists():
            logger.warning("Skipping %s — file no longer exists (already processed?).", file_path.name)
            return
        try:
            self._pipeline.process(file_path)
        except Exception as exc:
            logger.exception("Unhandled error processing %s: %s", file_path.name, exc)

    def _flush_batch(self) -> None:
        with self._batch_lock:
            files = list(self._batch_accumulator)
            self._batch_accumulator.clear()

        if not files:
            logger.info("Batch flush: no files queued.")
            return

        logger.info("Batch flush: processing %d file(s).", len(files))
        for path in files:
            self._process_one(path)
        logger.info("Batch flush complete.")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Stopping file watcher...")
        self._observer.stop()
        self._observer.join()
        logger.info("Shutdown complete.")
