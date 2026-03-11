"""Tests for brain/scheduler.py (component-level, no blocking run())."""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from unittest.mock import MagicMock, patch

import pytest

from watchdog.events import FileCreatedEvent, FileMovedEvent

from brain.scheduler import InboxEventHandler, Scheduler


# ---------------------------------------------------------------------------
# InboxEventHandler
# ---------------------------------------------------------------------------

@pytest.fixture
def queue():
    return Queue()


@pytest.fixture
def handler(queue):
    return InboxEventHandler(extensions={".pdf", ".png"}, queue=queue)


def test_handler_accepts_matching_extension(handler, queue, tmp_path):
    path = tmp_path / "doc.pdf"
    path.touch()
    event = FileCreatedEvent(str(path))
    handler.on_created(event)
    assert not queue.empty()
    assert queue.get() == path


def test_handler_rejects_non_matching_extension(handler, queue, tmp_path):
    path = tmp_path / "doc.docx"
    path.touch()
    event = FileCreatedEvent(str(path))
    handler.on_created(event)
    assert queue.empty()


def test_handler_ignores_directories(handler, queue, tmp_path):
    d = tmp_path / "subdir"
    d.mkdir()
    event = FileCreatedEvent(str(d))
    event.is_directory = True
    handler.on_created(event)
    assert queue.empty()


def test_handler_on_moved_accepted(handler, queue, tmp_path):
    path = tmp_path / "moved.pdf"
    path.touch()
    event = MagicMock(spec=FileMovedEvent)
    event.is_directory = False
    event.dest_path = str(path)
    handler.on_moved(event)
    assert not queue.empty()


def test_handler_on_moved_rejected(handler, queue, tmp_path):
    path = tmp_path / "moved.txt"
    path.touch()
    event = MagicMock(spec=FileMovedEvent)
    event.is_directory = False
    event.dest_path = str(path)
    handler.on_moved(event)
    assert queue.empty()


# ---------------------------------------------------------------------------
# Scheduler._flush_batch
# ---------------------------------------------------------------------------

def test_flush_batch_calls_pipeline_for_each_file(app_config):
    mock_pipeline = MagicMock()
    scheduler = Scheduler(app_config, mock_pipeline)

    files = [Path(f"/fake/doc{i}.pdf") for i in range(3)]
    scheduler._batch_accumulator.extend(files)

    scheduler._flush_batch()

    assert mock_pipeline.process.call_count == 3
    called_paths = {call.args[0] for call in mock_pipeline.process.call_args_list}
    assert called_paths == set(files)


def test_flush_batch_clears_accumulator(app_config):
    mock_pipeline = MagicMock()
    scheduler = Scheduler(app_config, mock_pipeline)
    scheduler._batch_accumulator.extend([Path("/fake/a.pdf"), Path("/fake/b.pdf")])

    scheduler._flush_batch()

    assert scheduler._batch_accumulator == []


def test_flush_batch_noop_when_empty(app_config):
    mock_pipeline = MagicMock()
    scheduler = Scheduler(app_config, mock_pipeline)
    scheduler._flush_batch()
    mock_pipeline.process.assert_not_called()


# ---------------------------------------------------------------------------
# Scheduler._setup_batch_schedule
# ---------------------------------------------------------------------------

def test_setup_batch_daily(app_config):
    app_config.scheduler.mode = "batch"
    app_config.scheduler.batch_schedule = "daily"
    app_config.scheduler.batch_time = "03:00"

    mock_pipeline = MagicMock()
    scheduler = Scheduler(app_config, mock_pipeline)

    import schedule as sched
    sched.clear()

    scheduler._setup_batch_schedule()
    jobs = sched.get_jobs()
    assert len(jobs) == 1

    sched.clear()  # cleanup


def test_setup_batch_weekly(app_config):
    app_config.scheduler.mode = "batch"
    app_config.scheduler.batch_schedule = "weekly"
    app_config.scheduler.batch_day = "monday"
    app_config.scheduler.batch_time = "10:00"

    mock_pipeline = MagicMock()
    scheduler = Scheduler(app_config, mock_pipeline)

    import schedule as sched
    sched.clear()

    scheduler._setup_batch_schedule()
    jobs = sched.get_jobs()
    assert len(jobs) == 1

    sched.clear()  # cleanup
