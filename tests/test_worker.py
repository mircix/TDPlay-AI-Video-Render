"""Background jobs (``tdplay.worker``): a render runs in its own process and outlives the app window."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tdplay import engine, worker
from tdplay.engine import Segment
from test_engine import FAKE, assert_seamless, make_settings

ROOT = Path(__file__).resolve().parent.parent
FAKE_UPSCALE = [sys.executable, str(Path(__file__).parent / "fake_upscale.py")]
posix_only = pytest.mark.skipif(os.name == "nt", reason="POSIX signals")


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    monkeypatch.setenv("TDPLAY_LTX_CMD", shlex.join(FAKE))
    monkeypatch.setenv("TDPLAY_UPSCALE_CMD", shlex.join(FAKE_UPSCALE))
    monkeypatch.setenv("TDPLAY_NO_NOTIFY", "1")


def events_of(project_dir, proc=None, timeout=120, on_event=None) -> list[engine.Event]:
    events, deadline = [], time.time() + timeout
    for event in worker.follow(project_dir, proc, poll=0.1):
        assert time.time() < deadline, "the job took too long"
        if event is not None:
            events.append(event)
            if on_event:
                on_event(event)
    return events


def first_step(project_dir, proc) -> None:
    for event in worker.follow(project_dir, proc, poll=0.1):
        if event is not None and event.kind == "progress" and "step" in event.text:
            return


def test_a_render_runs_in_its_own_process(tmp_path):
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2), Segment("b", 1)])
    proc = worker.start(project.dir, ["render"])
    events = events_of(project.dir, proc)
    assert events[0].kind == "queued" and events[-1].kind == "done", events[-1].text
    assert any(e.kind == "preview" for e in events) and all(e.at > 0 for e in events)
    assert proc.wait(timeout=30) == 0
    assert not worker.is_running(project.dir)
    assert engine.Project.load(project.dir).status == "complete"
    assert_seamless(events[-1].path, 49 + 24)
    # A window opened later replays the whole job.
    assert [e.kind for e in events_of(project.dir)] == [e.kind for e in events]

    proc = worker.start(project.dir, ["upscale", "--size", "1080p"])
    events = events_of(project.dir, proc)
    assert events[-1].kind == "done" and events[-1].path.endswith("render_1080p.mp4")


def test_the_render_outlives_the_app_that_started_it(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.05")
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2), Segment("b", 2)])
    # The "app" starts the job and is then killed together with everything in its session, the way
    # closing a Terminal window ends what runs in it.
    app = subprocess.Popen(
        [sys.executable, "-c", f"import time; from tdplay import worker; worker.start({str(project.dir)!r}, "
                               f"['render']); time.sleep(60)"],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), start_new_session=os.name != "nt")
    deadline = time.time() + 60
    while not worker.is_running(project.dir):
        assert time.time() < deadline and app.poll() is None, "the worker didn't start"
        time.sleep(0.1)
    if os.name == "nt":
        app.kill()
    else:
        os.killpg(app.pid, signal.SIGKILL)
    app.wait(timeout=30)
    events = events_of(project.dir)
    assert events[-1].kind == "done", events[-1].text


def test_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.3")
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2), Segment("b", 2)])
    proc = worker.start(project.dir, ["render"])
    first_step(project.dir, proc)
    worker.request_stop(project.dir)
    events = events_of(project.dir, proc, timeout=60)
    assert events[-1].kind == "stopped"
    assert proc.wait(timeout=30) == 0
    assert engine.Project.load(project.dir).status == "stopped"
    assert not (project.dir / worker.STOP).exists()


@posix_only
def test_terminating_the_worker_stops_the_model_too(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.3")
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2)])
    proc = worker.start(project.dir, ["render"])
    first_step(project.dir, proc)
    proc.terminate()
    assert events_of(project.dir, proc, timeout=60)[-1].kind == "stopped"
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    assert str(project.dir) not in out, "the model process is still running"


@posix_only
def test_a_killed_worker_is_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.3")
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2)])
    proc = worker.start(project.dir, ["render"])
    first_step(project.dir, proc)
    proc.kill()
    last = events_of(project.dir, proc, timeout=60)[-1]
    assert last.kind == "error" and "ended unexpectedly" in last.text


def test_a_job_that_cannot_start_says_why(tmp_path):
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "project.json").write_text("{not json")
    proc = worker.start(broken, ["render"])
    last = events_of(broken, proc, timeout=60)[-1]
    assert last.kind == "error" and "JSONDecodeError" in last.text


def test_a_new_window_shows_the_running_job(tmp_path, monkeypatch):
    from tdplay import app

    monkeypatch.setenv("FAKE_LTX_SLEEP", "0.05")
    monkeypatch.setattr(app, "PROJECTS_DIR", tmp_path)
    monkeypatch.setitem(app._state, "job", None)  # this app didn't start it (it was reopened)
    project = engine.create_project(tmp_path, make_settings(), [Segment("a", 2), Segment("b", 1)])
    worker.start(project.dir, ["render"])
    while not worker.is_running(project.dir):
        time.sleep(0.1)
    assert app.active_job() == str(project.dir)
    shown = app.resume(str(project.dir), True, True, False)  # Resume on a running project just shows it
    assert app.BACKGROUND_NOTE in next(shown)[0]
    shown.close()
    updates = list(app.attach())
    assert "Done!" in updates[-1][0]
    assert any(app.BACKGROUND_NOTE in update[0] for update in updates)
    assert app.active_job() is None
    # Opened again after it finished: the window shows how it ended.
    assert "Done!" in list(app.attach())[-1][0]
