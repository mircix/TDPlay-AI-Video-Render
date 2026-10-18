"""Background jobs: a render or an upscale runs in its own process, so it keeps going without the window.

The app starts ``python -m tdplay.worker PROJECT_DIR render`` (or ``upscale --size 4k``) detached
from itself: closing the window, quitting the app, closing Terminal or losing the browser
connection doesn't stop it. The worker runs the job with :class:`tdplay.engine.Runner`, appends
every event to ``events.jsonl`` in the project folder and holds a lock on ``worker.lock`` while it
runs. Any window, also one opened later, follows the job by reading that file (:func:`follow`).
Stop is a ``stop`` file the worker checks for. The worker keeps the computer awake while it works
and shows a notification when the job ends.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from tdplay import engine, media

EVENTS = "events.jsonl"
LOCK = "worker.lock"
STOP = "stop"
LOG = "worker.log"
#: Event kinds that end a job.
FINAL = {"done", "error", "stopped"}
#: A job whose worker can't be seen yet (it is still starting) counts as alive this long.
START_GRACE_SECONDS = 30


# -- the lock that says "a worker is running on this project" ----------------------------------------


def _try_lock(path: Path):
    """Open ``path`` and take an exclusive lock without waiting; ``None`` if another process holds it."""
    f = open(path, "a+b")  # noqa: SIM115 - the open file is the lock
    try:
        if os.name == "nt":
            import msvcrt

            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        return None
    return f


def _release(f) -> None:
    if os.name == "nt":
        import msvcrt

        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    f.close()


def is_running(project_dir: str | Path) -> bool:
    """Whether a worker is running on this project right now (its lock is held)."""
    path = Path(project_dir) / LOCK
    if not path.exists():
        return False
    f = _try_lock(path)
    if f is None:
        return True
    _release(f)
    return False


# -- starting, stopping and following a job (app side) ----------------------------------------------


def start(project_dir: str | Path, job: list[str]) -> subprocess.Popen:
    """Start a worker for ``job`` (``["render"]`` or ``["upscale", "--size", "4k", ...]``) on its own."""
    d = Path(project_dir)
    (d / STOP).unlink(missing_ok=True)
    # Written before the worker exists, so a window that follows the job never reads the previous one's events.
    (d / EVENTS).write_text(_dump(engine.Event("queued", "Starting…")), encoding="utf-8")
    python = Path(sys.executable)
    if python.name.lower() == "pythonw.exe" and python.with_name("python.exe").exists():
        python = python.with_name("python.exe")  # its output goes to worker.log; no console window either way
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
               PYTHONPATH=os.pathsep.join(filter(None, [str(engine.APP_ROOT), os.environ.get("PYTHONPATH")])))
    cmd = [str(python), "-m", "tdplay.worker", str(d), *job]
    with open(d / LOG, "ab") as log:
        log.write(f"\n--- {time.ctime()}: {' '.join(job)} ---\n".encode())
        log.flush()
        kwargs = dict(cwd=d, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if os.name != "nt":
            return subprocess.Popen(cmd, start_new_session=True, **kwargs)  # no SIGHUP from a closing Terminal
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
        try:  # leave the app's job object too, if it has one, so quitting the app can't end the render
            return subprocess.Popen(cmd, creationflags=flags | subprocess.CREATE_BREAKAWAY_FROM_JOB, **kwargs)
        except OSError:
            return subprocess.Popen(cmd, creationflags=flags, **kwargs)


def request_stop(project_dir: str | Path) -> None:
    (Path(project_dir) / STOP).touch()


def follow(project_dir: str | Path, proc: subprocess.Popen | None = None,
           poll: float = 0.3) -> Iterator[engine.Event | None]:
    """The job's events, from its first, as they are written. ``None`` while nothing new has happened.

    Ends after the job's final event. If the worker is gone without writing one (it crashed or was
    killed), ends with an ``error`` event. ``proc`` is the worker, if this process started it.
    """
    d = Path(project_dir)
    path = d / EVENTS
    pos, buffer = 0, b""
    while True:
        # Look at the worker before reading: if it was already gone, everything it wrote is in the file.
        alive = (proc is not None and proc.poll() is None) or is_running(d)
        try:
            with open(path, "rb") as f:
                f.seek(pos)
                chunk = f.read()
        except FileNotFoundError:
            chunk = b""
        pos += len(chunk)
        *lines, buffer = (buffer + chunk).split(b"\n")
        for line in lines:
            try:
                event = engine.Event(**json.loads(line))
            except (ValueError, TypeError):
                continue
            yield event
            if event.kind in FINAL:
                return
        if not alive and not lines:
            try:
                starting = proc is None and time.time() - path.stat().st_mtime < START_GRACE_SECONDS
            except OSError:
                starting = False
            if not starting:
                yield engine.Event("error", "The background job ended unexpectedly. Press Resume to try again."
                                            + _log_tail(d))
                return
        if not lines:
            yield None
        time.sleep(poll)


def _log_tail(d: Path, n: int = 15) -> str:
    try:
        lines = (d / LOG).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    tail = [line for line in lines if line.strip()][-n:]
    return "\n\n" + "\n".join(tail) if tail else ""


def _dump(event: engine.Event) -> str:
    data = asdict(event)
    data["at"] = data["at"] or time.time()
    return json.dumps(data, ensure_ascii=False) + "\n"


# -- the worker process ----------------------------------------------------------------------------------


def _keep_awake() -> None:
    """Keep the computer from idle-sleeping while this process runs."""
    if os.name == "nt":
        import ctypes

        es_continuous, es_system_required = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    elif sys.platform == "darwin":
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(os.getpid())], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass


_TOAST = r"""
[Windows.UI.Notifications.ToastNotificationManager, Windows.UI.Notifications, ContentType = WindowsRuntime] | Out-Null
$xml = [Windows.UI.Notifications.ToastNotificationManager]::GetTemplateContent([Windows.UI.Notifications.ToastTemplateType]::ToastText02)
$texts = $xml.GetElementsByTagName('text')
$texts.Item(0).AppendChild($xml.CreateTextNode($env:TDPLAY_TOAST_TITLE)) | Out-Null
$texts.Item(1).AppendChild($xml.CreateTextNode($env:TDPLAY_TOAST_TEXT)) | Out-Null
$app = '{1AC14E77-02E7-4E5D-B744-2EB1AE5198B7}\WindowsPowerShell\v1.0\powershell.exe'
[Windows.UI.Notifications.ToastNotificationManager]::CreateToastNotifier($app).Show([Windows.UI.Notifications.ToastNotification]::new($xml))
"""


def notify(title: str, text: str) -> None:
    """A system notification (best effort: the job is done either way)."""
    if os.environ.get("TDPLAY_NO_NOTIFY"):
        return
    try:
        if sys.platform == "darwin":
            subprocess.run(["osascript", "-e", "on run argv", "-e",
                            "display notification (item 2 of argv) with title (item 1 of argv)", "-e", "end run",
                            title, text], capture_output=True, timeout=15)
        elif os.name == "nt":
            env = dict(os.environ, TDPLAY_TOAST_TITLE=title, TDPLAY_TOAST_TEXT=text)
            subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", _TOAST], env=env,
                           capture_output=True, timeout=30, **media.NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        pass


def run(project_dir: str | Path, kind: str, size: str = "1080p", quality: str = "fast") -> int:
    d = Path(project_dir)
    lock = _try_lock(d / LOCK)
    if lock is None:
        print("Another job is already running on this project.", flush=True)
        return 1
    with open(d / EVENTS, "a", encoding="utf-8") as out:
        try:
            from tdplay.engine_torch import TorchRunner

            project = engine.Project.load(d)
            runner = TorchRunner(project) if project.settings.backend == "torch" else engine.Runner(project)
            finished = threading.Event()

            def stop() -> None:
                if not finished.is_set():
                    runner.cancel()

            def watch_stop_file() -> None:
                while not finished.wait(0.5):
                    if (d / STOP).exists():
                        stop()
                        return

            threading.Thread(target=watch_stop_file, daemon=True).start()
            if hasattr(signal, "SIGTERM"):  # `kill` ends the model process too, not just this one
                signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=stop, daemon=True).start())
            _keep_awake()
            last = None
            try:
                for event in (runner.run() if kind == "render" else runner.upscale(size, quality)):
                    out.write(_dump(event))
                    out.flush()
                    last = event
            finally:
                finished.set()
        except Exception as exc:  # the job couldn't even start (e.g. a broken project.json)
            out.write(_dump(engine.Event("error", f"{type(exc).__name__}: {exc}")))
            raise
        finally:
            (d / STOP).unlink(missing_ok=True)
            _release(lock)
    if last is not None and last.kind in ("done", "error"):
        what = "Your video is ready" if last.kind == "done" else "Your video stopped with an error"
        notify("TDPlay AI Video Render", f"{what}: {project.title[:60]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tdplay.worker", description="Run a render or upscale job.")
    parser.add_argument("project", type=Path)
    parser.add_argument("kind", choices=["render", "upscale"])
    parser.add_argument("--size", default="1080p", choices=["1080p", "4k"])
    parser.add_argument("--quality", default="fast", choices=["fast", "best"])
    args = parser.parse_args(argv)
    return run(args.project, args.kind, args.size, args.quality)


if __name__ == "__main__":
    sys.exit(main())
