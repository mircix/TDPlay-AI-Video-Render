"""Stand-in for pywebview: records what tdplay.mac_app asks for, checks the page loads, then "closes"."""

import json
import os
import urllib.request

settings = {}
_windows = []


class _Event(list):
    def __iadd__(self, handler):
        self.append(handler)
        return self


class _Events:
    def __init__(self):
        self.closing = _Event()


class Window:
    def __init__(self, title, url, **options):
        self.title, self.url, self.options = title, url, options
        self.events = _Events()
        self.confirm_close = False


def create_window(title, url, **options):
    window = Window(title, url, **options)
    _windows.append(window)
    return window


def start(**options):
    window = _windows[0]
    status = urllib.request.urlopen(window.url).status
    for handler in window.events.closing:
        handler()
    record = {
        "title": window.title, "url": window.url, "status": status, "confirm_close": window.confirm_close,
        "icon": options.get("icon"), "icon_exists": os.path.exists(options.get("icon") or ""),
        "downloads": settings.get("ALLOW_DOWNLOADS"), "quit_message": window.options["localization"],
    }
    with open(os.environ["STUB_WEBVIEW_RECORD"], "w") as f:
        json.dump(record, f)
