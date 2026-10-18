"""Stand-in for ``python -m tdplay.creator JOB.json``: the real process (``tdplay.creator.run``) with a stand-in
model that paints one color per seed (an outfit change keeps the photo in its top-left quarter).

It saves the request it got as ``seen.json`` next to the pictures. ``FAKE_CREATOR_FAIL=<n>`` makes picture n
run out of graphics memory, ``FAKE_CREATOR_CRASH=1`` ends the process without a last event, and
``FAKE_CREATOR_SLEEP`` (seconds) slows every picture down.
"""

import json
import os
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tdplay import creator  # noqa: E402

request = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
Path(request["out_dir"], "seen.json").write_text(json.dumps(request), encoding="utf-8")


def maker(mode):
    print("Loading the stand-in model", flush=True)  # a log line, not an event
    made = []

    def make(prompt, seed, width, height, photo):
        made.append(seed)
        if os.environ.get("FAKE_CREATOR_FAIL") == str(len(made)):
            raise RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB")
        if os.environ.get("FAKE_CREATOR_CRASH"):
            os._exit(3)
        time.sleep(float(os.environ.get("FAKE_CREATOR_SLEEP", "0")))
        picture = Image.new("RGB", (width, height), (seed * 37 % 256, seed * 91 % 256, 200 if photo else 50))
        if photo:
            with Image.open(photo) as source:
                picture.paste(source.resize((width // 2, height // 2)))
        return picture

    return make


sys.exit(creator.run(request, maker=maker))
