"""Stand-in for ``python -m tdplay.upscale``: plain ffmpeg scaling, same arguments and progress lines."""

import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("output")
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--quality", choices=["fast", "best"], required=True)
args = parser.parse_args()
for i in range(1, 6):
    print(f"PROGRESS {i}/5", flush=True)
vf = f"scale={args.width}:{args.height}:force_original_aspect_ratio=increase,crop={args.width}:{args.height}"
subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", args.input, "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "copy", args.output], check=True)
