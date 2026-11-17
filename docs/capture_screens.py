"""Capture app screenshots for the documentation (needs Chrome and a running app on :8501).

Run: python docs/capture_screens.py
Uses Chrome's DevTools protocol through `websockets` (installed with Streamlit), so no extra dependency.
Waits in real time, because Streamlit renders over a websocket after the page load event.
"""
from __future__ import annotations

import base64
import json
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

from websockets.sync.client import connect

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE = "http://localhost:8501/"
OUT = Path(__file__).resolve().parent / "images"
PORT = 9333
SHOTS = [  # (file, path, height)
    ("app_overview", "", 1320),
    ("app_alerts", "alerts", 1080),
    ("app_investigate", "investigate", 1300),
    ("app_check", "check?example=fraud", 1420),
    ("app_model", "model", 1000),
    ("app_live", "live", 900),
]


def capture(ws_url: str, url: str, height: int, wait: float) -> bytes:
    with connect(ws_url, max_size=64 * 1024 * 1024) as ws:
        seq = 0

        def call(method, **params):
            nonlocal seq
            seq += 1
            ws.send(json.dumps({"id": seq, "method": method, "params": params}))
            while True:
                msg = json.loads(ws.recv())
                if msg.get("id") == seq:
                    return msg.get("result", {})

        call("Emulation.setDeviceMetricsOverride", width=1440, height=height, deviceScaleFactor=1, mobile=False)
        call("Page.navigate", url=url)
        time.sleep(wait)  # shortcut: fixed wait for Streamlit to finish rendering; raise it on a slow machine
        return base64.b64decode(call("Page.captureScreenshot", format="png")["data"])


def main(wait: float = 14.0) -> None:
    profile = tempfile.mkdtemp(prefix="fg-chrome-")
    proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", "--no-first-run",
                             f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}", "about:blank"])
    try:
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                break
            except OSError:
                time.sleep(0.2)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        for name, path, height in SHOTS:
            png = capture(ws_url, BASE + path, height, wait)
            (OUT / f"{name}.png").write_bytes(png)
            print("wrote", name)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
