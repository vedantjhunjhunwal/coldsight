"""RTSP / file reader that always hands out the newest frame (stale frames are dropped)."""
from __future__ import annotations

import os
import threading
import time

import cv2


class LatestFrameReader(threading.Thread):
    def __init__(self, url: str, max_backoff: float = 8.0, listen: bool = False):
        """listen=False: pull from a camera / RTSP server (production).
        listen=True: act as the RTSP server and wait for a camera to publish (push mode)."""
        super().__init__(daemon=True)
        self.url, self.max_backoff, self.listen = url, max_backoff, listen
        self._lock = threading.Lock()
        self._frame, self._ts, self._idx, self._served = None, 0.0, -1, -1
        self.grabbed = self.dropped = self.reconnects = 0
        self.running = True

    def run(self):
        backoff = 0.5
        while self.running:
            opts = "rtsp_transport;tcp" + ("|rtsp_flags;listen|listen_timeout;60" if self.listen else "|stimeout;5000000")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                self.reconnects += 1
                time.sleep(backoff)
                backoff = min(backoff * 2, self.max_backoff)
                continue
            backoff = 0.5
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                with self._lock:
                    if self._idx > self._served:
                        self.dropped += 1
                    self._frame, self._ts = frame, time.time()
                    self._idx += 1
                    self.grabbed += 1
            cap.release()
            if self.running:
                self.reconnects += 1
                time.sleep(backoff)

    def read(self, timeout=2.0):
        """Return (idx, capture_time, frame) for the newest unseen frame, or None on timeout."""
        t_end = time.time() + timeout
        while time.time() < t_end:
            with self._lock:
                if self._idx > self._served and self._frame is not None:
                    self._served = self._idx
                    return self._idx, self._ts, self._frame
            time.sleep(0.002)
        return None

    def stop(self):
        self.running = False
