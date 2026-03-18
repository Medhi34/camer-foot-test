"""Video handling module for asynchronous frame extraction.

This module provides a robust OpenCV-based video reader with:
- asynchronous frame extraction using a producer thread + asyncio consumer
- optional preprocessing for heterogeneous-quality videos:
  - CLAHE contrast enhancement
  - denoising (Gaussian or fastNlMeans)

Architecture notes:
- `VideoHandler` is infrastructure-level code (io/ layer).
- Business logic must remain in vision/analytics modules.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import AsyncIterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PreprocessConfig:
    """Configuration for optional frame preprocessing.

    Attributes:
        enabled: Enable/disable preprocessing.
        adjust_contrast: Apply CLAHE contrast enhancement on L channel in LAB.
        denoise: Apply denoising filter.
        denoise_method: Denoising method: "gaussian" or "nlmeans".
        gaussian_kernel: Kernel size for Gaussian blur (must be odd).
        nlmeans_h: Strength parameter for fastNlMeansDenoisingColored.
        clahe_clip_limit: CLAHE clip limit.
        clahe_tile_grid_size: CLAHE tile grid size.
    """

    enabled: bool = False
    adjust_contrast: bool = True
    denoise: bool = False
    denoise_method: str = "gaussian"
    gaussian_kernel: int = 3
    nlmeans_h: float = 5.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: tuple[int, int] = (8, 8)


@dataclass(slots=True)
class FramePacket:
    """Container for a decoded frame.

    Attributes:
        frame_index: Zero-based frame index.
        timestamp_sec: Timestamp in seconds.
        frame: BGR image (OpenCV format).
    """

    frame_index: int
    timestamp_sec: float
    frame: np.ndarray


class VideoHandler:
    """Asynchronous video loader and frame extractor based on OpenCV.

    The class uses:
    - one producer thread reading frames from cv2.VideoCapture
    - one bounded queue for backpressure
    - an async generator interface for downstream consumers

    Example:
        >>> async for pkt in VideoHandler("match.mp4").iter_frames_async():
        ...     # send pkt.frame to vision pipeline
        ...     pass
    """

    def __init__(
        self,
        video_path: str,
        *,
        queue_size: int = 128,
        preprocess: PreprocessConfig | None = None,
        loop_sleep_sec: float = 0.001,
    ) -> None:
        """Initialize the video handler.

        Args:
            video_path: Path to input video.
            queue_size: Max number of frames buffered in memory.
            preprocess: Optional preprocessing configuration.
            loop_sleep_sec: Small async sleep used while waiting for queue data.

        Raises:
            ValueError: If queue_size <= 0.
        """
        if queue_size <= 0:
            raise ValueError("queue_size must be > 0")

        self.video_path = video_path
        self.queue_size = queue_size
        self.preprocess = preprocess or PreprocessConfig()
        self.loop_sleep_sec = loop_sleep_sec

        self._capture: cv2.VideoCapture | None = None
        self._fps: float = 0.0
        self._frame_count: int = 0
        self._width: int = 0
        self._height: int = 0

        self._queue: Queue[FramePacket | None] = Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._producer_thread: threading.Thread | None = None

    @property
    def fps(self) -> float:
        """Video FPS."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames if available from container metadata."""
        return self._frame_count

    @property
    def resolution(self) -> tuple[int, int]:
        """Video resolution as (width, height)."""
        return self._width, self._height

    def open(self) -> None:
        """Open the video file and read metadata.

        Raises:
            FileNotFoundError: If video cannot be opened.
        """
        logger.info("Opening video: %s", self.video_path)
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Unable to open video: {self.video_path}")

        self._capture = cap
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        logger.info(
            "Video metadata | fps=%.3f frames=%d resolution=%dx%d",
            self._fps,
            self._frame_count,
            self._width,
            self._height,
        )

    def close(self) -> None:
        """Release resources and stop producer if running."""
        logger.info("Closing video handler.")
        self.stop()

        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def start(self) -> None:
        """Start producer thread for asynchronous frame extraction.

        Raises:
            RuntimeError: If video is not opened.
        """
        if self._capture is None:
            raise RuntimeError("Video is not opened. Call open() first.")

        if self._producer_thread and self._producer_thread.is_alive():
            logger.debug("Producer already running.")
            return

        self._stop_event.clear()
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            name="video-frame-producer",
            daemon=True,
        )
        self._producer_thread.start()
        logger.info("Producer thread started.")

    def stop(self) -> None:
        """Signal producer stop and wait for clean shutdown."""
        self._stop_event.set()

        if self._producer_thread and self._producer_thread.is_alive():
            self._producer_thread.join(timeout=2.0)
            logger.info("Producer thread stopped.")

        self._producer_thread = None

    async def iter_frames_async(self) -> AsyncIterator[FramePacket]:
        """Asynchronously yield frame packets from the internal queue.

        Yields:
            FramePacket: Frame index, timestamp and image.

        Notes:
            A `None` sentinel in queue indicates stream completion.
        """
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                await asyncio.sleep(self.loop_sleep_sec)
                continue

            if item is None:
                logger.info("End-of-stream received from producer.")
                break

            yield item

    def _producer_loop(self) -> None:
        """Producer thread loop reading and preprocessing frames."""
        assert self._capture is not None, "Capture must be initialized before producer starts."

        frame_index = 0
        logger.debug("Entering producer loop.")

        try:
            while not self._stop_event.is_set():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    logger.info("No more frames or read failure at frame %d.", frame_index)
                    break

                if self.preprocess.enabled:
                    frame = self._preprocess_frame(frame)

                timestamp_sec = self._compute_timestamp_sec(frame_index)
                packet = FramePacket(
                    frame_index=frame_index,
                    timestamp_sec=timestamp_sec,
                    frame=frame,
                )

                # Backpressure-safe put
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(packet, timeout=0.05)
                        break
                    except Full:
                        continue

                frame_index += 1

        except Exception:
            logger.exception("Unhandled error in video producer loop.")
        finally:
            # Signal end-of-stream
            while True:
                try:
                    self._queue.put(None, timeout=0.05)
                    break
                except Full:
                    if self._stop_event.is_set():
                        continue
            logger.debug("Producer loop exited cleanly.")

    def _compute_timestamp_sec(self, frame_index: int) -> float:
        """Compute frame timestamp in seconds from frame index.

        Args:
            frame_index: Zero-based frame index.

        Returns:
            Timestamp in seconds.
        """
        if self._fps > 0:
            return frame_index / self._fps
        return 0.0

    def _preprocess_frame(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply optional preprocessing to improve frame quality.

        Steps:
            1) Contrast enhancement (CLAHE on LAB L-channel), optional.
            2) Denoising, optional.

        Args:
            frame_bgr: Input frame in BGR format.

        Returns:
            Processed frame in BGR format.
        """
        processed = frame_bgr

        if self.preprocess.adjust_contrast:
            processed = self._apply_clahe(processed)

        if self.preprocess.denoise:
            processed = self._apply_denoise(processed)

        return processed

    def _apply_clahe(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply CLAHE in LAB color space for robust local contrast boost."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        clahe = cv2.createCLAHE(
            clipLimit=self.preprocess.clahe_clip_limit,
            tileGridSize=self.preprocess.clahe_tile_grid_size,
        )
        l_enhanced = clahe.apply(l)

        merged = cv2.merge((l_enhanced, a, b))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _apply_denoise(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply denoising according to configured method."""
        method = self.preprocess.denoise_method.lower()

        if method == "gaussian":
            k = self.preprocess.gaussian_kernel
            if k % 2 == 0:
                k += 1
            return cv2.GaussianBlur(frame_bgr, (k, k), sigmaX=0.0)

        if method == "nlmeans":
            h = float(self.preprocess.nlmeans_h)
            return cv2.fastNlMeansDenoisingColored(
                frame_bgr,
                None,
                h,
                h,
                7,
                21,
            )

        logger.warning("Unknown denoise method '%s'. Returning frame unchanged.", method)
        return frame_bgr