import asyncio
import logging

from io.video_handler import PreprocessConfig, VideoHandler

logging.basicConfig(level=logging.INFO)


async def main() -> None:
    preprocess_cfg = PreprocessConfig(
        enabled=True,
        adjust_contrast=True,
        denoise=True,
        denoise_method="gaussian",
        gaussian_kernel=3,
    )

    vh = VideoHandler(
        "data/raw/match_cameroon.mp4",
        queue_size=96,
        preprocess=preprocess_cfg,
    )

    vh.open()
    vh.start()

    async for packet in vh.iter_frames_async():
        # ICI: envoyer packet.frame vers vision/detection.py
        # ex: detections = detector.infer(packet.frame)
        pass

    vh.close()


if __name__ == "__main__":
    asyncio.run(main())