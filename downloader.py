from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import requests


class VideoDownloader:
    def __init__(
        self,
        download_dir: str = "downloads",
        clip_dir: str = "clips",
        clip_seconds: int = 10,
        ffmpeg_bin: str = "ffmpeg",
        request_timeout: int = 90,
    ) -> None:
        self.download_dir = Path(download_dir)
        self.clip_dir = Path(clip_dir)
        self.clip_seconds = clip_seconds
        self.ffmpeg_bin = ffmpeg_bin
        self.request_timeout = request_timeout

        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.clip_dir.mkdir(parents=True, exist_ok=True)

    def _extract_video_url(self, video_payload: dict[str, Any]) -> str:
        video_data_raw = video_payload.get("video")
        video_data: dict[str, Any] = video_data_raw if isinstance(video_data_raw, dict) else {}

        candidates = [
            video_data.get("downloadAddr"),
            video_data.get("playAddr"),
            video_payload.get("downloadAddr"),
            video_payload.get("playAddr"),
        ]

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value

        raise ValueError("No downloadable video URL found in payload")

    def download_video(self, video_payload: dict[str, Any], username: str) -> Path:
        video_id = str(video_payload.get("id", "")).strip()
        if not video_id:
            raise ValueError("Missing video id in payload")

        video_url = self._extract_video_url(video_payload)
        target_file = self.download_dir / f"{username}_{video_id}.mp4"

        with requests.get(video_url, stream=True, timeout=self.request_timeout) as response:
            response.raise_for_status()
            with target_file.open("wb") as out_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        out_file.write(chunk)

        return target_file

    def extract_clip(self, input_file: Path) -> Path:
        clip_file = self.clip_dir / f"{input_file.stem}_clip.mp4"

        command = [
            self.ffmpeg_bin,
            "-y",
            "-i",
            str(input_file),
            "-t",
            str(self.clip_seconds),
            "-c",
            "copy",
            str(clip_file),
        ]
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed for {input_file.name}: {process.stderr.strip() or process.stdout.strip()}"
            )

        return clip_file

    def download_and_clip(self, video_payload: dict[str, Any], username: str) -> tuple[Path, Path]:
        downloaded = self.download_video(video_payload=video_payload, username=username)
        clip = self.extract_clip(downloaded)
        return downloaded, clip
