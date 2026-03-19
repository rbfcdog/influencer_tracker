import asyncio
import os
from typing import Any

from TikTokApi import TikTokApi


class TikTokScraper:
    def __init__(self, ms_token: str | None = None, browser: str = "chromium") -> None:
        self.ms_token = ms_token or os.getenv("ms_token")
        self.browser = browser or os.getenv("TIKTOK_BROWSER", "chromium")

    async def _fetch_recent_videos_async(self, username: str, count: int) -> list[dict[str, Any]]:
        videos: list[dict[str, Any]] = []
        ms_tokens = [self.ms_token] if self.ms_token else None
        async with TikTokApi() as api:
            await api.create_sessions(
                ms_tokens=ms_tokens,
                num_sessions=1,
                sleep_after=3,
                browser=self.browser,
            )

            user = api.user(username)
            async for video in user.videos(count=count):
                raw = video.as_dict
                raw.setdefault(
                    "webVideoUrl",
                    f"https://www.tiktok.com/@{username}/video/{raw.get('id', '')}",
                )
                videos.append(raw)

        return videos

    def fetch_recent_videos(self, username: str, count: int = 50) -> list[dict[str, Any]]:
        if not username.strip():
            raise ValueError("username is required")
        if count <= 0:
            raise ValueError("count must be greater than 0")
        return asyncio.run(self._fetch_recent_videos_async(username=username, count=count))
