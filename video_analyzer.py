from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar
from dotenv import load_dotenv

from analyzer import GeminiVideoAnalyzer, VIRALITY_PROMPT
from db import PostgresDB
from downloader import VideoDownloader
from scraper import TikTokScraper

T = TypeVar("T")

load_dotenv()


def _retry(operation_name: str, func: Callable[[], T], retries: int, delay_seconds: float) -> T:
	last_error: Exception | None = None
	for attempt in range(1, retries + 1):
		try:
			return func()
		except Exception as exc:  # pragma: no cover - runtime/network dependent
			last_error = exc
			print(f"[{operation_name}] attempt {attempt}/{retries} failed: {exc}")
			if attempt < retries:
				time.sleep(delay_seconds)

	if last_error is None:
		raise RuntimeError(f"{operation_name} failed without explicit error")
	raise last_error


def _save_raw_response(raw_dir: Path, username: str, video_id: str, raw_text: str) -> Path:
	raw_dir.mkdir(parents=True, exist_ok=True)
	safe_username = username.replace("/", "_")
	target = raw_dir / f"{safe_username}_{video_id}.txt"
	target.write_text(raw_text, encoding="utf-8")
	return target


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="TikTok → download → Gemini virality analysis → PostgreSQL")
	parser.add_argument("--username", required=True, help="TikTok username to fetch recent videos from")
	parser.add_argument("--count", type=int, default=20, help="Number of recent videos to process")
	parser.add_argument("--download-only", action="store_true", help="Only download videos (skip DB and Gemini analysis)")
	parser.add_argument("--database-url", default=None, help="PostgreSQL DSN (overrides DATABASE_URL env var)")
	parser.add_argument("--download-dir", default="downloads", help="Folder for full downloaded mp4 files")
	parser.add_argument("--clip-dir", default="clips", help="Folder for first-N-second clips")
	parser.add_argument("--raw-dir", default="raw_gemini", help="Folder for raw Gemini responses")
	parser.add_argument("--clip-seconds", type=int, default=10, help="Duration for clip extraction")
	parser.add_argument("--sleep-seconds", type=float, default=2.0, help="Rate limiting between videos")
	parser.add_argument("--retry-count", type=int, default=3, help="Retry attempts per step")
	parser.add_argument("--retry-delay", type=float, default=2.0, help="Delay between retries")
	parser.add_argument("--model", default="gemini-1.5-pro", help="Gemini model name")
	return parser


def run_pipeline(args: argparse.Namespace) -> None:
	if args.count <= 0:
		raise ValueError("--count must be greater than 0")
	if args.clip_seconds <= 0:
		raise ValueError("--clip-seconds must be greater than 0")

	database_url = args.database_url or os.getenv("DATABASE_URL")
	if not args.download_only and not database_url:
		raise ValueError("DATABASE_URL environment variable is required (or pass --database-url), unless using --download-only")

	scraper = TikTokScraper(
		ms_token=os.getenv("ms_token"),
		browser=os.getenv("TIKTOK_BROWSER", "chromium"),
	)
	downloader = VideoDownloader(
		download_dir=args.download_dir,
		clip_dir=args.clip_dir,
		clip_seconds=args.clip_seconds,
	)
	analyzer = None
	if not args.download_only:
		analyzer = GeminiVideoAnalyzer(
			api_key=os.getenv("GEMINI_API_KEY"),
			model_name=args.model,
			prompt=VIRALITY_PROMPT,
		)

	print(f"Fetching up to {args.count} recent videos for @{args.username}...")
	videos = _retry(
		operation_name="tiktok_fetch",
		func=lambda: scraper.fetch_recent_videos(username=args.username, count=args.count),
		retries=args.retry_count,
		delay_seconds=args.retry_delay,
	)
	print(f"Fetched: {len(videos)} videos")

	if args.download_only:
		for index, video in enumerate(videos, start=1):
			video_id = str(video.get("id", "unknown"))
			print(f"[{index}/{len(videos)}] Downloading video {video_id}")

			downloaded_path = _retry(
				operation_name=f"download_video:{video_id}",
				func=lambda: downloader.download_video(video_payload=video, username=args.username),
				retries=args.retry_count,
				delay_seconds=args.retry_delay,
			)
			print(f"Saved: {downloaded_path}")
			time.sleep(args.sleep_seconds)

		return

	raw_dir = Path(args.raw_dir)

	if database_url is None:
		raise RuntimeError("database_url must be set when not in --download-only mode")

	with PostgresDB(database_url) as db:
		db.initialize_schema()
		creator_profile = {
			"username": args.username,
			"source": "TikTokApi",
		}
		creator_id = db.upsert_creator(username=args.username, profile=creator_profile)

		for index, video in enumerate(videos, start=1):
			video_id = str(video.get("id", "unknown"))
			print(f"[{index}/{len(videos)}] Processing video {video_id}")

			db_video_id = db.upsert_video(creator_id=creator_id, video_payload=video)
			raw_response: str | None = None
			parsed_response: dict[str, Any] | None = None

			try:
				downloaded_path, clip_path = _retry(
					operation_name=f"download_and_clip:{video_id}",
					func=lambda: downloader.download_and_clip(video_payload=video, username=args.username),
					retries=args.retry_count,
					delay_seconds=args.retry_delay,
				)
				db.update_video_media_paths(
					video_id=db_video_id,
					downloaded_path=str(downloaded_path),
					clip_path=str(clip_path),
				)

				raw_response, parsed_response = _retry(
					operation_name=f"gemini_analyze:{video_id}",
					func=lambda: analyzer.analyze_video(clip_path),
					retries=args.retry_count,
					delay_seconds=args.retry_delay,
				)
				raw_file = _save_raw_response(raw_dir=raw_dir, username=args.username, video_id=video_id, raw_text=raw_response)
				print(f"Saved raw Gemini response: {raw_file}")

				db.insert_analysis(
					video_id=db_video_id,
					model_name=args.model,
					prompt=VIRALITY_PROMPT,
					raw_response=raw_response,
					parsed_json=parsed_response,
					status="success",
				)
			except Exception as exc:  # pragma: no cover - runtime/network dependent
				if raw_response:
					_save_raw_response(raw_dir=raw_dir, username=args.username, video_id=video_id, raw_text=raw_response)

				db.insert_analysis(
					video_id=db_video_id,
					model_name=args.model,
					prompt=VIRALITY_PROMPT,
					raw_response=raw_response,
					parsed_json=parsed_response,
					status="failed",
					error_message=str(exc),
				)
				print(f"Failed video {video_id}: {exc}")

			time.sleep(args.sleep_seconds)


def main() -> None:
	parser = build_parser()
	args = parser.parse_args()
	run_pipeline(args)


if __name__ == "__main__":
	main()
