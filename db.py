from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import psycopg  # type: ignore[reportMissingImports]


class PostgresDB:
    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("DATABASE_URL is required")
        self.dsn = dsn
        self.conn: psycopg.Connection[Any] | None = None

    def connect(self) -> None:
        conn = psycopg.connect(self.dsn)
        conn.autocommit = True
        self.conn = conn

    def close(self) -> None:
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self) -> "PostgresDB":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()

    def _require_conn(self) -> psycopg.Connection[Any]:
        if self.conn is None:
            raise RuntimeError("Database connection is not initialized")
        return self.conn

    def initialize_schema(self) -> None:
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS creators (
                    id BIGSERIAL PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE,
                    profile JSONB,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    id BIGSERIAL PRIMARY KEY,
                    creator_id BIGINT NOT NULL REFERENCES creators(id),
                    tiktok_video_id TEXT NOT NULL UNIQUE,
                    created_at_tiktok TIMESTAMPTZ,
                    web_video_url TEXT,
                    stats JSONB,
                    raw_payload JSONB,
                    downloaded_path TEXT,
                    clip_path TEXT,
                    inserted_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS video_analysis (
                    id BIGSERIAL PRIMARY KEY,
                    video_id BIGINT NOT NULL REFERENCES videos(id),
                    model_name TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    raw_response TEXT,
                    parsed_json JSONB,
                    status TEXT NOT NULL,
                    error_message TEXT,
                    analyzed_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

    def upsert_creator(self, username: str, profile: dict[str, Any] | None = None) -> int:
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO creators (username, profile, updated_at)
                VALUES (%s, %s::jsonb, NOW())
                ON CONFLICT (username)
                DO UPDATE SET profile = EXCLUDED.profile, updated_at = NOW()
                RETURNING id;
                """,
                (username, json.dumps(profile or {})),
            )
            creator_id = cur.fetchone()

        if creator_id is None:
            raise RuntimeError("Failed to upsert creator")
        return int(creator_id[0])

    @staticmethod
    def _parse_tiktok_datetime(video_payload: dict[str, Any]) -> datetime | None:
        value = video_payload.get("createTimeISO")
        if isinstance(value, str) and value.strip():
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None

        unix_value = video_payload.get("createTime")
        if isinstance(unix_value, (int, float)):
            return datetime.fromtimestamp(unix_value, tz=timezone.utc)

        return None

    def upsert_video(
        self,
        creator_id: int,
        video_payload: dict[str, Any],
    ) -> int:
        conn = self._require_conn()

        tiktok_video_id = str(video_payload.get("id", "")).strip()
        if not tiktok_video_id:
            raise ValueError("Video payload is missing id")

        stats = video_payload.get("stats") if isinstance(video_payload.get("stats"), dict) else {}
        web_video_url = video_payload.get("webVideoUrl")
        created_at_tiktok = self._parse_tiktok_datetime(video_payload)

        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO videos (
                    creator_id,
                    tiktok_video_id,
                    created_at_tiktok,
                    web_video_url,
                    stats,
                    raw_payload,
                    updated_at
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, NOW())
                ON CONFLICT (tiktok_video_id)
                DO UPDATE SET
                    creator_id = EXCLUDED.creator_id,
                    created_at_tiktok = EXCLUDED.created_at_tiktok,
                    web_video_url = EXCLUDED.web_video_url,
                    stats = EXCLUDED.stats,
                    raw_payload = EXCLUDED.raw_payload,
                    updated_at = NOW()
                RETURNING id;
                """,
                (
                    creator_id,
                    tiktok_video_id,
                    created_at_tiktok,
                    web_video_url,
                    json.dumps(stats),
                    json.dumps(video_payload),
                ),
            )
            video_id = cur.fetchone()

        if video_id is None:
            raise RuntimeError("Failed to upsert video")
        return int(video_id[0])

    def update_video_media_paths(self, video_id: int, downloaded_path: str, clip_path: str) -> None:
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE videos
                SET downloaded_path = %s,
                    clip_path = %s,
                    updated_at = NOW()
                WHERE id = %s;
                """,
                (downloaded_path, clip_path, video_id),
            )

    def insert_analysis(
        self,
        video_id: int,
        model_name: str,
        prompt: str,
        raw_response: str | None,
        parsed_json: dict[str, Any] | None,
        status: str,
        error_message: str | None = None,
    ) -> None:
        conn = self._require_conn()
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO video_analysis (
                    video_id,
                    model_name,
                    prompt,
                    raw_response,
                    parsed_json,
                    status,
                    error_message
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s);
                """,
                (
                    video_id,
                    model_name,
                    prompt,
                    raw_response,
                    json.dumps(parsed_json) if parsed_json is not None else None,
                    status,
                    error_message,
                ),
            )
