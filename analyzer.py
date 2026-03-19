from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import google.generativeai as genai  # type: ignore[reportMissingImports]

VIRALITY_PROMPT = """You are analyzing a TikTok video for virality patterns.

Return ONLY a JSON object with the following fields:

{
  \"hook_strength\": 0-10,
  \"hook_type\": \"curiosity | shock | storytelling | visual | none\",
  \"attention_grabbing_first_3s\": true/false,

  \"content_type\": \"educational | entertainment | lifestyle | meme | other\",

  \"editing_style\": {
    \"cut_speed\": \"slow | medium | fast\",
    \"has_subtitles\": true/false,
    \"has_text_overlay\": true/false
  },

  \"visual_features\": {
    \"has_face\": true/false,
    \"camera_distance\": \"close | medium | far\",
    \"scene_changes\": \"low | medium | high\"
  },

  \"audio_features\": {
    \"speech_present\": true/false,
    \"music_present\": true/false,
    \"emotion\": \"neutral | excited | dramatic | funny\"
  },

  \"engagement_drivers\": [
    \"list of reasons why this could go viral\"
  ],

  \"estimated_target_audience\": \"description\",

  \"loopable\": true/false,

  \"overall_virality_score\": 0-10
}
"""


class GeminiVideoAnalyzer:
    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.5-pro",
        prompt: str = VIRALITY_PROMPT,
    ) -> None:
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is required")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.prompt = prompt
        self.model_name = model_name

    @staticmethod
    def _extract_json_object(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = cleaned.replace("json", "", 1).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON object found in Gemini response")
        return cleaned[start : end + 1]

    def analyze_video(self, video_file: Path) -> tuple[str, dict[str, Any]]:
        uploaded_file = genai.upload_file(path=str(video_file))
        response = self.model.generate_content([self.prompt, uploaded_file])
        raw_text = response.text or ""

        if not raw_text.strip():
            raise ValueError("Gemini returned an empty response")

        json_str = self._extract_json_object(raw_text)
        parsed = json.loads(json_str)
        if not isinstance(parsed, dict):
            raise ValueError("Gemini response JSON is not an object")

        return raw_text, parsed
