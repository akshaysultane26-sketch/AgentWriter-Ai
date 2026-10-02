import os
import time
import urllib.parse
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(exist_ok=True)


def _pollinations(prompt: str):
    """Free, no API key. Anonymous use is limited to about 1 request per 15 seconds."""
    url = (
        "https://image.pollinations.ai/prompt/"
        + urllib.parse.quote(prompt)
        + "?width=1024&height=576&model=flux&nologo=true"
    )
    for _ in range(3):
        try:
            r = requests.get(url, timeout=90)
            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                return r.content, "jpg"
            if r.status_code == 429:
                time.sleep(16)  # rate limited: wait and try again
        except requests.RequestException:
            time.sleep(5)
    return None


def _gemini(prompt: str):
    """Backup provider. Only used if GOOGLE_API_KEY is set and the key allows image generation."""
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=key)
        model = os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-image")
        resp = client.models.generate_content(model=model, contents=prompt)
        for part in resp.candidates[0].content.parts:
            inline = getattr(part, "inline_data", None)
            if inline and inline.data:
                return inline.data, "png"
    except Exception as e:
        print("Gemini image failed:", e)
    return None


PROVIDERS = [_pollinations, _gemini]


def generate_image(prompt: str, name: str):
    """Returns the saved file path, or None if every provider fails."""
    if os.getenv("ENABLE_IMAGES", "true").lower() != "true":
        return None
    for provider in PROVIDERS:
        result = provider(prompt)
        if result:
            data, ext = result
            path = IMAGES_DIR / f"{name}.{ext}"
            path.write_bytes(data)
            return str(path)
    return None