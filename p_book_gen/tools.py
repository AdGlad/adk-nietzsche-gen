# p_book_gen/tools.py
"""
Function tools for saving per-chapter content to Google Cloud Storage.

Exposed tools:
- save_chapter_to_gcs(chapter_number: int, book_topic: str, content_markdown: str) -> dict
"""

import os
import re
import uuid
from datetime import datetime
from typing import Dict

from google.cloud import storage
from google.adk.tools.function_tool import FunctionTool

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

# Use an env var so you can point to any bucket you like.
BUCKET_NAME = os.environ.get("BOOK_GEN_BUCKET")

_storage_client = None


def _get_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _safe_slug(value: str) -> str:
    """
    Create a lowercase, URL-safe slug from a book topic or title.
    """
    value = value.strip()
    if not value:
        return "untitled"
    value = re.sub(r"[^a-zA-Z0-9_-]+", "-", value)
    value = value.strip("-").lower()
    return value or "untitled"


# --------------------------------------------------------------------
# TOOL IMPLEMENTATION
# --------------------------------------------------------------------

def save_chapter_to_gcs(
    chapter_number: int,
    book_topic: str,
    content_markdown: str,
) -> Dict[str, str]:
    """
    Save a single chapter's markdown content to Google Cloud Storage.

    Args:
        chapter_number: 1-based chapter number.
        book_topic: The overall topic of the book (used for folder naming).
        content_markdown: The chapter content in Markdown.

    Returns:
        A dict with the bucket name and object path where the chapter was stored.
    """
    if not BUCKET_NAME:
        raise RuntimeError(
            "BOOK_GEN_BUCKET environment variable is not set. "
            "Please export BOOK_GEN_BUCKET to the name of your GCS bucket."
        )

    client = _get_client()
    bucket = client.bucket(BUCKET_NAME)

    safe_topic = _safe_slug(book_topic)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    random_id = uuid.uuid4().hex[:8]

    object_name = (
        f"books/{safe_topic}/chapters/"
        f"{timestamp}-chap{chapter_number:02d}-{random_id}.md"
    )

    blob = bucket.blob(object_name)
    blob.upload_from_string(content_markdown, content_type="text/markdown; charset=utf-8")

    return {
        "bucket": BUCKET_NAME,
        "object_path": object_name,
        "gs_uri": f"gs://{BUCKET_NAME}/{object_name}",
    }


# ADK will infer the tool name and schema from the function signature.
save_chapter_to_gcs_tool = FunctionTool(save_chapter_to_gcs)

