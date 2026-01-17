# main.py
import json
import subprocess
import tempfile
from pathlib import Path

from flask import Flask, request
from google.cloud import storage

app = Flask(__name__)
storage_client = storage.Client()

# Only process these job files (recommended convention)
JOB_PATH_MARKER = "/jobs/"
JOB_SUFFIX = "-epub-job.json"


def parse_gs_uri(gs_uri: str):
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    path = gs_uri[len("gs://") :]
    bucket, _, blob = path.partition("/")
    if not bucket or not blob:
        raise ValueError(f"Invalid GCS URI: {gs_uri}")
    return bucket, blob


def download_text(gs_uri: str) -> str:
    bucket, blob = parse_gs_uri(gs_uri)
    return storage_client.bucket(bucket).blob(blob).download_as_text(encoding="utf-8")


def upload_file(gs_uri: str, local_path: str, content_type: str):
    bucket, blob = parse_gs_uri(gs_uri)
    storage_client.bucket(bucket).blob(blob).upload_from_filename(
        local_path, content_type=content_type
    )


@app.post("/")
def handle_event():
    # Eventarc delivers a CloudEvent-like JSON body. We defensively support a few shapes.
    envelope = request.get_json(silent=True) or {}

    # Common: { "bucket": "...", "name": "..." }
    bucket = envelope.get("bucket")
    name = envelope.get("name")

    # Sometimes nested under "data"
    if not bucket or not name:
        data = envelope.get("data") or {}
        bucket = bucket or data.get("bucket") or data.get("bucketId")
        name = name or data.get("name") or data.get("objectId")

    # Pub/Sub push shapes (fallback)
    if not bucket or not name:
        data = envelope.get("data") or envelope
        message = data.get("message", {})
        attrs = message.get("attributes", {})
        bucket = bucket or attrs.get("bucketId") or data.get("bucketId")
        name = name or attrs.get("objectId") or data.get("objectId")

    if not bucket or not name:
        return ("Missing bucket/name in event payload", 400)

    # Guard: only process your EPUB job JSON objects; ignore everything else in the bucket
    if (JOB_PATH_MARKER not in name) or (not name.endswith(JOB_SUFFIX)):
        return ("Ignored (not an epub job file)", 204)

    job_gs_uri = f"gs://{bucket}/{name}"

    try:
        job = json.loads(download_text(job_gs_uri))
    except Exception as e:
        return (f"Failed to read/parse job JSON from {job_gs_uri}: {e}", 400)

    # Required fields
    try:
        md_uri = job["manuscript_gs_uri"]
        epub_uri = job["output_epub_gs_uri"]
    except KeyError as e:
        return (f"Job JSON missing required field: {e}", 400)

    meta = job.get("metadata", {}) if isinstance(job.get("metadata", {}), dict) else {}

    title = meta.get("title", "Untitled Book")
    author = meta.get("author", "Unknown Author")
    subtitle = meta.get("subtitle", "")
    lang = meta.get("lang", "en-GB")

    try:
        # EPUB cannot use Kindle mbp:pagebreak. Remove/replace.
        md_text = download_text(md_uri).replace("<mbp:pagebreak />", "\n\n")
    except Exception as e:
        return (f"Failed to download manuscript: {md_uri}: {e}", 500)

    try:
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            md_path = td / "book.md"
            epub_path = td / "book.epub"
            md_path.write_text(md_text, encoding="utf-8")

            args = [
                "pandoc",
                str(md_path),
                "-o",
                str(epub_path),
                "--toc",
                "--metadata",
                f"lang={lang}",
                "--metadata",
                f"title={title}",
                "--metadata",
                f"author={author}",
            ]
            if subtitle:
                args += ["--metadata", f"subtitle={subtitle}"]

            subprocess.run(args, check=True, capture_output=True, text=True)

            upload_file(epub_uri, str(epub_path), "application/epub+zip")

    except subprocess.CalledProcessError as e:
        # pandoc stderr is often the best signal
        return (f"pandoc failed: {e.stderr or e.stdout or str(e)}", 500)
    except Exception as e:
        return (f"EPUB build failed: {e}", 500)

    return ("OK", 200)
