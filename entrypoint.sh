#!/bin/bash
# ============================================================
# PhishGuard++ — Container Startup Script
# Uses Python GCS client to download models, then starts server
# ============================================================
set -e

echo "🔍 Checking for trained models..."

python3 - <<'PYEOF'
import os
import sys
from pathlib import Path

bucket_name = os.getenv("GCS_BUCKET", "project-5d926acd-a531-48ca-bef-phishguard-models")
local_dir = Path("/app/models")
sentinel = local_dir / ".downloaded"

if sentinel.exists():
    print("✅ Models already present, skipping download.")
    sys.exit(0)

print(f"📥 Downloading models from gs://{bucket_name}/models/ ...")
try:
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix="models/"))
    for blob in blobs:
        # Strip the "models/" prefix to get the relative path
        rel = blob.name[len("models/"):]
        if not rel:
            continue
        dest = local_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        print(f"  ↓ {rel}")
        blob.download_to_filename(str(dest))
    sentinel.touch()
    print("✅ All models downloaded successfully.")
except Exception as e:
    print(f"❌ Model download failed: {e}")
    # Continue anyway — models may be partially cached or baked in
PYEOF

echo "🚀 Starting PhishGuard++ backend on port ${PORT:-8080}..."
exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8080}" --workers 1
