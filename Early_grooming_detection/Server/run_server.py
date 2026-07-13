#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent
CERT_DIR = ROOT / "certs"
CERT_FILE = CERT_DIR / "server.crt"
KEY_FILE = CERT_DIR / "server.key"

HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "8000"))
USE_TLS = os.environ.get("USE_TLS", "0") == "1"


def ensure_tls_certs() -> tuple[str, str] | None:
    if not USE_TLS:
        return None
    if CERT_FILE.is_file() and KEY_FILE.is_file():
        return str(CERT_FILE), str(KEY_FILE)

    print("TLS certificates not found — generating self-signed cert in certs/ …")
    script = ROOT / "scripts" / "generate_tls_certs.py"
    subprocess.run([sys.executable, str(script)], check=True)
    if not CERT_FILE.is_file() or not KEY_FILE.is_file():
        raise SystemExit("Failed to generate TLS certificates.")
    return str(CERT_FILE), str(KEY_FILE)


def main():
    ssl_kwargs = {}
    scheme = "http"
    if USE_TLS:
        cert, key = ensure_tls_certs()
        ssl_kwargs = {"ssl_certfile": cert, "ssl_keyfile": key}
        scheme = "https"

    print(f"Starting API on {scheme}://{HOST}:{PORT} (model: bert_goemotions)")
    print(f"Web client: {scheme}://127.0.0.1:{PORT}/app/")
    print("Transport: TLS + ECDH session keys + AES-256-GCM on /predict/secure")

    uvicorn.run(
        "serve:app",
        host=HOST,
        port=PORT,
        app_dir="models",
        **ssl_kwargs,
    )


if __name__ == "__main__":
    main()
