from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from typing import Any

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

HKDF_INFO = b"conversation-analyzer-v1"
SESSION_TTL_SECONDS = 900 
AES_NONCE_BYTES = 12

#session store
class SessionStore:
    #init session store
    def __init__(self) -> None:
        self._sessions: dict[str, tuple[bytes, float]] = {}
        self._lock = threading.Lock()

    #put session id and key
    def put(self, session_id: str, key: bytes) -> None:
        with self._lock:
            self._purge_expired()
            self._sessions[session_id] = (key, time.time())

    #get session id and key
    def get(self, session_id: str) -> bytes | None:
        with self._lock:
            self._purge_expired()
            entry = self._sessions.get(session_id)
            if entry is None:
                return None
            key, created = entry
            if time.time() - created > SESSION_TTL_SECONDS:
                del self._sessions[session_id]
                return None
            return key

    #purge expired sessions
    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            sid
            for sid, (_, created) in self._sessions.items()
            if now - created > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]


sessions = SessionStore()

def _derive_aes_key(shared_secret: bytes, session_id: str) -> bytes:
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=session_id.encode("utf-8"),
        info=HKDF_INFO,
    )
    return hkdf.derive(shared_secret)

def create_handshake(client_public_key_b64: str) -> dict[str, str]:
    try:
        client_bytes = base64.b64decode(client_public_key_b64, validate=True)
        client_key = serialization.load_der_public_key(client_bytes)
        if not isinstance(client_key, ec.EllipticCurvePublicKey):
            raise ValueError("Client key must be ECDH P-256.")
    except Exception as exc:
        raise ValueError("Invalid client public key.") from exc

    server_private = ec.generate_private_key(ec.SECP256R1())
    server_public = server_private.public_key()

    shared_secret = server_private.exchange(ec.ECDH(), client_key)
    session_id = secrets.token_urlsafe(32)
    aes_key = _derive_aes_key(shared_secret, session_id)
    sessions.put(session_id, aes_key)

    server_spki = server_public.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return {
        "session_id": session_id,
        "server_public_key": base64.b64encode(server_spki).decode("ascii"),
    }

def encrypt_json(session_id: str, payload: Any) -> dict[str, str]:
    key = sessions.get(session_id)
    if key is None:
        raise ValueError("Unknown or expired session.")

    plaintext = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    nonce = secrets.token_bytes(AES_NONCE_BYTES)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)

    return {
        "session_id": session_id,
        "iv": base64.b64encode(nonce).decode("ascii"),
        "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
    }


def decrypt_json(envelope: dict[str, str]) -> Any:
    session_id = envelope.get("session_id") or ""
    key = sessions.get(session_id)
    if key is None:
        raise ValueError("Unknown or expired session.")

    try:
        nonce = base64.b64decode(envelope["iv"], validate=True)
        ciphertext = base64.b64decode(envelope["ciphertext"], validate=True)
    except Exception as exc:
        raise ValueError("Invalid encrypted envelope.") from exc

    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return json.loads(plaintext.decode("utf-8"))
