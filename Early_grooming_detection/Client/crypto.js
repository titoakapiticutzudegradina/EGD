const HKDF_INFO = new TextEncoder().encode("conversation-analyzer-v1");
const SESSION_STORAGE_KEY = "ca-crypto-session";

/** @type {{ sessionId: string, aesKey: CryptoKey } | null} */
let activeSession = null;

function bufToB64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function b64ToBuf(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes.buffer;
}

async function deriveAesKey(sharedBits, sessionId) {
  const keyMaterial = await crypto.subtle.importKey("raw", sharedBits, "HKDF", false, [
    "deriveKey",
  ]);
  return crypto.subtle.deriveKey(
    {
      name: "HKDF",
      hash: "SHA-256",
      salt: new TextEncoder().encode(sessionId),
      info: HKDF_INFO,
    },
    keyMaterial,
    { name: "AES-GCM", length: 256 },
    false,
    ["encrypt", "decrypt"]
  );
}

async function performHandshake(apiBase) {
  const clientKeys = await crypto.subtle.generateKey(
    { name: "ECDH", namedCurve: "P-256" },
    true,
    ["deriveBits"]
  );

  const clientSpki = await crypto.subtle.exportKey("spki", clientKeys.publicKey);
  const res = await window.ApiClient.apiFetch(apiBase, "/crypto/handshake", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ client_public_key: bufToB64(clientSpki) }),
  });

  const body = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof body.detail === "string" ? body.detail : res.statusText;
    throw new Error(detail || "Handshake failed.");
  }

  const serverKey = await crypto.subtle.importKey(
    "spki",
    b64ToBuf(body.server_public_key),
    { name: "ECDH", namedCurve: "P-256" },
    false,
    []
  );

  const sharedBits = await crypto.subtle.deriveBits(
    { name: "ECDH", public: serverKey },
    clientKeys.privateKey,
    256
  );

  const aesKey = await deriveAesKey(sharedBits, body.session_id);
  activeSession = { sessionId: body.session_id, aesKey };

  sessionStorage.setItem(
    SESSION_STORAGE_KEY,
    JSON.stringify({ sessionId: body.session_id, apiBase })
  );

  return activeSession;
}

async function ensureSession(apiBase) {
  if (activeSession) {
    return activeSession;
  }
  return performHandshake(apiBase);
}

async function encryptPayload(apiBase, payload) {
  const { sessionId, aesKey } = await ensureSession(apiBase);
  const plaintext = new TextEncoder().encode(JSON.stringify(payload));
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt({ name: "AES-GCM", iv }, aesKey, plaintext);

  return {
    session_id: sessionId,
    iv: bufToB64(iv),
    ciphertext: bufToB64(ciphertext),
  };
}

async function decryptEnvelope(apiBase, envelope) {
  const { aesKey } = await ensureSession(apiBase);
  const iv = new Uint8Array(b64ToBuf(envelope.iv));
  const ciphertext = b64ToBuf(envelope.ciphertext);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv },
    aesKey,
    ciphertext
  );
  return JSON.parse(new TextDecoder().decode(plaintext));
}

async function securePredict(apiBase, payload) {
  const encrypted = await encryptPayload(apiBase, payload);
  const res = await window.ApiClient.apiFetch(apiBase, "/predict/secure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(encrypted),
  });

  const raw = await res.text();
  let body = {};
  try {
    body = raw ? JSON.parse(raw) : {};
  } catch {
    if (raw.includes("Unsupported method")) {
      throw new Error(
        `POST went to the wrong server. Use https://127.0.0.1:8000/app/ with python run_server.py`
      );
    }
    throw new Error("Invalid server response.");
  }

  if (!res.ok) {
    const detail = body.detail;
    const msg =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map((d) => d.msg ?? JSON.stringify(d)).join("; ")
          : res.statusText;
    throw new Error(msg || `Request failed (${res.status})`);
  }

  if (!body.iv || !body.ciphertext) {
    throw new Error("Server returned an unencrypted response.");
  }

  return decryptEnvelope(apiBase, body);
}

function clearSession() {
  activeSession = null;
  sessionStorage.removeItem(SESSION_STORAGE_KEY);
}

window.SecureTransport = {
  performHandshake,
  ensureSession,
  securePredict,
  clearSession,
};
