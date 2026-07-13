const API_PORT = "8000";

function networkErrorMessage(apiBase, cause) {
  const appUrl = `http://127.0.0.1:${API_PORT}/app/`;
  const httpsUrl = `https://127.0.0.1:${API_PORT}/app/`;
  if (apiBase.startsWith("https:")) {
    return (
      `Cannot reach ${apiBase} (${cause}). ` +
      `Start the server (cd Licenta && python run_server.py), open ${httpsUrl}, ` +
      `and accept the self-signed certificate in the browser first. ` +
      `Or use HTTP: USE_TLS=0 python run_server.py and open ${appUrl}`
    );
  }
  return (
    `Cannot reach ${apiBase} (${cause}). ` +
    `Start the server with: cd Licenta && python run_server.py — ` +
    `then open ${appUrl} (not file:// and not port 5500 alone).`
  );
}

async function probeHealth(apiBase) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 4000);
  try {
    const res = await fetch(`${apiBase}/health`, {
      method: "GET",
      signal: controller.signal,
    });
    return res.ok;
  } catch {
    return false;
  } finally {
    clearTimeout(timeout);
  }
}


async function resolveApiBase() {
  const { protocol, hostname, port } = window.location;

  if (protocol === "http:" || protocol === "https:") {
    if (port === API_PORT) {
      const ok = await probeHealth(window.location.origin);
      if (ok) {
        return window.location.origin;
      }
    }
  }

  const hosts = [...new Set([hostname, "127.0.0.1", "localhost"])];
  const schemes = ["http", "https"];

  for (const host of hosts) {
    for (const scheme of schemes) {
      const base = `${scheme}://${host}:${API_PORT}`;
      if (await probeHealth(base)) {
        return base;
      }
    }
  }

  throw new Error(networkErrorMessage(`http://127.0.0.1:${API_PORT}`, "server not reachable"));
}

async function apiFetch(apiBase, path, options = {}) {
  const url = `${apiBase}${path}`;
  try {
    return await fetch(url, options);
  } catch (err) {
    const cause = err instanceof Error ? err.message : "network error";
    throw new Error(networkErrorMessage(apiBase, cause));
  }
}

window.ApiClient = {
  API_PORT,
  resolveApiBase,
  apiFetch,
  networkErrorMessage,
};
