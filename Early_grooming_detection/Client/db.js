const SQL_JS_VERSION = "1.10.3";
const SQL_JS_BASE = `https://cdnjs.cloudflare.com/ajax/libs/sql.js/${SQL_JS_VERSION}`;
const IDB_NAME = "conversation-analyzer";
const IDB_STORE = "database";
const IDB_KEY = "sqlite";

/** @type {import("sql.js").Database | null} */
let db = null;

function utcNow() {
  return new Date().toISOString().replace(/\.\d{3}Z$/, "Z");
}

async function openIndexedDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_NAME, 1);
    req.onupgradeneeded = () => {
      req.result.createObjectStore(IDB_STORE);
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function loadDatabaseBytes() {
  const idb = await openIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = idb.transaction(IDB_STORE, "readonly");
    const get = tx.objectStore(IDB_STORE).get(IDB_KEY);
    get.onsuccess = () => resolve(get.result ?? null);
    get.onerror = () => reject(get.error);
  });
}

async function persistDatabase() {
  if (!db) return;
  const data = db.export();
  const idb = await openIndexedDB();
  return new Promise((resolve, reject) => {
    const tx = idb.transaction(IDB_STORE, "readwrite");
    tx.objectStore(IDB_STORE).put(data, IDB_KEY);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

function runSchema() {
  db.exec(`
    CREATE TABLE IF NOT EXISTS conversations (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      title TEXT NOT NULL,
      created_at TEXT NOT NULL,
      updated_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS blocks (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      conversation_id INTEGER NOT NULL,
      position INTEGER NOT NULL,
      raw_text TEXT NOT NULL,
      messages_json TEXT NOT NULL,
      created_at TEXT NOT NULL,
      FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_blocks_conversation
      ON blocks(conversation_id, position);
  `);
}

async function initDb() {
  const SQL = await window.initSqlJs({
    locateFile: (file) => `${SQL_JS_BASE}/${file}`,
  });

  const saved = await loadDatabaseBytes();
  db = saved ? new SQL.Database(saved) : new SQL.Database();
  runSchema();
  await persistDatabase();
}

function parseBlock(raw) {
  const lines = raw.split("\n").map((l) => l.trim()).filter(Boolean);
  if (lines.length === 0) return [];

  /** @type {{ author: string, text: string }[]} */
  const messages = [];

  for (const line of lines) {
    const colon = line.indexOf(":");
    if (colon === -1) {
      throw new Error(
        `Invalid line (use "Name: message"): ${line.slice(0, 60)}${line.length > 60 ? "…" : ""}`
      );
    }
    const author = line.slice(0, colon).trim();
    const text = line.slice(colon + 1).trim();
    if (!author) throw new Error(`Missing speaker name: ${line.slice(0, 60)}`);
    if (!text) continue;
    messages.push({ author, text });
  }

  if (messages.length === 0) {
    throw new Error("No messages found in block.");
  }
  return messages;
}

function queryAll(sql, params = []) {
  const stmt = db.prepare(sql);
  try {
    if (params.length > 0) {
      stmt.bind(params);
    }
    const rows = [];
    while (stmt.step()) {
      rows.push(stmt.getAsObject());
    }
    return rows;
  } finally {
    stmt.free();
  }
}

function queryOne(sql, params = []) {
  const rows = queryAll(sql, params);
  return rows[0] ?? null;
}

function listConversations() {
  return queryAll(`
    SELECT c.id, c.title, c.created_at, c.updated_at,
           COUNT(b.id) AS block_count
    FROM conversations c
    LEFT JOIN blocks b ON b.conversation_id = c.id
    GROUP BY c.id
    ORDER BY c.updated_at DESC
  `).map((r) => ({
    id: Number(r.id),
    title: r.title,
    created_at: r.created_at,
    updated_at: r.updated_at,
    block_count: r.block_count,
  }));
}

function getConversation(conversationId) {
  const id = Number(conversationId);
  const row = queryOne(
    `SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?`,
    [id]
  );
  if (!row) return null;

  const blockRows = queryAll(
    `SELECT id, position, raw_text, messages_json, created_at
     FROM blocks WHERE conversation_id = ?
     ORDER BY position ASC`,
    [id]
  );

  return {
    id: Number(row.id),
    title: row.title,
    created_at: row.created_at,
    updated_at: row.updated_at,
    blocks: blockRows.map((b) => ({
      id: b.id,
      position: b.position,
      raw_text: b.raw_text,
      messages: JSON.parse(b.messages_json),
      created_at: b.created_at,
    })),
  };
}

async function createConversation(title = null) {
  const now = utcNow();
  const finalTitle =
    title && String(title).trim()
      ? String(title).trim()
      : `Conversation ${now.slice(0, 10)}`;

  const row = queryOne(
    `INSERT INTO conversations (title, created_at, updated_at)
     VALUES (?, ?, ?)
     RETURNING id`,
    [finalTitle, now, now]
  );
  if (!row || row.id == null) {
    throw new Error("Failed to create conversation.");
  }

  await persistDatabase();
  const conv = getConversation(Number(row.id));
  if (!conv) {
    throw new Error("Failed to load new conversation.");
  }
  return conv;
}

async function addBlock(conversationId, rawText) {
  const id = Number(conversationId);
  const conv = getConversation(id);
  if (!conv) return null;

  const messages = parseBlock(rawText);
  const now = utcNow();

  const posRow = queryOne(
    `SELECT COALESCE(MAX(position), -1) + 1 AS pos FROM blocks WHERE conversation_id = ?`,
    [id]
  );
  if (!posRow) {
    throw new Error("Failed to resolve block position.");
  }
  const position = Number(posRow.pos);

  const inserted = queryOne(
    `INSERT INTO blocks (conversation_id, position, raw_text, messages_json, created_at)
     VALUES (?, ?, ?, ?, ?)
     RETURNING id`,
    [id, position, rawText, JSON.stringify(messages), now]
  );
  if (!inserted || inserted.id == null) {
    throw new Error("Failed to add block.");
  }

  db.run(`UPDATE conversations SET updated_at = ? WHERE id = ?`, [now, id]);

  await persistDatabase();

  const blockId = Number(inserted.id);
  return {
    block: {
      id: blockId,
      conversation_id: id,
      position,
      raw_text: rawText,
      messages,
      created_at: now,
    },
    conversation: getConversation(id),
  };
}

async function deleteConversation(conversationId) {
  const id = Number(conversationId);
  const row = queryOne(`SELECT id FROM conversations WHERE id = ?`, [id]);
  if (!row) return false;

  db.run(`DELETE FROM blocks WHERE conversation_id = ?`, [id]);
  db.run(`DELETE FROM conversations WHERE id = ?`, [id]);
  await persistDatabase();
  return true;
}

async function deleteBlock(blockId) {
  const row = queryOne(`SELECT conversation_id FROM blocks WHERE id = ?`, [blockId]);
  if (!row) return false;

  const conversationId = row.conversation_id;
  db.run(`DELETE FROM blocks WHERE id = ?`, [blockId]);

  const remaining = queryAll(
    `SELECT id FROM blocks WHERE conversation_id = ? ORDER BY position ASC, id ASC`,
    [conversationId]
  );
  remaining.forEach((blk, pos) => {
    db.run(`UPDATE blocks SET position = ? WHERE id = ?`, [pos, blk.id]);
  });

  const now = utcNow();
  db.run(`UPDATE conversations SET updated_at = ? WHERE id = ?`, [now, conversationId]);

  await persistDatabase();
  return true;
}

window.ConversationDb = {
  initDb,
  parseBlock,
  listConversations,
  getConversation,
  createConversation,
  addBlock,
  deleteBlock,
  deleteConversation,
};
