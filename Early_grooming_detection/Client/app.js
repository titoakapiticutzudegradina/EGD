const DB = window.ConversationDb;

let API_BASE = "";
const SecureTransport = window.SecureTransport;

const MESSAGES_AT_10_PERCENT = 50;
const FULL_CONVERSATION_MESSAGES = MESSAGES_AT_10_PERCENT * 10;

/** @typedef {{ author: string, text: string }} ChatMessage */
/** @typedef {{ id: number, position: number, raw_text: string, messages: ChatMessage[], created_at: string }} Block */
/** @typedef {{ id: number, title: string, created_at: string, updated_at: string, blocks?: Block[], block_count?: number }} Conversation */

/** @type {Conversation | null} */
let currentConversation = null;

/** @type {Map<number, object>} */
const analysisByConversationId = new Map();

const DISPLAY_TITLES_KEY = "conversation-display-titles";

/** @type {Record<string, string>} */
let displayTitles = {};

function loadDisplayTitles() {
  try {
    const raw = localStorage.getItem(DISPLAY_TITLES_KEY);
    displayTitles = raw ? JSON.parse(raw) : {};
    if (displayTitles == null || typeof displayTitles !== "object") {
      displayTitles = {};
    }
  } catch {
    displayTitles = {};
  }
}

function saveDisplayTitles() {
  localStorage.setItem(DISPLAY_TITLES_KEY, JSON.stringify(displayTitles));
}

function getDisplayTitle(conv) {
  const override = displayTitles[String(conv.id)];
  return override != null && String(override).trim()
    ? String(override).trim()
    : conv.title;
}

function setDisplayTitle(conversationId, title) {
  const key = String(conversationId);
  const trimmed = String(title).trim();
  if (!trimmed) {
    delete displayTitles[key];
  } else {
    displayTitles[key] = trimmed;
  }
  saveDisplayTitles();
}

function clearDisplayTitle(conversationId) {
  delete displayTitles[String(conversationId)];
  saveDisplayTitles();
}

function conversationListLabel(conv) {
  const count = conv.block_count ?? 0;
  return `${getDisplayTitle(conv)} (${count} block${count === 1 ? "" : "s"})`;
}

const els = {
  conversationList: document.getElementById("conversation-list"),
  newConvBtn: document.getElementById("new-conv-btn"),
  currentTitle: document.getElementById("current-title"),
  existingTargetRadio: document.querySelector('input[name="target"][value="existing"]'),
  newTargetRadio: document.querySelector('input[name="target"][value="new"]'),
  blocksContainer: document.getElementById("blocks-container"),
  addForm: document.getElementById("add-form"),
  conversationBlock: document.getElementById("conversation-block"),
  analyzeBtn: document.getElementById("analyze-btn"),
  resultPanel: document.getElementById("result-panel"),
  resultSummary: document.getElementById("result-summary"),
  flaggedBlock: document.getElementById("flagged-block"),
  flaggedList: document.getElementById("flagged-list"),
  moreDetails: document.getElementById("more-details"),
  moreContent: document.getElementById("more-content"),
  error: document.getElementById("error"),
};

const authorColors = new Map();
const palette = ["#60a5fa", "#a78bfa", "#34d399", "#fbbf24", "#f472b6"];

function colorForAuthor(author) {
  const key = author.toLowerCase();
  if (!authorColors.has(key)) {
    authorColors.set(key, palette[authorColors.size % palette.length]);
  }
  return authorColors.get(key);
}

function showError(text) {
  if (!text) {
    els.error.classList.add("hidden");
    els.error.textContent = "";
    return;
  }
  els.error.textContent = text;
  els.error.classList.remove("hidden");
}

async function predict(messages, conversationId, blockMessageCounts) {
  if (!SecureTransport) {
    throw new Error("Encryption module (crypto.js) failed to load.");
  }
  return SecureTransport.securePredict(API_BASE, {
    messages: messages.map((m) => ({ text: m.text, author: m.author })),
    conversation_id: conversationId != null ? String(conversationId) : null,
    block_message_counts: blockMessageCounts,
  });
}

function progressFromMessageCount(messageCount) {
  if (messageCount <= 0) return 0;
  return Math.min(1, messageCount / FULL_CONVERSATION_MESSAGES);
}

function blockCheckpointLabel(cumulativeMessageCount) {
  const pct = (progressFromMessageCount(cumulativeMessageCount) * 100).toFixed(1);
  const suffix =
    cumulativeMessageCount > 0
      ? ` · ${cumulativeMessageCount} msg${cumulativeMessageCount === 1 ? "" : "s"}`
      : "";
  return `${pct}% of conversation${suffix}`;
}

function blockMessageCounts() {
  if (!currentConversation?.blocks) return [];
  return currentConversation.blocks.map(
    (b) => b.messages.filter((m) => (m.text || "").trim()).length
  );
}

function isNewTarget() {
  return els.newTargetRadio.checked;
}

function updateTargetUi() {
  const hasCurrent = currentConversation != null;
  els.existingTargetRadio.disabled = !hasCurrent;
  if (!hasCurrent && !isNewTarget()) {
    els.newTargetRadio.checked = true;
  }
}

function hideResultPanel() {
  els.resultPanel.classList.add("hidden");
}

function syncResultPanel() {
  const id = currentConversation?.id;
  if (id != null && analysisByConversationId.has(id)) {
    renderResult(analysisByConversationId.get(id), { fromCache: true });
  } else {
    hideResultPanel();
  }
}

function invalidateAnalysis(conversationId) {
  if (conversationId == null) return;
  analysisByConversationId.delete(conversationId);
  if (currentConversation?.id === conversationId) {
    hideResultPanel();
  }
}

function setCurrentConversation(conv) {
  currentConversation = conv;
  els.currentTitle.textContent = conv ? getDisplayTitle(conv) : "New conversation";
  els.analyzeBtn.disabled = !conv || !(conv.blocks && conv.blocks.length > 0);
  updateTargetUi();
  renderBlocks();
  highlightSidebar();
  syncResultPanel();
}

function startNewConversationDraft() {
  currentConversation = null;
  els.currentTitle.textContent = "New conversation";
  els.newTargetRadio.checked = true;
  els.analyzeBtn.disabled = true;
  updateTargetUi();
  renderBlocks();
  highlightSidebar();
  hideResultPanel();
  showError("");
}

function highlightSidebar() {
  const items = els.conversationList.querySelectorAll("li.conv-item");
  items.forEach((li) => {
    const id = Number(li.dataset.id);
    li.classList.toggle(
      "active",
      currentConversation != null && currentConversation.id === id
    );
  });
}

function renderConversationList(conversations) {
  els.conversationList.innerHTML = "";
  if (conversations.length === 0) {
    const empty = document.createElement("li");
    empty.className = "list-empty";
    empty.textContent = "No saved conversations";
    els.conversationList.appendChild(empty);
    return;
  }

  for (const conv of conversations) {
    const li = document.createElement("li");
    li.dataset.id = String(conv.id);
    li.className = "conv-item";
    if (currentConversation?.id === conv.id) li.classList.add("active");

    const row = document.createElement("div");
    row.className = "conv-row";

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "conv-select";
    btn.textContent = conversationListLabel(conv);
    btn.addEventListener("click", () => loadConversation(conv.id));

    const actions = document.createElement("div");
    actions.className = "conv-actions";

    const renameBtn = document.createElement("button");
    renameBtn.type = "button";
    renameBtn.className = "conv-action conv-rename";
    renameBtn.title = "Rename";
    renameBtn.setAttribute("aria-label", "Rename conversation");
    renameBtn.textContent = "✎";
    renameBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      renameConversationHandler(conv.id);
    });

    const deleteBtn = document.createElement("button");
    deleteBtn.type = "button";
    deleteBtn.className = "conv-action conv-delete";
    deleteBtn.title = "Delete";
    deleteBtn.setAttribute("aria-label", "Delete conversation");
    deleteBtn.textContent = "×";
    deleteBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversationHandler(conv.id);
    });

    actions.append(renameBtn, deleteBtn);
    row.append(btn, actions);
    li.appendChild(row);
    els.conversationList.appendChild(li);
  }
}

function renameConversationHandler(conversationId) {
  const conv = DB.getConversation(conversationId);
  if (!conv) {
    showError("Conversation not found.");
    refreshConversationList();
    return;
  }

  const current = getDisplayTitle(conv);
  const next = window.prompt("Rename conversation (display only):", current);
  if (next == null) return;

  const trimmed = next.trim();
  if (!trimmed) {
    showError("Name cannot be empty.");
    return;
  }

  if (trimmed === conv.title) {
    clearDisplayTitle(conversationId);
  } else {
    setDisplayTitle(conversationId, trimmed);
  }
  showError("");

  if (currentConversation?.id === conversationId) {
    els.currentTitle.textContent = getDisplayTitle(conv);
  }
  refreshConversationList();
}

async function deleteConversationHandler(conversationId) {
  const conv = DB.getConversation(conversationId);
  if (!conv) {
    showError("Conversation not found.");
    refreshConversationList();
    return;
  }

  const label = getDisplayTitle(conv);
  if (
    !window.confirm(
      `Delete "${label}" and all its blocks? This cannot be undone.`
    )
  ) {
    return;
  }

  showError("");
  try {
    if (!(await DB.deleteConversation(conversationId))) {
      showError("Conversation not found.");
      refreshConversationList();
      return;
    }

    clearDisplayTitle(conversationId);
    analysisByConversationId.delete(conversationId);

    if (currentConversation?.id === conversationId) {
      startNewConversationDraft();
    }
    refreshConversationList();
  } catch (err) {
    showError(
      err instanceof Error ? err.message : "Could not delete conversation."
    );
  }
}

function refreshConversationList() {
  renderConversationList(DB.listConversations());
}

function loadConversation(id) {
  const conv = DB.getConversation(id);
  if (!conv) {
    showError("Conversation not found.");
    refreshConversationList();
    return;
  }
  setCurrentConversation(conv);
  els.existingTargetRadio.checked = true;
  updateTargetUi();
}

function renderBlocks() {
  els.blocksContainer.innerHTML = "";
  const blocks = currentConversation?.blocks ?? [];

  if (blocks.length === 0) {
    const hint = document.createElement("p");
    hint.className = "empty-hint";
    hint.textContent = currentConversation
      ? "No blocks in this conversation yet."
      : "No blocks yet. Paste a conversation below.";
    els.blocksContainer.appendChild(hint);
    return;
  }

  let cumulativeMessages = 0;
  for (const block of blocks) {
    const section = document.createElement("section");
    section.className = "block-group";
    section.dataset.blockId = String(block.id);

    const header = document.createElement("div");
    header.className = "block-header";

    const title = document.createElement("span");
    title.className = "block-title";
    const msgCount = block.messages.filter((m) => (m.text || "").trim()).length;
    cumulativeMessages += msgCount;
    title.textContent = `Block ${block.position + 1} — ${blockCheckpointLabel(cumulativeMessages)}`;

    const removeBtn = document.createElement("button");
    removeBtn.type = "button";
    removeBtn.className = "secondary remove-block-btn";
    removeBtn.textContent = "Remove block";
    removeBtn.addEventListener("click", () => removeBlockHandler(block.id));

    header.append(title, removeBtn);

    const messagesEl = document.createElement("div");
    messagesEl.className = "block-messages";

    for (const msg of block.messages) {
      const row = document.createElement("div");
      row.className = "message-row";

      const author = document.createElement("span");
      author.className = "message-author";
      author.textContent = msg.author;
      author.style.color = colorForAuthor(msg.author);

      const text = document.createElement("span");
      text.className = "message-text";
      text.textContent = msg.text;

      row.append(author, text);
      messagesEl.appendChild(row);
    }

    section.append(header, messagesEl);
    els.blocksContainer.appendChild(section);
  }
}

async function removeBlockHandler(blockId) {
  showError("");
  try {
    if (!(await DB.deleteBlock(blockId))) {
      showError("Block not found.");
      return;
    }
    if (currentConversation) {
      invalidateAnalysis(currentConversation.id);
      loadConversation(currentConversation.id);
    }
    refreshConversationList();
  } catch (err) {
    showError(err instanceof Error ? err.message : "Could not remove block.");
  }
}

async function addBlockHandler() {
  showError("");
  const raw = els.conversationBlock.value;
  if (!raw.trim()) return;

  try {
    let conversationId = currentConversation?.id;

    if (isNewTarget()) {
      const created = await DB.createConversation();
      if (!created?.id) {
        showError("Could not create conversation.");
        return;
      }
      conversationId = created.id;
    }

    if (conversationId == null) {
      showError("Select a conversation or choose “New conversation”.");
      return;
    }

    const result = await DB.addBlock(conversationId, raw);
    if (!result) {
      showError("Conversation not found.");
      return;
    }

    els.conversationBlock.value = "";
    invalidateAnalysis(conversationId);
    setCurrentConversation(result.conversation);
    els.existingTargetRadio.checked = true;
    refreshConversationList();
  } catch (err) {
    showError(err instanceof Error ? err.message : "Could not add block.");
  }
}

function allMessagesFlat() {
  if (!currentConversation?.blocks) return [];
  return currentConversation.blocks.flatMap((b) => b.messages);
}

function renderResult(data, options = {}) {
  if (!options.fromCache && currentConversation?.id != null) {
    analysisByConversationId.set(currentConversation.id, data);
  }
  els.resultPanel.classList.remove("hidden");
  if (els.moreDetails) els.moreDetails.open = false;

  const predatory = data.predatory ?? data.label === 1;
  const verdict = document.createElement("div");
  verdict.className = `verdict ${predatory ? "flagged" : "safe"}`;
  verdict.textContent = predatory
    ? "Flagged — conversation classified as predatory"
    : "Clear — not classified as predatory";

  els.resultSummary.innerHTML = "";
  els.resultSummary.appendChild(verdict);

  const flat = allMessagesFlat();
  if (els.moreContent) {
    els.moreContent.innerHTML = "";
    const stats = [
      ["Score", (data.score ?? 0).toFixed(4)],
      ["Label", String(data.label ?? 0)],
      ["Messages analyzed", String(data.message_count ?? flat.length)],
      [
        "Conversation progress",
        data.progress != null
          ? `${(data.progress * 100).toFixed(1)}%`
          : "—",
      ],
      [
        "Threshold",
        data.threshold != null ? (data.threshold ?? 0).toFixed(4) : "—",
      ],
      ["Model", data.model ?? "—"],
    ];

    for (const [label, value] of stats) {
      const row = document.createElement("div");
      row.className = "stat-row";
      row.innerHTML = `<span>${label}</span><strong>${value}</strong>`;
      els.moreContent.appendChild(row);
    }
  }

  const flagged = data.flagged_messages ?? [];
  if (flagged.length > 0) {
    els.flaggedBlock.classList.remove("hidden");
    els.flaggedList.innerHTML = "";
    for (const item of flagged) {
      const li = document.createElement("li");
      const msg = flat[item.index];
      const who = msg ? msg.author : "Unknown";
      li.textContent = `[${who}] ${item.text} (contribution: ${item.contribution.toFixed(4)})`;
      els.flaggedList.appendChild(li);
    }
  } else {
    els.flaggedBlock.classList.add("hidden");
  }
}

async function analyze() {
  showError("");
  if (!currentConversation?.id) {
    showError("Add at least one block before analyzing.");
    return;
  }

  const flat = allMessagesFlat();
  const counts = blockMessageCounts();
  if (flat.length === 0 || counts.length === 0) {
    showError("No messages to analyze.");
    return;
  }

  els.analyzeBtn.disabled = true;
  els.analyzeBtn.textContent = "Analyzing…";
  const analyzedId = currentConversation.id;

  try {
    const body = await predict(flat, analyzedId, counts);
    analysisByConversationId.set(analyzedId, body);
    if (currentConversation?.id === analyzedId) {
      renderResult(body, { fromCache: true });
    }
  } catch (err) {
    hideResultPanel();
    const message = err instanceof Error ? err.message : "Request failed.";
    showError(message);
  } finally {
    els.analyzeBtn.disabled = !(currentConversation?.blocks?.length);
    els.analyzeBtn.textContent = "Analyze conversation";
  }
}

els.addForm.addEventListener("submit", (e) => {
  e.preventDefault();
  addBlockHandler();
});

els.analyzeBtn.addEventListener("click", analyze);
els.newConvBtn.addEventListener("click", startNewConversationDraft);

document.querySelectorAll('input[name="target"]').forEach((radio) => {
  radio.addEventListener("change", updateTargetUi);
});

async function init() {
  if (!window.ApiClient) {
    showError("api.js failed to load.");
    return;
  }
  if (!DB) {
    showError("Database script failed to load. Check that db.js is present.");
    return;
  }
  if (!SecureTransport) {
    showError("crypto.js failed to load. Encrypted transport is required.");
    return;
  }
  try {
    API_BASE = await window.ApiClient.resolveApiBase();
    SecureTransport.clearSession();
    await DB.initDb();
    loadDisplayTitles();
    await SecureTransport.ensureSession(API_BASE);
    refreshConversationList();
    startNewConversationDraft();
  } catch (err) {
    showError(err instanceof Error ? err.message : "Could not initialize.");
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
