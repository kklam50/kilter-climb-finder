const healthEl = document.getElementById("health");
const formEl = document.getElementById("search-form");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
const selectedEl = document.getElementById("selected");
const selectedClimbEl = document.getElementById("selected-climb");
const pickerEl = document.getElementById("picker");
const pickerListEl = document.getElementById("picker-list");
const resultsEl = document.getElementById("results");
const matchesEl = document.getElementById("matches");
const contextEl = document.getElementById("context-text");

function setHealth(ok) {
  healthEl.textContent = ok ? "server online" : "server unreachable";
  healthEl.className = `health ${ok ? "health--ok" : "health--bad"}`;
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    const data = await res.json();
    setHealth(res.ok && data.engine_loaded);
  } catch {
    setHealth(false);
  }
}

function showStatus(message, kind) {
  statusEl.textContent = message;
  statusEl.className = `status status--${kind}`;
  statusEl.hidden = false;
}

function hideStatus() {
  statusEl.hidden = true;
}

// --- Board rendering -------------------------------------------------------
// board.png: 1600x1236, bolt-on grid at 60px per 8 board units, x=72 at px 800,
// y=152 at px 70. Only the 12x12 region is shown (viewBox crops the rest).
const BOARD_VIEWBOX = "285 35 1030 1180";
const ROLE_COLORS = { 12: "#00dd00", 13: "#00ffff", 14: "#ff00ff", 15: "#ffa500" };

let boardModalEl = null;

function closeBoardModal() {
  if (boardModalEl) boardModalEl.hidden = true;
}

function openBoardModal(svgEl, title) {
  if (!boardModalEl) {
    boardModalEl = document.createElement("div");
    boardModalEl.className = "board-modal";
    boardModalEl.innerHTML = `
      <div class="board-modal__dialog" role="dialog" aria-modal="true">
        <button type="button" class="board-modal__close" aria-label="Close">&times;</button>
        <div class="board-modal__title"></div>
        <div class="board-modal__body"></div>
      </div>`;
    boardModalEl.addEventListener("click", (e) => {
      if (e.target === boardModalEl || e.target.closest(".board-modal__close")) closeBoardModal();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closeBoardModal();
    });
    document.body.appendChild(boardModalEl);
  }
  boardModalEl.querySelector(".board-modal__title").textContent = title;
  const body = boardModalEl.querySelector(".board-modal__body");
  body.replaceChildren(svgEl.cloneNode(true));
  boardModalEl.hidden = false;
}

function boardPx(x, y, mirrored) {
  const bx = mirrored ? 144 - x : x;
  return [800 + (bx - 72) * 7.5, 70 + (152 - y) * 7.5];
}

async function renderBoard(container, climbId, mirrored = false, title = "") {
  try {
    const res = await fetch(`${API_BASE_URL}/climbs/${encodeURIComponent(climbId)}/holds`);
    if (!res.ok) return;
    const holds = await res.json();
    const rings = holds.map((h) => {
      const [cx, cy] = boardPx(h.x, h.y, mirrored);
      return `<circle cx="${cx}" cy="${cy}" r="34" fill="none" stroke="${ROLE_COLORS[h.role_id] ?? "#fff"}" stroke-width="7"/>`;
    }).join("");
    container.innerHTML = `<svg viewBox="${BOARD_VIEWBOX}" class="board">
      <image href="board.png" width="1600" height="1236"/>${rings}</svg>`;
    const svg = container.querySelector("svg");
    svg.addEventListener("click", () => openBoardModal(svg, title));
  } catch (err) {
    console.warn("board render failed", err);
  }
}

function loadThumbnail(container, climbId, mirrored = false, title = "") {
  renderBoard(container.querySelector(".board-thumb"), climbId, mirrored, title);
}

function renderMatches(data) {
  matchesEl.innerHTML = "";

  if (data.matches.length === 0) {
    matchesEl.innerHTML = `<p>No matches found.</p>`;
  }

  for (const match of data.matches) {
    const card = document.createElement("div");
    card.className = "match-card";
    const angleChips = renderAngleChips(match.angles);
    card.innerHTML = `
      <div class="board-thumb"></div>
      <div class="match-card__name">${escapeHtml(match.climb_name ?? match.climb_id)}</div>
      <div class="match-card__angles">${angleChips}</div>
      <div class="match-card__footer">
        <span class="match-card__meta">
          ${match.is_mirrored ? "mirrored · " : ""}${match.matched_window_count} matched windows
        </span>
        <span class="match-card__score">${match.score.toFixed(3)}</span>
      </div>
    `;
    matchesEl.appendChild(card);
    loadThumbnail(card, match.climb_id, match.is_mirrored, match.climb_name ?? match.climb_id);
  }

  contextEl.textContent = data.context;
  resultsEl.hidden = false;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function getFormParams() {
  return {
    mode: document.getElementById("mode").value,
    top_k: document.getElementById("top-k").value,
  };
}

async function lookupClimbsByName(name) {
  const res = await fetch(`${API_BASE_URL}/climbs/lookup?${new URLSearchParams({ name })}`);

  // 404 here means "no climb in the db/index matches" -- not an error case,
  // it's the fork into the (stubbed) unindexed-search flow.
  if (res.status === 404) {
    return null;
  }

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Lookup failed with status ${res.status}`);
  }

  return res.json();
}

async function searchUnindexedClimb(name) {
  showStatus("Climb not found in database — checking alternate search…", "loading");

  try {
    const res = await fetch(`${API_BASE_URL}/climbs/unindexed-search?${new URLSearchParams({ name })}`);
    const data = await res.json();
    showStatus(data.message, "info");
  } catch {
    showStatus("Climb not found, and alternate search is unavailable.", "error");
  }
}

async function fetchRecommendations(climbId) {
  const params = new URLSearchParams(getFormParams());
  const res = await fetch(
    `${API_BASE_URL}/climbs/${encodeURIComponent(climbId)}/recommendations?${params}`
  );

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Request failed with status ${res.status}`);
  }

  return res.json();
}

function renderAngleChips(angles) {
  if (!angles.length) return `<span class="angle-chip">no logged angles</span>`;
  return angles
    .map((a) => `<span class="angle-chip">${a.angle}° · ${escapeHtml(a.grade ?? "ungraded")}</span>`)
    .join("");
}

function renderSelected(candidate) {
  selectedClimbEl.innerHTML = `
    <div class="board-thumb"></div>
    <div class="selected-climb__body">
      <div class="match-card__name">${escapeHtml(candidate.climb_name)}</div>
      <div class="match-card__meta">
        set by ${escapeHtml(candidate.setter_username)} · ${escapeHtml(candidate.created_at)}
      </div>
      <div class="match-card__angles">${renderAngleChips(candidate.angles)}</div>
    </div>
  `;
  loadThumbnail(selectedClimbEl, candidate.climb_id, false, candidate.climb_name);
  selectedEl.hidden = false;
}

async function selectClimb(candidate) {
  pickerEl.hidden = true;
  resultsEl.hidden = true;
  renderSelected(candidate);
  submitBtn.disabled = true;
  showStatus("Loading recommendations…", "loading");

  try {
    const data = await fetchRecommendations(candidate.climb_id);
    hideStatus();
    renderMatches(data);
  } catch (err) {
    showStatus(err.message, "error");
  } finally {
    submitBtn.disabled = false;
  }
}

function renderPicker(candidates) {
  pickerListEl.innerHTML = "";

  for (const candidate of candidates) {
    const option = document.createElement("button");
    option.type = "button";
    option.className = "picker-option";
    option.innerHTML = `
      <span>${escapeHtml(candidate.climb_name)}</span>
      <span class="picker-option__meta">
        set by ${escapeHtml(candidate.setter_username)} · ${escapeHtml(candidate.created_at)}
      </span>
    `;
    option.addEventListener("click", () => selectClimb(candidate));
    pickerListEl.appendChild(option);
  }

  pickerEl.hidden = false;
}

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();

  const climbName = document.getElementById("climb-name").value.trim();
  if (!climbName) return;

  pickerEl.hidden = true;
  resultsEl.hidden = true;
  selectedEl.hidden = true;
  submitBtn.disabled = true;
  showStatus("Looking up climb…", "loading");

  try {
    const candidates = await lookupClimbsByName(climbName);

    if (candidates === null) {
      await searchUnindexedClimb(climbName);
      return;
    }

    if (candidates.length === 1) {
      await selectClimb(candidates[0]);
      return;
    }

    hideStatus();
    renderPicker(candidates);
  } catch (err) {
    showStatus(err.message, "error");
  } finally {
    submitBtn.disabled = false;
  }
});

checkHealth();
