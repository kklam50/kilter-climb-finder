const healthEl = document.getElementById("health");
const formEl = document.getElementById("search-form");
const submitBtn = document.getElementById("submit-btn");
const statusEl = document.getElementById("status");
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

function renderMatches(data) {
  matchesEl.innerHTML = "";

  if (data.matches.length === 0) {
    matchesEl.innerHTML = `<p>No matches found.</p>`;
  }

  for (const match of data.matches) {
    const card = document.createElement("div");
    card.className = "match-card";
    card.innerHTML = `
      <div>
        <div class="match-card__name">${escapeHtml(match.climb_name ?? match.climb_id)}</div>
        <div class="match-card__meta">
          ${escapeHtml(match.climb_grade ?? "unknown grade")}
          · angle ${match.angle ?? "?"}°
          ${match.is_mirrored ? "· mirrored" : ""}
          · ${match.matched_window_count} matched windows
        </div>
      </div>
      <div class="match-card__score">${match.score.toFixed(3)}</div>
    `;
    matchesEl.appendChild(card);
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
    angle_tolerance: document.getElementById("angle-tolerance").value,
    difficulty_tolerance: document.getElementById("difficulty-tolerance").value,
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

async function selectClimb(climbId) {
  pickerEl.hidden = true;
  resultsEl.hidden = true;
  submitBtn.disabled = true;
  showStatus("Loading recommendations…", "loading");

  try {
    const data = await fetchRecommendations(climbId);
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
    option.addEventListener("click", () => selectClimb(candidate.climb_id));
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
  submitBtn.disabled = true;
  showStatus("Looking up climb…", "loading");

  try {
    const candidates = await lookupClimbsByName(climbName);

    if (candidates === null) {
      await searchUnindexedClimb(climbName);
      return;
    }

    if (candidates.length === 1) {
      await selectClimb(candidates[0].climb_id);
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
