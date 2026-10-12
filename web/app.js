// Chat page for server.py. No framework: fetch + a small server-sent-events reader.
const $ = (s, el = document) => el.querySelector(s);
const chat = $("#chat");
const empty = $("#empty");
const form = $("#composer");
const input = $("#input");
const send = $("#send");
const webBox = $("#web");
const agentBox = $("#agent");
const stateEl = $("#server-state");

let history = [];   // [{role, content}] sent back to the server for follow-up questions
let busy = false;
let ready = false;

// ------------------------------------------------------------ preferences
function loadPref(key) { try { return localStorage.getItem(key) === "1"; } catch { return false; } }
function savePref(key, on) { try { localStorage.setItem(key, on ? "1" : "0"); } catch {} }
webBox.checked = loadPref("lawrag.web");
agentBox.checked = loadPref("lawrag.agent");
webBox.addEventListener("change", () => savePref("lawrag.web", webBox.checked));
agentBox.addEventListener("change", () => savePref("lawrag.agent", agentBox.checked));
$("#mode-web").title = "Also search Indian Kanoon judgments and trusted legal websites";
$("#mode-agent").title = "An agent picks its own searches step by step. Slower.";

// ------------------------------------------------------------ server status
async function pollStatus() {
  try {
    const r = await fetch("/api/status");
    const s = await r.json();
    if (s.error) {
      stateEl.dataset.state = "error";
      stateEl.textContent = "Server error";
      stateEl.title = s.error;
      return;
    }
    if (!s.web_available) {
      webBox.checked = false;
      webBox.disabled = true;
      $("#mode-web").classList.add("off");
      $("#mode-web").title = "Web search is not set up on the server";
    }
    if (s.ready) {
      ready = true;
      stateEl.dataset.state = "ready";
      stateEl.textContent = s.model;
      stateEl.title = "Language model running locally";
      updateSend();
      return;
    }
  } catch {
    stateEl.dataset.state = "error";
    stateEl.textContent = "Server offline";
  }
  setTimeout(pollStatus, 1500);
}
pollStatus();

// ------------------------------------------------------------ composer
function updateSend() { send.disabled = busy || !ready || !input.value.trim(); }
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 192) + "px"; }
input.addEventListener("input", () => { autosize(); updateSend(); });
input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); form.requestSubmit(); }
});
form.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = input.value.trim();
  if (!q || busy || !ready) return;
  input.value = ""; autosize();
  ask(q);
});
for (const b of document.querySelectorAll(".examples button")) {
  b.addEventListener("click", () => { if (ready && !busy) ask(b.textContent); });
}
$("#new-chat").addEventListener("click", () => {
  if (busy) return;
  history = [];
  for (const m of chat.querySelectorAll(".msg")) m.remove();
  empty.hidden = false;
  input.focus();
});

// ------------------------------------------------------------ rendering
const nearBottom = () => chat.scrollHeight - chat.scrollTop - chat.clientHeight < 120;
const toBottom = () => { chat.scrollTop = chat.scrollHeight; };

function addUser(text) {
  const el = $("#tpl-user").content.firstElementChild.cloneNode(true);
  $(".bubble", el).textContent = text;
  chat.append(el);
  toBottom();
}

function link(href, text) {
  const a = document.createElement("a");
  a.href = href; a.textContent = text; a.target = "_blank"; a.rel = "noopener";
  return a;
}

function renderSources(box, sources) {
  if (!sources.length) return;
  const statutes = sources.filter((s) => s.kind === "statute").length;
  const other = sources.length - statutes;
  $("summary", box).textContent = `Sources: ${statutes} section${statutes === 1 ? "" : "s"}` +
    (other ? `, ${other} web / case law` : "");
  const ul = $("ul", box);
  for (const s of sources) {
    const li = document.createElement("li");
    const head = document.createElement("div");
    head.className = "src-head";
    const id = document.createElement("span");
    id.className = "src-id"; id.textContent = s.id;
    const badge = document.createElement("span");
    badge.className = "badge" + (s.cited ? " cited" : "");
    badge.textContent = s.cited ? "cited" : s.kind === "statute" ? "not cited" : s.kind;
    head.append(id, link(s.url, s.label), badge);
    li.append(head);
    const links = Object.entries(s.links || {});
    if (links.length) {
      const row = document.createElement("div");
      row.className = "src-links";
      for (const [name, url] of links) row.append(link(url, name.replace(/\s*\(.*\)$/, "")));
      li.append(row);
    }
    if (s.text) {
      const t = document.createElement("div");
      t.className = "src-text"; t.textContent = s.text;
      li.append(t);
    }
    ul.append(li);
  }
  box.hidden = false;
}

// ------------------------------------------------------------ streaming
async function* readEvents(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

async function ask(query) {
  busy = true; updateSend();
  empty.hidden = true;
  addUser(query);
  const mode = agentBox.checked ? "agent" : webBox.checked && !webBox.disabled ? "web" : "default";

  const el = $("#tpl-bot").content.firstElementChild.cloneNode(true);
  const status = $(".status", el), statusText = $(".status-text", el);
  const steps = $(".steps", el), answer = $(".answer", el);
  statusText.textContent = "Starting…";
  chat.append(el);
  toBottom();

  let streamed = "", final = null;
  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, history, web: mode === "web", agent: mode === "agent" }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).error || `Server returned ${r.status}`);
    for await (const { event, data } of readEvents(r)) {
      const stick = nearBottom();
      if (event === "status") statusText.textContent = data.text;
      else if (event === "step") {
        const li = document.createElement("li");
        li.textContent = data.text;
        if (data.thought) {
          const th = document.createElement("span");
          th.className = "thought"; th.textContent = data.thought;
          li.append(th);
        }
        steps.append(li); steps.hidden = false;
      } else if (event === "token") {
        status.hidden = true;
        answer.classList.add("streaming");
        streamed += data.text;
        answer.textContent = streamed;
      } else if (event === "done") final = data;
      else if (event === "error") throw new Error(data.text);
      if (stick) toBottom();
    }
    if (!final) throw new Error("The answer stopped before it was complete.");

    status.hidden = true;
    answer.classList.remove("streaming");
    answer.innerHTML = final.html;  // rendered on the server from markdown with raw HTML disabled
    for (const a of answer.querySelectorAll("a")) { a.target = "_blank"; a.rel = "noopener"; }
    renderSources($(".sources", el), final.sources);
    if (steps.children.length) {
      const n = steps.children.length;
      statusText.textContent = `Researched in ${n} step${n === 1 ? "" : "s"}`;
      status.hidden = false;
      $(".dot", status).style.animation = "none";
    }
    const meta = $(".meta", el);
    $(".time", meta).textContent = final.timings.total_s != null ? `${final.timings.total_s} s` : "";
    for (const b of meta.querySelectorAll(".icon-btn")) {
      b.addEventListener("click", () => {
        for (const o of meta.querySelectorAll(".icon-btn")) o.setAttribute("aria-pressed", String(o === b));
        fetch("/api/feedback", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, up: b.dataset.up === "1", mode, answer: final.raw,
                                 sources: final.sources.map((s) => s.label) }),
        }).catch(() => {});
      });
    }
    meta.hidden = false;
    history.push({ role: "user", content: query }, { role: "assistant", content: final.raw || answer.textContent });
  } catch (err) {
    status.hidden = true;
    answer.classList.remove("streaming");
    const p = document.createElement("p");
    p.className = "error";
    p.textContent = err.message || String(err);
    answer.append(p);
  } finally {
    busy = false; updateSend();
    toBottom();
    input.focus();
  }
}

input.focus();
