#pragma once

// Web UI for diffuse-server — returned as a single HTML page string.
// Includes CSS and JavaScript inline for a self-contained interface.

inline std::string get_web_ui_html() {
    return std::string(R"HTML(<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>diffuse-server</title>
<style>
:root {
    --bg: #0f0f0f;
    --bg-panel: #1a1a1a;
    --bg-input: #242424;
    --bg-bubble-user: #2563eb;
    --bg-bubble-assistant: #262626;
    --bg-bubble-system: #1e293b;
    --text: #e5e5e5;
    --text-muted: #888;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --border: #333;
    --radius: 12px;
    --green: #22c55e;
    --red: #ef4444;
    --yellow: #eab308;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    height: 100vh;
    display: flex;
    overflow: hidden;
}
/* Sidebar */
.sidebar {
    width: 280px;
    background: var(--bg-panel);
    border-right: 1px solid var(--border);
    display: flex;
    flex-direction: column;
    padding: 16px;
    gap: 12px;
    flex-shrink: 0;
    overflow-y: auto;
}
.sidebar h1 {
    font-size: 18px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 4px;
}
.sidebar h1 .dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.model-info {
    background: var(--bg-input);
    border-radius: 8px;
    padding: 12px;
    font-size: 13px;
}
.model-info .label { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px; }
.model-info .value { font-weight: 600; margin-bottom: 8px; }
.model-info .value:last-child { margin-bottom: 0; }
.section-title {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--text-muted);
    margin-top: 8px;
    margin-bottom: 4px;
}
.setting-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 4px 0;
    font-size: 13px;
}
.setting-row label { color: var(--text-muted); }
.setting-row input[type="number"], .setting-row select {
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 4px 8px;
    width: 100px;
    font-size: 13px;
}
.setting-row input[type="range"] {
    width: 120px;
    accent-color: var(--accent);
}
.api-key-row {
    display: flex;
    gap: 4px;
}
.api-key-row input {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: 6px;
    padding: 4px 8px;
    font-size: 13px;
}
.clear-btn {
    background: transparent;
    border: 1px solid var(--border);
    color: var(--text-muted);
    border-radius: 8px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 13px;
    transition: all 0.2s;
}
.clear-btn:hover { border-color: var(--red); color: var(--red); }
/* Main area */
.main {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
}
.chat-header {
    padding: 12px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.chat-header .title { font-weight: 600; font-size: 15px; }
.chat-header .stats { font-size: 12px; color: var(--text-muted); }
.chat-messages {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.message {
    max-width: 80%;
    padding: 12px 16px;
    border-radius: var(--radius);
    font-size: 14px;
    line-height: 1.6;
    white-space: pre-wrap;
    word-break: break-word;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.message.user {
    background: var(--bg-bubble-user);
    color: white;
    align-self: flex-end;
    border-bottom-right-radius: 4px;
}
.message.assistant {
    background: var(--bg-bubble-assistant);
    align-self: flex-start;
    border-bottom-left-radius: 4px;
}
.message.system {
    background: var(--bg-bubble-system);
    color: var(--text-muted);
    font-size: 13px;
    font-style: italic;
    max-width: 90%;
}
.message.error {
    background: rgba(239, 68, 68, 0.15);
    border: 1px solid var(--red);
    color: var(--red);
    font-size: 13px;
}
.message .meta {
    font-size: 11px;
    color: var(--text-muted);
    margin-top: 6px;
}
.diffusion-progress {
    align-self: flex-start;
    background: var(--bg-bubble-assistant);
    border-radius: var(--radius);
    padding: 12px 16px;
    font-size: 13px;
    color: var(--text-muted);
}
.diffusion-progress .bar {
    width: 200px;
    height: 4px;
    background: var(--border);
    border-radius: 2px;
    margin-top: 8px;
    overflow: hidden;
}
.diffusion-progress .bar-fill {
    height: 100%;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.3s ease;
    width: 0%;
}
/* Input area */
.input-area {
    padding: 16px 24px;
    border-top: 1px solid var(--border);
    display: flex;
    gap: 12px;
    align-items: flex-end;
}
.input-area textarea {
    flex: 1;
    background: var(--bg-input);
    border: 1px solid var(--border);
    color: var(--text);
    border-radius: var(--radius);
    padding: 12px 16px;
    font-size: 14px;
    font-family: inherit;
    resize: none;
    min-height: 48px;
    max-height: 200px;
    transition: border-color 0.2s;
}
.input-area textarea:focus {
    outline: none;
    border-color: var(--accent);
}
.input-area button {
    background: var(--accent);
    color: white;
    border: none;
    border-radius: var(--radius);
    padding: 12px 24px;
    font-size: 14px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
}
.input-area button:hover:not(:disabled) { background: var(--accent-hover); }
.input-area button:disabled { opacity: 0.5; cursor: not-allowed; }
/* Scrollbar */
.chat-messages::-webkit-scrollbar, .sidebar::-webkit-scrollbar { width: 6px; }
.chat-messages::-webkit-scrollbar-track, .sidebar::-webkit-scrollbar-track { background: transparent; }
.chat-messages::-webkit-scrollbar-thumb, .sidebar::-webkit-scrollbar-thumb {
    background: var(--border); border-radius: 3px;
}
/* Empty state */
.empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    color: var(--text-muted);
    gap: 8px;
}
.empty-state .icon { font-size: 48px; opacity: 0.3; }
.empty-state .text { font-size: 16px; }
</style>
</head>
<body>
<div class="sidebar">
    <h1><span class="dot"></span> diffuse-server</h1>
    <div class="model-info" id="modelInfo">
        <div class="label">Model</div>
        <div class="value" id="modelName">Loading...</div>
        <div class="label">Parameters</div>
        <div class="value" id="modelParams">—</div>
    </div>

    <div class="section-title">Generation</div>
    <div class="setting-row">
        <label>Max tokens</label>
        <input type="number" id="maxTokens" value="256" min="1" max="4096">
    </div>
    <div class="setting-row">
        <label>Steps</label>
        <input type="number" id="steps" value="16" min="1" max="64">
    </div>
    <div class="setting-row">
        <label>Temperature</label>
        <input type="number" id="temperature" value="0.0" min="0" max="2" step="0.1">
    </div>
    <div class="setting-row">
        <label>Threshold</label>
        <input type="number" id="threshold" value="0.95" min="0" max="1" step="0.05">
    </div>
    <div class="setting-row">
        <label>Remasking</label>
        <select id="remasking">
            <option value="low_confidence" selected>low_confidence</option>
            <option value="random">random</option>
        </select>
    </div>

    <div class="section-title">System Prompt</div>
    <textarea id="systemPrompt" style="background:var(--bg-input);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:8px;font-size:13px;font-family:inherit;resize:vertical;width:100%;height:60px;">You are a helpful assistant.</textarea>

    <div class="section-title">Authentication</div>
    <div class="api-key-row">
        <input type="password" id="apiKey" placeholder="API key (optional)">
    </div>

    <div style="flex:1;"></div>
    <button class="clear-btn" onclick="clearChat()">Clear Chat</button>
</div>

<div class="main">
    <div class="chat-header">
        <div class="title">Diffusion Chat</div>
        <div class="stats" id="statsText">Ready</div>
    </div>
    <div class="chat-messages" id="messages">
        <div class="empty-state" id="emptyState">
            <div class="icon">&#9728;</div>
            <div class="text">Send a message to start generating</div>
        </div>
    </div>
    <div class="input-area">
        <textarea id="inputText" placeholder="Type your message..." rows="1"
            onkeydown="handleKey(event)" oninput="autoResize(this)"></textarea>
        <button id="sendBtn" onclick="sendMessage()">Send</button>
    </div>
</div>

<script>
const API_BASE = window.location.origin;
let chatHistory = [];
let isGenerating = false;

function getAuthHeaders() {
    const key = document.getElementById('apiKey').value.trim();
    const headers = { 'Content-Type': 'application/json' };
    if (key) headers['Authorization'] = 'Bearer ' + key;
    return headers;
}

async function loadModelInfo() {
    try {
        const resp = await fetch(API_BASE + '/v1/models', { headers: getAuthHeaders() });
        const data = await resp.json();
        if (data.data && data.data.length > 0) {
            const m = data.data[0];
            document.getElementById('modelName').textContent = m.id.split('/').pop();
            const meta = m.meta || {};
            document.getElementById('modelParams').textContent =
                `${meta.n_layer || '?'} layers, ${meta.n_embd || '?'} dim, ${meta.n_vocab || '?'} vocab`;
        }
    } catch (e) {
        document.getElementById('modelName').textContent = 'Error loading';
    }
}

function addMessage(role, content, meta) {
    document.getElementById('emptyState')?.remove();
    const div = document.createElement('div');
    div.className = 'message ' + role;
    div.textContent = content;
    if (meta) {
        const m = document.createElement('div');
        m.className = 'meta';
        m.textContent = meta;
        div.appendChild(m);
    }
    document.getElementById('messages').appendChild(div);
    scrollBottom();
    return div;
}

function addProgress() {
    const div = document.createElement('div');
    div.className = 'diffusion-progress';
    div.id = 'progressIndicator';
    div.innerHTML = 'Generating... <div class="bar"><div class="bar-fill" id="progressBar"></div></div>';
    document.getElementById('messages').appendChild(div);
    scrollBottom();
    return div;
}

function removeProgress() {
    document.getElementById('progressIndicator')?.remove();
}

function scrollBottom() {
    const el = document.getElementById('messages');
    el.scrollTop = el.scrollHeight;
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 200) + 'px';
}

function handleKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

function clearChat() {
    chatHistory = [];
    document.getElementById('messages').innerHTML =
        '<div class="empty-state" id="emptyState"><div class="icon">&#9728;</div><div class="text">Send a message to start generating</div></div>';
    document.getElementById('statsText').textContent = 'Ready';
}

async function sendMessage() {
    const input = document.getElementById('inputText');
    const text = input.value.trim();
    if (!text || isGenerating) return;

    isGenerating = true;
    document.getElementById('sendBtn').disabled = true;
    input.value = '';
    autoResize(input);

    addMessage('user', text);
    const progressEl = addProgress();
    const startTime = performance.now();

    const systemPrompt = document.getElementById('systemPrompt').value;

    // Build messages array
    const messages = [];
    if (systemPrompt.trim()) {
        messages.push({ role: 'system', content: systemPrompt });
    }
    for (const m of chatHistory) {
        messages.push(m);
    }
    messages.push({ role: 'user', content: text });

    const body = {
        messages: messages,
        max_tokens: parseInt(document.getElementById('maxTokens').value) || 256,
        n_steps: parseInt(document.getElementById('steps').value) || 16,
        temperature: parseFloat(document.getElementById('temperature').value) || 0.0,
        remasking: document.getElementById('remasking').value,
        threshold: parseFloat(document.getElementById('threshold').value) || 0.95,
        seed: 42,
    };

    try {
        const resp = await fetch(API_BASE + '/v1/chat/completions', {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify(body),
        });

        const data = await resp.json();

        if (!resp.ok) {
            const errMsg = data.error?.message || 'Request failed (HTTP ' + resp.status + ')';
            removeProgress();
            addMessage('error', errMsg);
        } else {
            removeProgress();
            const assistantText = data.choices?.[0]?.message?.content || '(no response)';
            const info = data.diffuse_info || {};
            const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
            const meta = `${info.diffusion_steps || '?'} steps, ${info.tokens_per_second?.toFixed(1) || '?'} tok/s, ${elapsed}s`;

            addMessage('assistant', assistantText, meta);
            chatHistory.push({ role: 'user', content: text });
            chatHistory.push({ role: 'assistant', content: assistantText });

            document.getElementById('statsText').textContent =
                `${info.tokens_per_second?.toFixed(1) || '?'} tok/s`;
        }
    } catch (e) {
        removeProgress();
        addMessage('error', 'Network error: ' + e.message);
    } finally {
        isGenerating = false;
        document.getElementById('sendBtn').disabled = false;
    }
}

// Initialize
loadModelInfo();
document.getElementById('inputText').focus();
</script>
</body>
</html>)HTML");
}
