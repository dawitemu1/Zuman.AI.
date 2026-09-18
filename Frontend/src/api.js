// ── Config ────────────────────────────────────────────────────
// Use Vite's same-origin proxy in development; FastAPI stays internal on port 8000.
export const BASE_URL = "";

function apiErrorMessage(payload, fallback) {
  const detail = payload?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((item) => item?.msg || item?.type || JSON.stringify(item))
      .join("; ");
  }
  return payload?.message || fallback;
}

// ── Chat (non-streaming fallback) ─────────────────────────────
export async function sendChat(messages, opts = {}) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages,
      model:       opts.model       ?? "banking-model-gemma4-12b",
      temperature: opts.temperature ?? 0.7,
      max_tokens:  opts.max_tokens  ?? 4096,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(apiErrorMessage(err, res.statusText || "Chat request failed"));
  }
  return (await res.json()).reply;
}

// ── Streaming chat ────────────────────────────────────────────
/**
 * Streams the response token by token via SSE.
 *
 * Callbacks:
 *   onToken(string)  — called for every token chunk received
 *   onDone()         — stream finished cleanly
 *   onError(Error)   — upstream or network error
 *
 * Returns an AbortController — call .abort() to stop early.
 */
export function streamChat(messages, { onToken, onDone, onError }, opts = {}) {
  const ctrl = new AbortController();

  (async () => {
    try {
      const res = await fetch(`${BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: ctrl.signal,
        body: JSON.stringify({
          messages,
          model:       opts.model       ?? "banking-model-gemma4-12b",
          temperature: opts.temperature ?? 0.7,
          max_tokens:  opts.max_tokens  ?? 4096,
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(apiErrorMessage(err, res.statusText || "Stream request failed"));
      }

      const reader  = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let   buffer  = "";

      // eslint-disable-next-line no-constant-condition
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // Decode chunk and add to buffer
        buffer += decoder.decode(value, { stream: true });

        // SSE events are delimited by double newline
        const events = buffer.split("\n\n");
        // Last element is incomplete — keep it in buffer
        buffer = events.pop() ?? "";

        for (const event of events) {
          // An SSE event may span multiple lines; grab the data: line
          for (const line of event.split("\n")) {
            const trimmed = line.trim();
            if (!trimmed.startsWith("data:")) continue;

            const raw = trimmed.slice(5).trim();
            if (!raw) continue;

            let payload;
            try { payload = JSON.parse(raw); }
            catch { continue; }

            if (payload.error) throw new Error(payload.error);
            if (typeof payload.token === "string") onToken(payload.token);
            if (payload.done) { onDone(); return; }
          }
        }
      }

      onDone();
    } catch (err) {
      if (err.name === "AbortError") return;   // user hit stop — silence
      onError(err);
    }
  })();

  return ctrl;
}

// ── Text-to-Speech ────────────────────────────────────────────
/**
 * Calls /text-to-speech.
 * Returns a blob URL you can pass to new Audio(url).
 *
 * The gateway TTS supports en / am / or.
 * Tigrinya (ti) and Somali (so) are not supported by the voice gateway,
 * so we fall back to English for those languages.
 */
const TTS_SUPPORTED = new Set(["en", "am", "or"]);

export async function speakText(text, language = "en") {
  if (typeof window === "undefined" || !window.speechSynthesis) return null;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = TTS_SUPPORTED.has(language) ? language : "en-US";
  window.speechSynthesis.cancel();
  await new Promise((resolve, reject) => {
    utterance.onend = resolve;
    utterance.onerror = reject;
    window.speechSynthesis.speak(utterance);
  });
  return null;
}

// ── Speech-to-Speech ──────────────────────────────────────────
export async function sendSpeechToSpeech(audioBlob, sourceLang, targetLang) {
  const form = new FormData();
  form.append("file",            audioBlob, "recording.webm");
  form.append("source_language", sourceLang);
  form.append("target_language", targetLang);

  const res = await fetch(`${BASE_URL}/speech-to-speech`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Speech-to-speech failed");
  }
  const translatedText = res.headers.get("x-translated-text") ?? "";
  return { audioUrl: URL.createObjectURL(await res.blob()), translatedText };
}
