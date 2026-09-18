import { useState, useRef } from "react";
import { Mic, MicOff, Play, Loader, X, Volume2, Languages } from "lucide-react";
import styles from "./SpeechPanel.module.css";
import { BASE_URL } from "../api";

// Language labels for display
const LANG_LABEL = { en: "English", am: "Amharic", or: "Oromo", auto: "Auto" };

// Send audio to /speech-to-speech with specified source/target languages
async function speechToSpeech(audioBlob, source = "auto", target = "auto") {
  const form = new FormData();
  form.append("file",            audioBlob, "recording.webm");
  form.append("source_language", source);
  form.append("target_language", target);

  const res = await fetch(`${BASE_URL}/speech-to-speech`, {
    method: "POST",
    body:   form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Speech request failed");
  }

  const translatedText = res.headers.get("x-translated-text")  ?? "";
  const sourceLang     = res.headers.get("x-source-language")  ?? "";
  const targetLang     = res.headers.get("x-target-language")  ?? "";
  const audioUrl       = URL.createObjectURL(await res.blob());

  return { audioUrl, translatedText, sourceLang, targetLang };
}

export default function SpeechPanel({ onClose }) {
  const [recording, setRecording] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl,  setAudioUrl]  = useState(null);
  const [loading,   setLoading]   = useState(false);
  const [result,    setResult]    = useState(null);   // { audioUrl, translatedText, sourceLang, targetLang }
  const [error,     setError]     = useState("");
  const [sourceLang, setSourceLang] = useState("auto");
  const [targetLang, setTargetLang] = useState("auto");

  const mediaRef  = useRef(null);
  const chunksRef = useRef([]);
  const outRef    = useRef(null);

  // ── Record ────────────────────────────────────────────────
  const startRecording = async () => {
    setError(""); setResult(null); setAudioBlob(null); setAudioUrl(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      chunksRef.current = [];
      mr.ondataavailable = (e) => chunksRef.current.push(e.data);
      mr.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        setAudioBlob(blob);
        setAudioUrl(URL.createObjectURL(blob));
        stream.getTracks().forEach((t) => t.stop());
      };
      mr.start();
      mediaRef.current = mr;
      setRecording(true);
    } catch {
      setError("Microphone access denied. Please allow microphone in your browser.");
    }
  };

  const stopRecording = () => {
    mediaRef.current?.stop();
    setRecording(false);
  };

  // ── File upload ───────────────────────────────────────────
  const handleFile = (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setAudioBlob(file);
    setAudioUrl(URL.createObjectURL(file));
    setResult(null); setError("");
  };

  // ── Send ──────────────────────────────────────────────────
  const handleSend = async () => {
    if (!audioBlob) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const res = await speechToSpeech(audioBlob, sourceLang, targetLang);
      setResult(res);
      // Auto-play at max volume
      if (outRef.current) {
        outRef.current.volume = 1.0;
        outRef.current.play().catch(() => {});
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={styles.overlay} role="dialog" aria-modal="true" aria-label="Speech">
      <div className={styles.panel}>

        {/* ── Header ── */}
        <div className={styles.header}>
          <Volume2 size={18} />
          <h2 className={styles.title}>Speak &amp; Listen</h2>
          <button className={styles.closeBtn} onClick={onClose} aria-label="Close"><X size={18} /></button>
        </div>

        {/* ── Auto-detect badge ── */}
        <div className={styles.autoBadge}>
          <Languages size={14} />
          Language auto-detected from your speech — English, Amharic &amp; Oromo supported
        </div>

        {/* ── Language selectors (in-modal; prevents redirect) ── */}
        <div className={styles.langRow}>
          <label className={styles.langLabel}>
            Source:
            <select value={sourceLang} onChange={(e) => setSourceLang(e.target.value)} className={styles.langSelect}>
              <option value="auto">Auto</option>
              <option value="en">English</option>
              <option value="am">Amharic</option>
              <option value="or">Oromo</option>
            </select>
          </label>
          <label className={styles.langLabel}>
            Target:
            <select value={targetLang} onChange={(e) => setTargetLang(e.target.value)} className={styles.langSelect}>
              <option value="auto">Auto</option>
              <option value="en">English</option>
              <option value="am">Amharic</option>
              <option value="or">Oromo</option>
            </select>
          </label>
        </div>

        {/* ── Record / Stop ── */}
        <div className={styles.recordRow}>
          {recording ? (
            <button className={`${styles.btn} ${styles.stopRec}`} onClick={stopRecording}>
              <MicOff size={17} /> Stop recording
            </button>
          ) : (
            <button className={`${styles.btn} ${styles.startRec}`} onClick={startRecording}>
              <Mic size={17} /> {audioBlob ? "Re-record" : "Record"}
            </button>
          )}
          {recording && <span className={styles.pulse}>● Recording…</span>}
        </div>

        {/* ── File upload ── */}
        <div className={styles.orRow}>
          <span className={styles.orText}>or upload a file</span>
          <label className={styles.fileLabel}>
            Choose file
            <input type="file" accept="audio/*" className={styles.fileInput} onChange={handleFile} />
          </label>
        </div>

        {/* ── Input preview ── */}
        {audioUrl && !recording && (
          <div className={styles.previewBox}>
            <span className={styles.previewLabel}>Your recording</span>
            <audio src={audioUrl} controls className={styles.audio} />
          </div>
        )}

        {/* ── Send button ── */}
        {audioBlob && !recording && (
          <button
            className={`${styles.btn} ${styles.sendBtn}`}
            onClick={handleSend}
            disabled={loading}
          >
            {loading
              ? <><Loader size={15} className={styles.spin} /> Detecting &amp; translating…</>
              : <><Play size={15} /> Detect language &amp; Speak</>}
          </button>
        )}

        {/* ── Error ── */}
        {error && <p className={styles.error}>⚠ {error}</p>}

        {/* ── Result ── */}
        {result && (
          <div className={styles.resultBox}>
            {/* Show detected language */}
            {result.sourceLang && (
              <p className={styles.detectedLang}>
                Detected: <strong>{LANG_LABEL[result.sourceLang] ?? result.sourceLang}</strong>
                {result.targetLang && result.targetLang !== result.sourceLang &&
                  <> → <strong>{LANG_LABEL[result.targetLang] ?? result.targetLang}</strong></>}
              </p>
            )}
            {result.translatedText && (
              <p className={styles.translatedText}>"{result.translatedText}"</p>
            )}
            <span className={styles.previewLabel}>Response</span>
            <audio
              ref={outRef}
              src={result.audioUrl}
              controls
              className={styles.audio}
            />
          </div>
        )}
      </div>
    </div>
  );
}
