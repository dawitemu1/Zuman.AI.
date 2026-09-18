import { useRef, useEffect, useState } from "react";
import { ArrowUp, Square, Mic, Paperclip, X } from "lucide-react";
import styles from "./ChatInput.module.css";

export default function ChatInput({
  value, onChange, onSend, onStop, onOpenSpeech, loading, disabled,
}) {
  const textareaRef = useRef(null);
  const fileRef     = useRef(null);
  const [files, setFiles] = useState([]);

  // Auto-grow textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(ta.scrollHeight, 200) + "px";
  }, [value]);

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!loading && value.trim()) onSend();
    }
  };

  const handleFile = (e) => {
    const picked = Array.from(e.target.files ?? []);
    if (picked.length) setFiles((prev) => [...prev, ...picked]);
    e.target.value = "";
  };

  const removeFile = (i) => setFiles((prev) => prev.filter((_, idx) => idx !== i));
  const canSend = !disabled && value.trim().length > 0;

  return (
    <div className={styles.wrap}>
      <div className={styles.outer}>

        {/* File chips above box */}
        {files.length > 0 && (
          <div className={styles.chips}>
            {files.map((f, i) => (
              <span key={i} className={styles.chip}>
                <Paperclip size={11} />
                <span className={styles.chipName}>{f.name}</span>
                <button className={styles.chipX} onClick={() => removeFile(i)} aria-label="Remove">
                  <X size={10} />
                </button>
              </span>
            ))}
          </div>
        )}

        {/* ── Single-row pill: 📎  textarea  🎤  ↑ ── */}
        <div className={styles.box}>

          {/* LEFT — paperclip icon */}
          <label className={styles.iconBtn} title="Attach file" aria-label="Attach file">
            <Paperclip size={18} />
            <input
              ref={fileRef}
              type="file"
              multiple
              className={styles.hiddenFile}
              onChange={handleFile}
            />
          </label>

          {/* CENTER — textarea */}
          <textarea
            ref={textareaRef}
            className={styles.textarea}
            placeholder="Ask Selam anything…"
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={disabled}
            rows={1}
            aria-label="Message input"
          />

          {/* RIGHT — mic icon */}
          <button
            className={styles.iconBtn}
            onClick={onOpenSpeech}
            title="Voice input"
            aria-label="Open voice input"
            type="button"
          >
            <Mic size={18} />
          </button>

          {/* RIGHT — send / stop */}
          {loading ? (
            <button
              className={`${styles.sendBtn} ${styles.stopBtn}`}
              onClick={onStop}
              title="Stop"
              aria-label="Stop"
            >
              <Square size={14} fill="currentColor" />
            </button>
          ) : (
            <button
              className={`${styles.sendBtn} ${canSend ? styles.sendActive : ""}`}
              onClick={onSend}
              disabled={!canSend}
              title="Send (Enter)"
              aria-label="Send"
            >
              <ArrowUp size={16} strokeWidth={2.5} />
            </button>
          )}
        </div>

        <p className={styles.hint}>
          Selam may make mistakes. Verify important banking details with CBE directly.
        </p>
      </div>
    </div>
  );
}
