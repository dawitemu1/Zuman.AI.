import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css";
import {
  Copy, Check, Volume2, Loader,
  RefreshCw, ThumbsUp, ThumbsDown,
} from "lucide-react";
import { useState, useRef } from "react";
import { speakText } from "../api";
import melaaiAvatar from "../assets/melaai.jpg";
import styles from "./ChatMessage.module.css";

// ── Code block with language label + copy button ──────────────
function CodeBlock({ className, children }) {
  const [copied, setCopied] = useState(false);
  const lang = /language-(\w+)/.exec(className || "")?.[1] ?? "code";
  const code = String(children).replace(/\n$/, "");

  const copy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div className={styles.codeWrapper}>
      <div className={styles.codeHeader}>
        <span className={styles.codeLang}>{lang}</span>
        <button className={styles.codeCopyBtn} onClick={copy} title="Copy code">
          {copied ? <><Check size={12} /> Copied</> : <><Copy size={12} /> Copy code</>}
        </button>
      </div>
      <pre className={styles.codeBlock}>
        <code className={className}>{children}</code>
      </pre>
    </div>
  );
}

// ── Markdown renderer ─────────────────────────────────────────
function MdContent({ content }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        // Code: block vs inline
        code({ node, inline, className, children, ...props }) {
          if (inline) {
            return <code className={styles.inlineCode} {...props}>{children}</code>;
          }
          return <CodeBlock className={className}>{children}</CodeBlock>;
        },
        // Block elements
        p:          ({ children }) => <p className={styles.p}>{children}</p>,
        h1:         ({ children }) => <h1 className={styles.h1}>{children}</h1>,
        h2:         ({ children }) => <h2 className={styles.h2}>{children}</h2>,
        h3:         ({ children }) => <h3 className={styles.h3}>{children}</h3>,
        h4:         ({ children }) => <h4 className={styles.h4}>{children}</h4>,
        ul:         ({ children }) => <ul className={styles.ul}>{children}</ul>,
        ol:         ({ children }) => <ol className={styles.ol}>{children}</ol>,
        li:         ({ children }) => <li className={styles.li}>{children}</li>,
        strong:     ({ children }) => <strong className={styles.strong}>{children}</strong>,
        em:         ({ children }) => <em className={styles.em}>{children}</em>,
        del:        ({ children }) => <del className={styles.del}>{children}</del>,
        blockquote: ({ children }) => <blockquote className={styles.blockquote}>{children}</blockquote>,
        hr:         ()             => <hr className={styles.hr} />,
        a:          ({ href, children }) => (
          <a href={href} className={styles.link} target="_blank" rel="noopener noreferrer">{children}</a>
        ),
        // Tables (GFM)
        table: ({ children }) => (
          <div className={styles.tableWrap}>
            <table className={styles.table}>{children}</table>
          </div>
        ),
        thead: ({ children }) => <thead className={styles.thead}>{children}</thead>,
        th:    ({ children }) => <th className={styles.th}>{children}</th>,
        td:    ({ children }) => <td className={styles.td}>{children}</td>,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

// ── Main component ────────────────────────────────────────────
export default function ChatMessage({ role, content, streaming, lang, onRegenerate }) {
  const isUser = role === "user";

  const [copied,   setCopied]   = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [ttsErr,   setTtsErr]   = useState("");
  const [liked,    setLiked]    = useState(null);
  const audioRef = useRef(null);

  const handleCopy = () => {
    navigator.clipboard.writeText(content).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const handleSpeak = async () => {
    if (speaking) {
      audioRef.current?.pause();
      audioRef.current = null;
      setSpeaking(false);
      return;
    }
    setTtsErr("");
    setSpeaking(true);
    try {
      await speakText(content, lang ?? "en");
      setSpeaking(false);
    } catch (err) {
      setSpeaking(false);
      setTtsErr(err.message);
    }
  };

  const handleLike    = () => setLiked((v) => v === "up"   ? null : "up");
  const handleDislike = () => setLiked((v) => v === "down" ? null : "down");

  // ═══ USER bubble ═════════════════════════════════════════
  if (isUser) {
    return (
      <div className={styles.userRow}>
        <div className={styles.userBubble}>
          <p className={styles.plainText}>{content}</p>
          <div className={styles.userActions}>
            <button className={styles.miniBtn} onClick={handleCopy} title="Copy">
              {copied ? <Check size={12} /> : <Copy size={12} />}
            </button>
          </div>
        </div>
      </div>
    );
  }

  // ═══ AI message ══════════════════════════════════════════
  return (
    <div className={styles.aiRow}>
      <div className={`${styles.aiAvatar} ${streaming ? styles.avatarSpin : ""} ${streaming ? styles.avatarBottom : styles.avatarTop}`}>
        <img src={melaaiAvatar} alt="Selam" className={styles.avatarImg} />
      </div>

      <div className={styles.aiContent}>
        <div className={styles.aiBubble}>
          {streaming && content === "" ? (
            <span className={styles.thinking}>
              <span className={styles.thinkDot} />
              <span className={styles.thinkDot} />
              <span className={styles.thinkDot} />
              <span className={styles.thinkLabel}>Thinking…</span>
            </span>
          ) : (
            <>
              <MdContent content={content} />
              {streaming && <span className={styles.cursor} aria-hidden="true" />}
            </>
          )}
        </div>

        {/* Action bar — shown once streaming is done */}
        {!streaming && (
          <div className={styles.aiActions}>
            <button className={styles.actionBtn} onClick={handleCopy} title="Copy response">
              {copied ? <Check size={13} /> : <Copy size={13} />}
            </button>

            {onRegenerate && (
              <button className={styles.actionBtn} onClick={onRegenerate} title="Regenerate">
                <RefreshCw size={13} />
              </button>
            )}

            <button
              className={`${styles.actionBtn} ${liked === "up" ? styles.liked : ""}`}
              onClick={handleLike}
              title="Good response"
            >
              <ThumbsUp size={13} />
            </button>

            <button
              className={`${styles.actionBtn} ${liked === "down" ? styles.disliked : ""}`}
              onClick={handleDislike}
              title="Bad response"
            >
              <ThumbsDown size={13} />
            </button>

            <button
              className={`${styles.actionBtn} ${speaking ? styles.speaking : ""}`}
              onClick={handleSpeak}
              title={speaking ? "Stop" : "Read aloud"}
            >
              {speaking
                ? <Loader size={13} className={styles.spin} />
                : <Volume2 size={13} />}
            </button>

            {ttsErr && <span className={styles.ttsErr}>{ttsErr}</span>}
          </div>
        )}
      </div>
    </div>
  );
}
