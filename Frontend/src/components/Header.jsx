import { Trash2, Sun, Moon } from "lucide-react";
import styles from "./Header.module.css";

export default function Header({ title, theme, onClear, onToggleSidebar, onToggleTheme }) {
  const isDark = theme === "dark";
  return (
    <header className={styles.header}>
      {/* Hamburger */}
      <button className={styles.menuBtn} onClick={onToggleSidebar} aria-label="Toggle sidebar">
        <span className={styles.bar} />
        <span className={styles.bar} />
        <span className={styles.bar} />
      </button>

      <h2 className={styles.title}>{title || "New Chat"}</h2>

      <div className={styles.actions}>
        {/* Theme toggle */}
        <button
          className={styles.iconBtn}
          onClick={onToggleTheme}
          title={isDark ? "Switch to light mode" : "Switch to dark mode"}
          aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
        >
          {isDark ? <Sun size={16} /> : <Moon size={16} />}
        </button>

        {/* Clear conversation */}
        <button
          className={`${styles.iconBtn} ${styles.dangerBtn}`}
          onClick={onClear}
          title="Clear conversation"
          aria-label="Clear conversation"
        >
          <Trash2 size={16} />
        </button>
      </div>
    </header>
  );
}
