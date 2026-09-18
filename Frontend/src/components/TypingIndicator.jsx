import { Bot } from "lucide-react";
import styles from "./TypingIndicator.module.css";

export default function TypingIndicator() {
  return (
    <div className={styles.wrapper}>
      <div className={styles.avatar}>
        <Bot size={16} />
      </div>
      <div className={styles.bubble}>
        <span className={styles.dot} />
        <span className={styles.dot} />
        <span className={styles.dot} />
      </div>
    </div>
  );
}
