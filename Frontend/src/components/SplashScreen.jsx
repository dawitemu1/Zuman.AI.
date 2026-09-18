import { useEffect, useState } from "react";
import selamAvatar from "../assets/melaai.jpg";
import styles from "./SplashScreen.module.css";

const WORDS = [
  { text: "I am",       delay: 0.0  },
  { text: "Selam,",     delay: 0.35, highlight: true },
  { text: "CBE",        delay: 0.72, highlight: true },
  { text: "Assistant.", delay: 1.05, highlight: true },
  { text: "What can",   delay: 1.55 },
  { text: "I help",     delay: 1.85 },
  { text: "you?",       delay: 2.15 },
];

export default function SplashScreen({ onDone }) {
  const [phase, setPhase] = useState("in");

  useEffect(() => {
    const t = setTimeout(() => setPhase("out"), 3400);
    return () => clearTimeout(t);
  }, []);

  const handleTransitionEnd = () => {
    if (phase === "out") onDone?.();
  };

  return (
    <div
      className={`${styles.splash} ${phase === "out" ? styles.fadeOut : styles.fadeIn}`}
      onTransitionEnd={handleTransitionEnd}
    >
      <span className={`${styles.orb} ${styles.orb1}`} />
      <span className={`${styles.orb} ${styles.orb2}`} />
      <span className={`${styles.orb} ${styles.orb3}`} />
      <span className={`${styles.orb} ${styles.orb4}`} />

      <div className={styles.avatarWrap}>
        <span className={styles.ring1} />
        <span className={styles.ring2} />
        <img src={selamAvatar} alt="Selam" className={styles.avatar} />
      </div>

      <p className={styles.sentence}>
        {WORDS.map(({ text, delay, highlight }) => (
          <span
            key={text}
            className={`${styles.word} ${highlight ? styles.highlight : ""}`}
            style={{ animationDelay: `${delay}s` }}
          >
            {text}
          </span>
        ))}
      </p>

      <p className={styles.sub}>Powered by the local CBE fine-tuned model</p>
    </div>
  );
}
