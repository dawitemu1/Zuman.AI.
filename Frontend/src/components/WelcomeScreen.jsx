import selamAvatar from "../assets/melaai.jpg";
import styles from "./WelcomeScreen.module.css";

const LANGS = [
  { code: "en", label: "English" },
  { code: "am", label: "አማርኛ" },
  { code: "or", label: "Afaan Oromo" },
  { code: "ti", label: "ትግርኛ" },
  { code: "so", label: "Soomaali" },
];

export default function WelcomeScreen() {
  return (
    <div className={styles.wrapper}>
      <div className={styles.icon}>
        <img src={selamAvatar} alt="Selam" className={styles.avatarImg} />
      </div>

      <h1 className={styles.title}>ሰላም! I'm Selam 👋</h1>

      <p className={styles.subtitle}>
        Your 24/7 <strong>CBE virtual banking assistant</strong>.
      </p>

      {/* Language badges */}
      <div className={styles.langBadges}>
        {LANGS.map((l) => (
          <span key={l.code} className={styles.badge}>{l.label}</span>
        ))}
      </div>
    </div>
  );
}
