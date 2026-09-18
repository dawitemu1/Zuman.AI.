import { useEffect, useRef } from "react";
import ChatMessage from "./ChatMessage";
import WelcomeScreen from "./WelcomeScreen";
import styles from "./ChatWindow.module.css";

export default function ChatWindow({ messages, loading, onSuggest, lang, onRegenerate }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const visibleMessages = messages.filter((m) => m.role !== "system");

  return (
    <div className={styles.window}>
      {visibleMessages.length === 0 && !loading ? (
        <WelcomeScreen />
      ) : (
        <div className={styles.messages}>
          {visibleMessages.map((m, idx) => {
            // Only the last assistant message gets a regenerate button
            const isLastAi =
              m.role === "assistant" &&
              idx === visibleMessages.length - 1;

            return (
              <ChatMessage
                key={m.id}
                role={m.role}
                content={m.content}
                streaming={m.streaming}
                lang={lang}
                onRegenerate={isLastAi && !m.streaming ? onRegenerate : undefined}
              />
            );
          })}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
