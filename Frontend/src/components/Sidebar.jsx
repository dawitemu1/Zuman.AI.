import { MessageSquarePlus, Trash2, MessageSquare, Bot } from "lucide-react";
import styles from "./Sidebar.module.css";

// Group conversations by Today / Yesterday / This week / Older
function groupConversations(convs) {
  const now   = Date.now();
  const day   = 86_400_000;
  const groups = [
    { label: "Today",      items: [] },
    { label: "Yesterday",  items: [] },
    { label: "This week",  items: [] },
    { label: "Older",      items: [] },
  ];
  for (const c of convs) {
    const age = now - (c.createdAt ?? now);
    if      (age < day)       groups[0].items.push(c);
    else if (age < 2 * day)   groups[1].items.push(c);
    else if (age < 7 * day)   groups[2].items.push(c);
    else                      groups[3].items.push(c);
  }
  return groups.filter((g) => g.items.length > 0);
}

export default function Sidebar({ conversations, activeId, onNew, onSelect, onDelete }) {
  const groups = groupConversations(conversations);

  return (
    <aside className={styles.sidebar}>
      {/* Brand */}
      <div className={styles.brand}>
        <Bot size={20} />
        <span>CBE Local Assistant</span>
      </div>

      {/* New chat */}
      <button className={styles.newBtn} onClick={onNew}>
        <MessageSquarePlus size={15} />
        New chat
      </button>

      {/* Grouped list */}
      <nav className={styles.list}>
        {conversations.length === 0 && (
          <p className={styles.empty}>No conversations yet</p>
        )}

        {groups.map((group) => (
          <div key={group.label} className={styles.group}>
            <p className={styles.groupLabel}>{group.label}</p>
            {group.items.map((c) => (
              <div
                key={c.id}
                className={`${styles.item} ${c.id === activeId ? styles.active : ""}`}
                onClick={() => onSelect(c.id)}
                role="button"
                tabIndex={0}
                onKeyDown={(e) => e.key === "Enter" && onSelect(c.id)}
              >
                <MessageSquare size={13} className={styles.itemIcon} />
                <span className={styles.itemTitle}>{c.title}</span>
                <button
                  className={styles.deleteBtn}
                  onClick={(e) => { e.stopPropagation(); onDelete(c.id); }}
                  aria-label="Delete conversation"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        ))}
      </nav>
    </aside>
  );
}
