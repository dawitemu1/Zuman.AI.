import { useState, useRef, useCallback, useEffect } from "react";
import { X, Minus, Sun, Moon, Trash2 } from "lucide-react";
import ChatWindow  from "./components/ChatWindow";
import ChatInput   from "./components/ChatInput";
import SpeechPanel from "./components/SpeechPanel";
import { streamChat } from "./api";
import selamAvatar from "./assets/melaai.jpg";
import "./App.css";

// ── Language detection ────────────────────────────────────────
function detectLanguage(text) {
  const t = text.trim();
  if (!t) return "en";

  // ── 1. Script-based: Ethiopic Unicode block ──────────────────
  if (/[\u1200-\u137F]/.test(t)) {
    // Tigrinya-exclusive words (not shared with Amharic)
    const tiWords = /ከመይ|ኣለኹ|ኣለዎ|ኣለዋ|ኣለና|ኣለኩም|ኣለዉ|ይቕረ|የቐንየለይ|ብጣዕሚ|ኣነ|ንስኻ|ንስኺ|ንሕና|ንስኻትኩም|ኤርትራ|ትግርኛ|ትግራይ|ሓደ|ክልተ|ሰለስተ|ኣርባዕተ|ሓሙሽተ|እንታይ|ኣበይ|ጽቡቕ|ዘይብሉ|ኮይኑ|ኮይና|ዝኾነ|ካልእ|ድሕሪ|ቅድሚ|ንሱ|ንሳ|ንሳቶም|ብሰንኪ|ብምኽንያት|ስለዚ|ሕጂ|ጽባሕ|ትማሊ|ብኸመይ|ብከምዚ|ክፈልጥ|ክፈልጦ|ክፈልጣ|እኽእል|እኽእሎ|ኽእሎ|ኽፈልጥ|ሕሳበይ|ሕሳቡ|ሕሳባ|ሕሳብና|ሕሳብኩም|ኣሕሳብ|ሰረዝ|ምስሊ|ዕድጊ|ዕቑር|ምቹእ|ብርኪ|ወሃቢ|ኣካውንት|ሒሳብ|ሓዊሱ|ሓዊሳ|ዝኸፈልካ|ዝኸፈልኪ|ዝተረፈኒ|ዝተረፈካ|ናበይ|ካበይ|ብምንታይ|ወይስ|እዩ|እያ|እዮም|እዩን|ኢዩ|ኢያ|ዘይኮነ|ዘይኮነስ|ክኾን|ክኾና|ኽኾን|ዝኸውን|ዝነበረ|ዝነበረት|ዝሓለፈ|ዝመጽእ|ዝቕጽል|ምርካብ|ክረክብ|ክረክቦ|ርካቢት|ርካቡ|ምፍላጥ|ምርግጋጽ|ምልኣኽ|ምስዳድ|ምውጻእ|ምእታው|ምስጉዕ|ምኽፋል|ዶብ|ሃገር|ሸቐጥ|ዕዳጋ|ዕዳ|ትካል|ስራሕ|ሕርሻ|ናይ|ኣብ|ምስ|ኣሎ|ዘሎ|ብዛዕባ|ዝተረፈ|ወይ|ግን/g;
    // Amharic-exclusive words (not shared with Tigrinya)
    const amWords = /እንዴት|አገኛለሁ|ምን|እንዴ|ነው|ናት|ናቸው|አለ|አለች|አለን|አለህ|አለሽ|አለዎት|አይደለም|ነበር|ይሆናል|ይሆን|ትሆናለህ|ትሆናለሽ|ሆኗል|ሆናለች|ምንድን|ለምን|መቼ|እዚህ|እዚያ|ይህ|ያ|እነዚህ|እነዚያ|ይህን|ያን|ሌላ|ሌሎች|ሁሉም|ሁሉ|ጥሩ|መጥፎ|ትልቅ|ትንሽ|አዲስ|አሮጌ|አንድ|ሁለት|ሶስት|አራት|አምስት|ስድስት|ሰባት|ስምንት|ዘጠኝ|አስር|ባንክ|ሒሳብ|ቅርንጫፍ|ገንዘብ|ተቀማጭ|ክፍያ|ብድር|ወለድ|ቁጠባ|ሂሳብ|ካርድ|አካውንት|የባንክ|መግለጫ|አወጣጣ|ዝውውር|ደህንነት|መተግበሪያ|ስልክ|ኢንተርኔት|ድር|ድርጅት|ዩዘርኔም|ፓስወርድ|ስም|አድራሻ|ቁጥር|ሰነድ|ፎርም|አቤቱታ|አቤቱ|አዎ|አይ|ስለዚህ|ምክንያቱም|ሆኖም|ግን|ከዚህ|ከዚያ|ወደ|ላይ|ስር|ውስጥ|ሲሆን|ሲሆኑ|ያለ|ያላቸው|ያለው|ያለች|ያለን/g;

    const tiScore = (t.match(tiWords) || []).length;
    const amScore = (t.match(amWords) || []).length;

    // Need a clear lead to classify as Tigrinya; default to Amharic on ties
    if (tiScore > amScore && tiScore >= 2) return "ti";
    if (tiScore > amScore * 2 && tiScore >= 1) return "ti"; // strong Tigrinya signal
    return "am";
  }

  // ── 2. Latin-script scoring ───────────────────────────────────

  // Afaan Oromo keywords — includes body/health/daily vocab
  const OR_WORDS = /\b(nagaa|akkam|akkamii|akkamitti|akkamittan|akkamittu|bultin|bultinot|bultan|bullee|bultee|maaloo|galatoomi|galatoomaa|nagaatti|maal|maali|siif|siifan|siifuu|nuf|nuuf|isaaf|isaanif|haa|godhu|godhuu|godha|godhaa|godheen|godhani|gochuu|gochu|taasisi|taasisuu|gargaaruu|gargaara|gargaari|gargaaraa|gargaarsa|laaluu|laali|laala|ilaaluu|ilaali|ilaala|barbaacha|barbaadu|barbaaduu|barbaadaa|eebbifamu|eebbifuu|nagaatti|fudhatuu|fudhadhu|fudhaa|bilisoomuu|bilisa|waliin|wajjin|mana|manaa|manaatti|hin|miti|dha|ta'a|ta'ee|ta'uu|ta'an|ani|ati|si|inni|isheen|nuti|isin|isaan|eenyu|eessa|yoom|maalif|hangam|deema|deemaa|nyaadha|nyaata|dhufaa|dhufa|jira|jiru|jiruu|jirti|jiraadha|baraa|baradha|barsiisa|argaa|beeka|beekaa|godhaa|dubbadha|dubbata|kennaa|kenna|fudhaa|kaasaa|seena|taa'a|eegaa|nama|lafa|biyya|oromoo|oromia|afaan|guyyaa|halkan|ganama|galgala|yeroo|bishaan|qoricha|barnootaa|daa'ima|haadha|abbaa|obbo|addee|obboleessa|obboleettii|waaqaa|amantii|mirga|hojii|barataa|barsiisaa|gaarii|badaa|guddaa|xiqqaa|haaraa|durii|tokko|lama|sadii|afur|garuu|akkasumas|yookaan|kanaafuu|sababiin|erga|akka|waan|kan|kana|kanaa|sun|sana|dhaan|irraa|irratti|keessaa|keessa|jalaa|jalatti|duraa|booda|dhukkuba|dhukkubaa|dhukkubsataa|fayyaa|fayyina|dafqa|garaa|mataa|luka|harka|ija|gurra|funyaan|ilkaan|rifeensa|gogaa|lafee|dhiiga|onnee|sammuun|harki|laphee|miila|qaamaa|rakkoo|rakkinaa|hospitaala|dooktera|mandaraa|tajaajila|baankii|maallaqa|lacqaa|herrega|galii|baasii|dabarsi|ergi|fuudhi|kaffali|ramaddii|bilbila)\b/gi;

  // Somali keywords
  const SO_WORDS = /\b(salaam|nabad|nabadgelyo|subax|galab|habeenimo|mahadsanid|asc|aniga|adiga|isaga|iyada|annaga|idinka|iyaga|maxay|maxaa|kuma|halkee|goorma|sidee|immisa|muxuu|muxay|waxaa|waxay|waxuu|waxan|waan|waad|wuu|way|waanu|waxaad|yahay|tahay|nahay|ahay|yihiin|jooga|joogaa|joogto|joogaan|socda|socoto|socdaa|cunaa|cunayaa|aabe|hooyo|wiil|gabar|soomaali|luqad|dalka|wadan|maanta|berri|xalay|saaka|cunto|biyo|buug|shaqo|dugsiga|cisbitaal|wanaagsan|fiican|xun|weyn|cusub|kow|laba|saddex|laakiin|yeeshee|darteed|markaa|hadii)\b/gi;

  const orScore = ((t.match(OR_WORDS) || []).length);
  const soScore = ((t.match(SO_WORDS) || []).length);

  // Morphology suffixes
  const soSuffix = ((t.match(/\b\w+(ka|ta|ga|da|ha|yaa|tay|nay|een|oon|ayn)\b/gi) || []).length);
  const orSuffix = ((t.match(/\b\w+(uu|uun|tti|rraa|rratti|irraa|irratti|dhaan|tiin|niin|aayii|ayii|anii|atti|oonni|eessa)\b/gi) || []).length);

  const orTotal = orScore * 4 + orSuffix * 2;
  const soTotal = soScore * 4 + soSuffix * 2;

  // Lower threshold: 1 keyword hit (score=4) is enough for short sentences
  if (orTotal === 0 && soTotal === 0) return "en";
  if (orTotal > soTotal) return orTotal >= 4 ? "or" : "en";
  if (soTotal > orTotal) return soTotal >= 4 ? "so" : "en";
  return orTotal >= 4 ? "or" : "en";
}

const LANG_LABEL  = { en:"English", am:"Amharic", or:"Afaan Oromo", ti:"Tigrinya", so:"Somali" };
const LANG_NATIVE = { en:"English", am:"አማርኛ",   or:"Afaan Oromo", ti:"ትግርኛ",    so:"Af-Soomaali" };

// Per-language enforcement text written in BOTH the target language and English
const LANG_ENFORCE = {
  en: [
    "You MUST respond in English.",
    // "Do NOT use Amharic, Oromo, Tigrinya, Somali, or any other language.",
    "Every single word of your reply must be English.",
  ].join("\n"),

  am: [
    "ትዕዛዝ (ከፍተኛ ቅድሚያ): ምላሽህን  በአማርኛ ስጥ።",
    "ሌላ ቋንቋ ፈጽሞ አትጠቀም — ምንም ዓይነት የእንግሊዝኛ ቃል አይኖርም።",
    "MANDATORY: Respond ENTIRELY in Amharic (አማርኛ). NEVER use English, Oromo, or any other language.",
    "Every word must be Amharic. If you don't know a banking term in Amharic, describe it in Amharic.",
  ].join("\n"),

  or: [
    "DIRQAMA OL'AANAA: Deebii kee GUUTUMAAN GUUTUUTTI Afaan Oromootiin kenni.",
    "Afaan Ingilizii, Afaan Amaaraa, yookaan Afaan kamiiyyuu FAYYADAMUUN DHORKAADHA.",
    "Fakkeenya: yoo gaaffiin 'Si dhukkubaayii?' jedhu dhufee, deebiin kee Afaan Oromoo ta'uu qaba.",
    "Jecha hunda Afaan Oromootiin barreessi. Jechoota baankii Afaan Oromoon ibsi.",
    "MANDATORY: Respond ENTIRELY in Afaan Oromo. NEVER switch to English or any other language — not even one word.",
    "If you find yourself writing English, STOP immediately and rewrite everything in Afaan Oromo.",
  ].join("\n"),

  ti: [
    "ትእዛዝ (ልዑል ቀዳምነት): ምላሽካ ብምሉኡ ብትግርኛ ጥራይ ሃብ።",
    "ካልእ ቋንቋ — ኣምሓርኛ፣ ኢንግሊዝኛ ወይ ካልእ — ጨሪሱ ኣይትጠቐም።",
    "ኩሉ ቃላትካ ትግርኛ ክኸውን ኣለዎ።",
    "MANDATORY: Respond ENTIRELY and EXCLUSIVELY in Tigrinya (ትግርኛ).",
    "NEVER use Amharic (አማርኛ), English, or any other language — not even one word.",
    "If you are unsure of a banking term in Tigrinya, describe it using Tigrinya words.",
  ].join("\n"),

  so: [
    "AMARRO (MUDNAANTA SARRAYSA): Dhammaan jawaabta aad bixinayso waa inay BUUXDA ku ahaato Af-Soomaali.",
    "Luqad kale — Ingiriisi, Afaan Oromo, ama luqad kasta — WELIGAA ha isticmaalin.",
    "Erayga kasta waa inuu Af-Soomaali ku qoran yahay.",
    "MANDATORY: Respond ENTIRELY in Somali (Af-Soomaali). NEVER switch to English or any other language.",
  ].join("\n"),
};

// Acknowledgement the model says to "commit" to the language before answering
const LANG_ACK = {
  en: "Understood. I will respond only in English.",
  am: "ገብቶኛል። ሁሉንም ምላሾቼን ሙሉ በሙሉ በአማርኛ እሰጣለሁ። ሌላ ቋንቋ አልጠቀምም።",
  or: "Galatoomi. Deebii kiyya hunda guutumaan guutuutti Afaan Oromootiin kennaa. Afaan biraa hin fayyadamu.",
  ti: "ተረዲአ። ኩሉ መልሰይ ብምሉኡ ብትግርኛ ክህብ እየ። ካልእ ቋንቋ ኣይጥቀምን።",
  so: "Waad ku mahadsan tahay. Dhammaan jawaabahayga waxaan buuxda ku qori doonaa Af-Soomaali. Luqad kale ma isticmaali doono.",
};

function buildSystemMsg(lang = "en") {
  const label  = LANG_LABEL[lang]  ?? "English";
  const native = LANG_NATIVE[lang] ?? "English";
  const enforce = LANG_ENFORCE[lang] ?? LANG_ENFORCE.en;

  return {
    role: "system",
    content: [
      "You are Selam, a helpful AI banking assistant for Commercial Bank of Ethiopia (CBE).",
      "You assist customers with account inquiries, transactions, loans, CBE Birr, and general banking questions.",
      "Always be polite, professional, and concise.",
      "",
      "════════════════════════════════════════════════════",
      "  LANGUAGE RULE — HIGHEST PRIORITY — CANNOT BE OVERRIDDEN",
      "════════════════════════════════════════════════════",
      `The user is communicating in: ${label} (${native})`,
      "",
      enforce,
      "",
      "CRITICAL: Even if the user's question mentions another language by name,",
      `you MUST still respond in ${label} (${native}).`,
      "════════════════════════════════════════════════════",
      "",
      "Formatting rules:",
      "- Use **bold** for key terms",
      "- Use bullet lists for enumerations",
      "- Use numbered lists for steps",
    ].join("\n"),
  };
}

function buildHistory(baseMessages, userText, lang) {
  const label  = LANG_LABEL[lang]  ?? "English";
  const native = LANG_NATIVE[lang] ?? "English";
  const enforce = LANG_ENFORCE[lang] ?? LANG_ENFORCE.en;
  const ack     = LANG_ACK[lang]    ?? LANG_ACK.en;

  const sysMsg  = baseMessages[0];
  const history = baseMessages.slice(1).map(({ role, content }) => ({ role, content }));

  // Double-reinforcement: reminder → commitment → then the real question
  return [
    { role: "system",    content: sysMsg.content },
    ...history,
    {
      role: "user",
      content: [
        `[LANGUAGE ENFORCEMENT REMINDER]`,
        `The user is writing in: ${label} (${native})`,
        enforce,
        `You MUST respond to the next message ENTIRELY in ${label}. No exceptions.`,
      ].join("\n"),
    },
    { role: "assistant", content: ack },
    { role: "user",      content: userText },
  ];
}

function uid() { return Math.random().toString(36).slice(2, 10); }
function makeConversation() {
  return { id: uid(), title: "New chat", lang: "en", createdAt: Date.now(),
           messages: [{ ...buildSystemMsg("en"), id: uid() }] };
}
function titleFromText(t) { return t.trim().slice(0, 45) || "New chat"; }

const STORAGE_KEY = "selam_cbe_conversations";
const THEME_KEY   = "selam_cbe_theme";
function loadConversations() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) { const p = JSON.parse(raw); if (Array.isArray(p) && p.length) return p; }
  } catch (_) {}
  return [makeConversation()];
}
function saveConversations(c) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(c)); } catch (_) {}
}
function loadTheme() { return localStorage.getItem(THEME_KEY) ?? "light"; }

// Safe random position: keeps floater well inside viewport on all devices
function randomPos() {
  const avatarR = 80;  // half-width of floater + ring overflow
  const vw = Math.max(window.innerWidth  - avatarR * 2, avatarR);
  const vh = Math.max(window.innerHeight - avatarR * 2, avatarR);
  // On small phones keep floater away from corners so speech bubble stays visible
  const minX = avatarR;
  const minY = window.innerHeight > 600 ? avatarR : avatarR + 20;
  return {
    x: minX + Math.random() * vw,
    y: minY + Math.random() * vh,
  };
}

// Full typewriter message
const TYPEWRITER_MSG = "Hi! I'm Selam, CBE Assistant. What can I help you?";

// ═══════════════════════════════════════════════════════════════
export default function App() {
  // ── Widget state ─────────────────────────────────────────
  const [open,      setOpen]      = useState(false);
  const [animState, setAnimState] = useState("hidden");

  // ── Floater state ─────────────────────────────────────────
  const [floatPos,  setFloatPos]  = useState({ x: 160, y: 220 });
  const [isFlying,  setIsFlying]  = useState(false);
  const [typedText, setTypedText] = useState("");
  const [isErasing, setIsErasing] = useState(false);
  const hoveredRef = useRef(false); // use ref so wander timer reads latest value
  const [hovered,   setHovered]   = useState(false);

  // ── Chat state ────────────────────────────────────────────
  const [conversations, setConvsRaw] = useState(loadConversations);
  const [activeId,  setActiveId]  = useState(() => loadConversations()[0]?.id ?? "");
  const [input,     setInput]     = useState("");
  const [loading,   setLoading]   = useState(false);
  const [speechOpen, setSpeech]   = useState(false);
  const [theme,     setThemeRaw]  = useState(loadTheme);

  const abortRef    = useRef(null);
  const floatTimer  = useRef(null);
  const typeTimer   = useRef(null);

  // Apply theme
  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
  }, [theme]);
  const setTheme = useCallback((t) => {
    setThemeRaw(t);
    localStorage.setItem(THEME_KEY, t);
  }, []);

  // ── Typewriter loop (runs whenever floater is visible) ────
  useEffect(() => {
    if (open) return; // hide when chat is open

    let charIdx   = 0;
    let erasing   = false;
    let pauseEnd  = false;

    const tick = () => {
      if (!erasing) {
        // typing phase
        charIdx++;
        setTypedText(TYPEWRITER_MSG.slice(0, charIdx));
        setIsErasing(false);
        if (charIdx === TYPEWRITER_MSG.length) {
          // hold for 2.2s then start erasing
          pauseEnd = true;
          typeTimer.current = setTimeout(() => {
            pauseEnd = false;
            erasing = true;
            typeTimer.current = setTimeout(tick, 60);
          }, 2200);
          return;
        }
      } else {
        // erasing phase
        charIdx--;
        setTypedText(TYPEWRITER_MSG.slice(0, charIdx));
        setIsErasing(true);
        if (charIdx === 0) {
          // pause then retype
          erasing = false;
          typeTimer.current = setTimeout(() => {
            setIsErasing(false);
            typeTimer.current = setTimeout(tick, 80);
          }, 900);
          return;
        }
      }
      // typing: ~65ms/char, erasing: ~35ms/char
      typeTimer.current = setTimeout(tick, erasing ? 35 : 65);
    };

    // Start after a short initial delay
    typeTimer.current = setTimeout(tick, 800);
    return () => clearTimeout(typeTimer.current);
  }, [open]);

  // ── Wander when floater is visible ───────────────────────
  // Timer always runs; on each tick, only moves if mouse is NOT over the avatar
  useEffect(() => {
    if (open) return;
    const wander = () => {
      if (!hoveredRef.current) {
        setFloatPos(randomPos());
      }
      floatTimer.current = setTimeout(wander, 6500 + Math.random() * 3000);
    };
    floatTimer.current = setTimeout(wander, 1000);
    return () => clearTimeout(floatTimer.current);
  }, [open]);

  // ── Click floater → fly to corner → open chat ────────────
  const handleAvatarClick = useCallback(() => {
    clearTimeout(floatTimer.current);
    clearTimeout(typeTimer.current);
    setTypedText("");
    setIsFlying(true);
    setTimeout(() => {
      setIsFlying(false);
      setOpen(true);
      setAnimState("entering");
      requestAnimationFrame(() =>
        requestAnimationFrame(() => setAnimState("visible"))
      );
    }, 550);
  }, []);

  // ── Close chat → resume floater ──────────────────────────
  const closeWidget = useCallback(() => {
    setAnimState("exiting");
    setTimeout(() => {
      setOpen(false);
      setAnimState("hidden");
      setFloatPos(randomPos());
    }, 260);
  }, []);

  // ── Conversations ─────────────────────────────────────────
  const setConversations = useCallback((updater) => {
    setConvsRaw((prev) => {
      const next = typeof updater === "function" ? updater(prev) : updater;
      saveConversations(next);
      return next;
    });
  }, []);

  const active = conversations.find((c) => c.id === activeId) ?? conversations[0];

  const clearConversation = useCallback(() => {
    setConversations((prev) => prev.map((c) => c.id !== activeId ? c : {
      ...c, title: "New chat", lang: "en",
      messages: [{ ...buildSystemMsg("en"), id: uid() }],
    }));
    setInput("");
  }, [activeId, setConversations]);

  // ── Message helpers ───────────────────────────────────────
  const appendMessage = useCallback((convId, msg) =>
    setConversations((p) => p.map((c) =>
      c.id === convId ? { ...c, messages: [...c.messages, msg] } : c)),
  [setConversations]);

  const appendToken = useCallback((convId, token) =>
    setConversations((p) => p.map((c) => {
      if (c.id !== convId) return c;
      const msgs = [...c.messages];
      const last = msgs[msgs.length - 1];
      if (last?.role === "assistant")
        msgs[msgs.length - 1] = { ...last, content: last.content + token, streaming: true };
      return { ...c, messages: msgs };
    })),
  [setConversations]);

  const finalizeStream = useCallback((convId) =>
    setConversations((p) => p.map((c) => {
      if (c.id !== convId) return c;
      return { ...c, messages: c.messages.map((m) =>
        m.role === "assistant" && m.streaming ? { ...m, streaming: false } : m) };
    })),
  [setConversations]);

  const setTitle = useCallback((convId, title) =>
    setConversations((p) => p.map((c) =>
      c.id === convId ? { ...c, title } : c)),
  [setConversations]);

  // ── Send ─────────────────────────────────────────────────
  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;

    const detectedLang = detectLanguage(text);
    const convId       = activeId;
    const isFirst      = active.messages.filter((m) => m.role === "user").length === 0;

    let baseMessages = active.messages;
    if (detectedLang !== (active.lang ?? "en")) {
      const newSys = { ...buildSystemMsg(detectedLang), id: uid() };
      baseMessages = [newSys, ...baseMessages.slice(1)];
      setConversations((p) => p.map((c) =>
        c.id === convId ? { ...c, lang: detectedLang, messages: baseMessages } : c));
    }

    const userMsg = { id: uid(), role: "user", content: text };
    appendMessage(convId, userMsg);
    if (isFirst) setTitle(convId, titleFromText(text));
    setInput("");
    setLoading(true);

    const assistantId = uid();
    appendMessage(convId, { id: assistantId, role: "assistant", content: "", streaming: true });

    abortRef.current = streamChat(
      buildHistory(baseMessages, text, detectedLang),
      {
        onToken: (token) => appendToken(convId, token),
        onDone: () => {
          finalizeStream(convId);
          setLoading(false);
        },
        onError: (err) => {
          setConversations((p) => p.map((c) => {
            if (c.id !== convId) return c;
            return { ...c, messages: c.messages.map((m) => m.id === assistantId ? {
              ...m, streaming: false,
              content: `⚠️ **Error:** ${err.message}\n\nMake sure the backend server is running.`,
            } : m) };
          }));
          setLoading(false);
        },
      }
    );
  }, [input, loading, activeId, active, appendMessage, appendToken, finalizeStream, setTitle, setConversations]);

  const handleStop = useCallback(() => {
    abortRef.current?.abort();
    finalizeStream(activeId);
    setLoading(false);
  }, [activeId, finalizeStream]);

  const handleSuggest = useCallback((text) => setInput(text), []);

  // ── Regenerate ────────────────────────────────────────────
  const handleRegenerate = useCallback(() => {
    if (loading) return;
    const convId   = activeId;
    const msgs     = active?.messages ?? [];
    const lastUser = [...msgs].reverse().find((m) => m.role === "user");
    if (!lastUser) return;

    setConversations((p) => p.map((c) => {
      if (c.id !== convId) return c;
      const trimmed = [...c.messages];
      while (trimmed.length && trimmed[trimmed.length - 1].role === "assistant") trimmed.pop();
      return { ...c, messages: trimmed };
    }));

    setLoading(true);
    const convLang = active?.lang ?? "en";
    const cleanHist = msgs
      .filter((m) => !m.content?.startsWith("[REMINDER:"))
      .slice(0, msgs.findLastIndex((m) => m.role === "user") + 1);
    const lastUserMsg = [...cleanHist].reverse().find((m) => m.role === "user");
    const hasSys = cleanHist.some((m) => m.role === "system");
    const histBase = hasSys
      ? cleanHist
      : [{ ...buildSystemMsg(convLang), id: uid() }, ...cleanHist.filter((m) => m.role !== "system")];
    const history = buildHistory(histBase, lastUserMsg?.content ?? "", convLang);

    const assistantId = uid();
    appendMessage(convId, { id: assistantId, role: "assistant", content: "", streaming: true });

    abortRef.current = streamChat(history, {
      onToken: (token) => appendToken(convId, token),
      onDone: () => {
        finalizeStream(convId);
        setLoading(false);
        setConversations((p) => {
          const conv = p.find((c) => c.id === convId);
          const last = conv && [...conv.messages].reverse().find((m) => m.role === "assistant");
          if (last?.content)
            speakText(last.content, conv.lang ?? "en")
              .then((url) => { const a = new Audio(url); a.volume = 1.0; a.play().catch(() => {}); })
              .catch(() => {});
          return p;
        });
      },
      onError: (err) => {
        setConversations((p) => p.map((c) => {
          if (c.id !== convId) return c;
          return { ...c, messages: c.messages.map((m) => m.id === assistantId
            ? { ...m, streaming: false, content: `⚠️ **Error:** ${err.message}` } : m) };
        }));
        setLoading(false);
      },
    });
  }, [loading, activeId, active, setConversations, appendMessage, appendToken, finalizeStream]);

  const isDark = theme === "dark";

  // ═══ Render ════════════════════════════════════════════════
  return (
    <>
      {/* ══ FLOATING AVATAR — wanders the screen ══ */}
      {!open && (
        <div
          className="floater"
          onClick={handleAvatarClick}
          onMouseEnter={() => {
            hoveredRef.current = true;
            // Cancel any pending wander move so position is frozen immediately
            clearTimeout(floatTimer.current);
          }}
          onMouseLeave={() => {
            hoveredRef.current = false;
            // Restart wander timer with a fresh delay after mouse leaves
            const wander = () => {
              if (!hoveredRef.current) {
                setFloatPos(randomPos());
              }
              floatTimer.current = setTimeout(wander, 6500 + Math.random() * 3000);
            };
            floatTimer.current = setTimeout(wander, 6500 + Math.random() * 3000);
          }}
          role="button"
          tabIndex={0}
          aria-label="Open Selam chat"
          onKeyDown={(e) => e.key === "Enter" && handleAvatarClick()}
        >
          {/* Rings stay inside .floater so they move with it */}
          <span className="floaterRing1" aria-hidden="true" />
          <span className="floaterRing2" aria-hidden="true" />
          <span className="floaterRing3" aria-hidden="true" />
          <span className="floaterRing4" aria-hidden="true" />
          <span className="floaterDot1"  aria-hidden="true" />
          <span className="floaterDot2"  aria-hidden="true" />
          <span className="floaterDot3"  aria-hidden="true" />

          {/* Avatar */}
          <img src={selamAvatar} alt="Selam" className="floaterImg" />

          {/* Speech bubble with typewriter text */}
          {typedText.length > 0 && (
            <div className="speechBubble" role="status" aria-live="polite">
              <span className="speechText">
                {typedText}
                <span className={`cursor ${isErasing ? "cursorErasing" : ""}`} aria-hidden="true">|</span>
              </span>
              <span className="speechTail" aria-hidden="true" />
            </div>
          )}
        </div>
      )}

      {/* ══ CHAT WIDGET — bottom-right corner ══ */}
      {open && (
        <div
          className={`widget ${
            animState === "entering" ? "widgetEnter"
            : animState === "visible" ? "widgetVisible"
            : "widgetExit"
          }`}
          role="dialog"
          aria-label="Selam – CBE Chat Assistant"
        >
          {/* Header */}
          <div className={`widgetHeader${loading ? " widgetHeaderGenerating" : ""}`}>
            <div className="headerLogoWrap">
              <span className="headerLogoRing"  aria-hidden="true" />
              <span className="headerDot1"      aria-hidden="true" />
              <span className="headerDot2"      aria-hidden="true" />
              <div className="headerAvatar">
                <img src={selamAvatar} alt="Selam" />
              </div>
            </div>
            <div className="headerInfo">
              <div className="headerName">Selam</div>
              <div className="headerSub">
                <span className="onlineDot" aria-hidden="true" />
                Online · CBE Virtual Assistant
              </div>
            </div>
            <span className="cbeLogo">CBE</span>

            <button className="headerBtn"
              onClick={() => setTheme(isDark ? "light" : "dark")}
              title={isDark ? "Light mode" : "Dark mode"}
              aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}>
              {isDark ? <Sun size={14} /> : <Moon size={14} />}
            </button>
            <button className="headerBtn" onClick={clearConversation}
              title="Clear chat" aria-label="Clear conversation">
              <Trash2 size={14} />
            </button>
            <button className="headerBtn" onClick={closeWidget}
              title="Minimise" aria-label="Minimise">
              <Minus size={14} />
            </button>
            <button className="headerBtn" onClick={closeWidget}
              title="Close" aria-label="Close">
              <X size={14} />
            </button>
          </div>

          {/* Chat body */}
          <div className="widgetBody">
            <ChatWindow
              messages={active?.messages ?? []}
              loading={loading}
              onSuggest={handleSuggest}
              lang={active?.lang ?? "en"}
              onRegenerate={handleRegenerate}
            />
            <ChatInput
              value={input}
              onChange={setInput}
              onSend={handleSend}
              onStop={handleStop}
              onOpenSpeech={() => setSpeech(true)}
              loading={loading}
              disabled={false}
            />
          </div>
        </div>
      )}

      {speechOpen && <SpeechPanel onClose={() => setSpeech(false)} />}
    </>
  );
}
