import os
import hashlib
import re
import sys
import warnings
import contextlib
import io
import torch

# Unsloth 2026.9.4 still touches a removed PyTorch Dynamo setting during import.
# Filter only that known dependency warning; keep other warnings visible.
warnings.filterwarnings(
    "ignore",
    message=r".*inline_inbuilt_nn_modules.*",
    category=FutureWarning,
    module=r"unsloth.*",
)
warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
)

try:
    from unsloth import FastLanguageModel
except Exception:
    FastLanguageModel = None
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextStreamer,
)
from peft import PeftModel
import shutil
import subprocess
import json
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from typing import List

# Optional imports
try:
    from transformers import TextIteratorStreamer
except Exception:
    TextIteratorStreamer = None

try:
    from ddgs import DDGS

    def ddg(query: str, max_results: int = 5):
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore", RuntimeWarning)
            return list(DDGS().text(query, max_results=max_results))
except Exception:
    try:
        from duckduckgo_search import DDGS

        def ddg(query: str, max_results: int = 5):
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore", RuntimeWarning)
                return list(DDGS().text(query, max_results=max_results))
    except Exception:
        ddg = None

try:
    from langdetect import detect as _detect_lang
except Exception:
    _detect_lang = None

try:
    import dokobot
except Exception:
    dokobot = None

try:
    import redis
except Exception:
    redis = None


# Redis is intentionally disabled: stale cached answers were echoing questions.
REDIS_ENABLED = False
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_SESSION_ID = os.getenv("REDIS_SESSION_ID", "cli-default")
REDIS_SESSION_TTL = int(os.getenv("REDIS_SESSION_TTL", "1800"))
REDIS_FAQ_TTL = int(os.getenv("REDIS_FAQ_TTL", "86400"))
REDIS_PROMPT_TTL = int(os.getenv("REDIS_PROMPT_TTL", "86400"))
REDIS_MAX_HISTORY_MESSAGES = 12
# The A100 has enough VRAM for BF16 12B inference; 4-bit is slower here.
LOAD_IN_4BIT = os.getenv("LOAD_IN_4BIT", "0") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
_REDIS_CLIENT = None

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _clean_generated_text(text: str) -> str:
    """Remove model control text and placeholders before it reaches the user or Redis."""
    text = re.sub(r"<NA>|<na>|<unk>|<pad>", "", text)
    text = re.sub(r"(?:^|\n)\s*(?:NA|N/A|\[NA\])\s*(?=\n|$)", "", text, flags=re.IGNORECASE)
    marker_pattern = r"(?:<end_of_turn>|<start_of_turn>|<turn\|>|<\|turn\|>|(?:^|\n)\s*(?:model|assistant)\s*(?::|\n))"
    return re.split(marker_pattern, text, maxsplit=1)[0].strip()


def _stream_safe_text(text: str) -> str:
    """Hide chat markers and any text generated after the first turn."""
    marker_re = re.compile(r"<(?:end_of_turn|start_of_turn|turn\|>|\|turn\|>)")
    match = marker_re.search(text)
    if match:
        return text[:match.start()]
    # Hold a possible incomplete marker at the end of a chunk.
    marker_prefixes = ("<end_of_turn", "<start_of_turn", "<turn|", "<|turn|")
    cut = len(text)
    for prefix in marker_prefixes:
        for index in range(max(0, len(text) - len(prefix)), len(text)):
            if text[index:].startswith(prefix[: len(text) - index]):
                cut = min(cut, index)
    return text[:cut]


def _count_sentence_endings(text: str) -> int:
    """Count sentence punctuation without treating dots inside URLs as endings."""
    url_pattern = re.compile(r"(?:https?://|www\.)[^\s<>()]+")
    masked_parts = []
    last_end = 0
    for match in url_pattern.finditer(text):
        masked_parts.append(text[last_end:match.start()])
        url = match.group(0)
        trailing_punctuation = url[-1] if url[-1] in ".!?።፧" else ""
        url_without_punctuation = url[:-1] if trailing_punctuation else url
        masked_parts.append(" " * len(url))
        if trailing_punctuation and re.search(r"https?://[^\s<>()]*\.[^\s<>()]+$|www\.[^\s<>()]*\.[^\s<>()]+$", url_without_punctuation):
            masked_parts.append(trailing_punctuation)
        last_end = match.end()
    masked_parts.append(text[last_end:])
    return len(re.findall(r"[.!?።፧]", "".join(masked_parts)))


def _is_question_echo(question: str, answer: str) -> bool:
    """Reject cached or generated text that only repeats the user's question."""
    normalize = lambda value: re.sub(r"\s+", " ", value).strip().casefold()
    return bool(question.strip() and answer.strip() and normalize(question) == normalize(answer))

def _is_allowed_conversation(question: str) -> bool:
    """Allow greetings and assistant-identity questions through the model."""
    normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    return bool(re.fullmatch(
        r"(?:hi|hello|hey|good morning|good afternoon|good evening|who are you|what are you|"
        r"how are you|ati eenyu|eenyu ati)",
        normalized,
    ))

def _get_redis_client():
    global _REDIS_CLIENT
    if not REDIS_ENABLED or redis is None:
        return None
    if _REDIS_CLIENT is None:
        try:
            _REDIS_CLIENT = redis.Redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            _REDIS_CLIENT.ping()
        except Exception as exc:
            print(f"Redis memory disabled: {exc}")
            _REDIS_CLIENT = None
    return _REDIS_CLIENT


def _redis_session_key() -> str:
    return f"chat:session:{REDIS_SESSION_ID}:messages"


def _redis_prompt_key(language: str) -> str:
    return f"chat:prompt:v2:{language}"


def _sanitize_history(history: list[dict], language: str | None = None) -> list[dict[str, str]]:
    sanitized = []
    for message in history:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            continue
        content = _clean_generated_text(content)
        if role == "assistant" and sanitized and sanitized[-1]["role"] == "user" and _is_question_echo(sanitized[-1]["content"], content):
            sanitized.pop()
            continue
        if content:
            sanitized.append({"role": role, "content": content})
    if language:
        language_history = []
        for index in range(0, len(sanitized) - 1):
            user_message = sanitized[index]
            assistant_message = sanitized[index + 1]
            if user_message["role"] != "user" or assistant_message["role"] != "assistant":
                continue
            if _detect_language_from_text(user_message["content"]) == language:
                language_history.extend((user_message, assistant_message))
        sanitized = language_history
    return sanitized[-REDIS_MAX_HISTORY_MESSAGES:]


def _rewrite_redis_history(client, history: list[dict[str, str]]) -> None:
    key = _redis_session_key()
    encoded = [json.dumps(message, ensure_ascii=False) for message in history]
    with client.pipeline() as pipe:
        pipe.delete(key)
        if encoded:
            pipe.rpush(key, *encoded)
            pipe.expire(key, REDIS_SESSION_TTL)
        pipe.execute()


def _load_redis_history(language: str | None = None) -> list[dict[str, str]]:
    client = _get_redis_client()
    if client is None:
        return []
    try:
        raw_history = []
        for item in client.lrange(_redis_session_key(), 0, -1):
            try:
                raw_history.append(json.loads(item))
            except (TypeError, json.JSONDecodeError):
                continue
        history = _sanitize_history(raw_history, language)
        _rewrite_redis_history(client, history)
        return history
    except Exception as exc:
        print(f"Could not load Redis conversation memory: {exc}")
        return []


def _save_redis_exchange(user_question: str, answer: str) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        cleaned_answer = _clean_generated_text(answer)
        if not user_question.strip() or not cleaned_answer or _is_question_echo(user_question, cleaned_answer):
            return
        with client.pipeline() as pipe:
            pipe.rpush(_redis_session_key(), json.dumps({"role": "user", "content": user_question}, ensure_ascii=False))
            pipe.rpush(_redis_session_key(), json.dumps({"role": "assistant", "content": cleaned_answer}, ensure_ascii=False))
            pipe.ltrim(_redis_session_key(), -REDIS_MAX_HISTORY_MESSAGES, -1)
            pipe.expire(_redis_session_key(), REDIS_SESSION_TTL)
            pipe.execute()
    except Exception as exc:
        print(f"Could not save Redis conversation memory: {exc}")


def _faq_key(question: str, language: str) -> str:
    value = f"{language}:{question.strip().lower()}".encode("utf-8")
    return f"chat:faq:{hashlib.sha256(value).hexdigest()}"


def _get_cached_faq(question: str, language: str) -> str | None:
    client = _get_redis_client()
    if client is None:
        return None
    try:
        cached_reply = client.get(_faq_key(question, language))
        cleaned_reply = _clean_generated_text(cached_reply) if cached_reply else None
        if cleaned_reply and _is_question_echo(question, cleaned_reply):
            client.delete(_faq_key(question, language))
            return None
        return cleaned_reply
    except Exception as exc:
        print(f"Could not read Redis FAQ cache: {exc}")
        return None


def _cache_faq(question: str, language: str, answer: str) -> None:
    client = _get_redis_client()
    if client is None:
        return
    try:
        cleaned_answer = _clean_generated_text(answer)
        if cleaned_answer and not _is_question_echo(question, cleaned_answer):
            client.set(_faq_key(question, language), cleaned_answer, ex=REDIS_FAQ_TTL)
    except Exception as exc:
        print(f"Could not write Redis FAQ cache: {exc}")


def _get_cached_system_prompt(language: str, prompt: str) -> str:
    client = _get_redis_client()
    if client is None:
        return prompt
    try:
        key = _redis_prompt_key(language)
        cached_prompt = client.get(key)
        if cached_prompt:
            return cached_prompt
        client.set(key, prompt, ex=REDIS_PROMPT_TTL)
    except Exception as exc:
        print(f"Could not read/write Redis system prompt cache: {exc}")
    return prompt


def _dokobot_search(query: str, max_results: int = 5) -> List[dict]:
    """Try to run `dokobot` search (Python API or CLI) and return results as list[dict].

    This is optional — if `dokobot` is not installed or the CLI is missing,
    the function returns an empty list.
    """
    results: List[dict] = []
    # Try Python import first (best-effort; dokobot API may differ)
    if dokobot is not None:
        try:
            if hasattr(dokobot, "search"):
                raw = dokobot.search(query, max_results=max_results)
            elif hasattr(dokobot, "run_skill"):
                raw = dokobot.run_skill("web-search", query, max_results=max_results)
            else:
                raw = None
            if raw:
                for r in raw[:max_results]:
                    results.append({
                        "title": r.get("title") or r.get("name") or "",
                        "href": r.get("url") or r.get("href") or r.get("link") or "",
                        "body": r.get("snippet") or r.get("summary") or r.get("excerpt") or "",
                        "source": "dokobot",
                    })
                return results
        except Exception:
            pass

    # Try CLI if available
    dokobot_cli = shutil.which("dokobot")
    if not dokobot_cli:
        return []

    try:
        # Attempt a JSON-output search command; this may vary by dokobot version.
        proc = subprocess.run([dokobot_cli, "search", "--query", query, "--max-results", str(max_results), "--json"], capture_output=True, text=True, timeout=20)
        if proc.returncode != 0:
            return []
        raw = json.loads(proc.stdout)
        for r in (raw or [])[:max_results]:
            results.append({
                "title": r.get("title") or r.get("name") or "",
                "href": r.get("url") or r.get("href") or r.get("link") or "",
                "body": r.get("snippet") or r.get("summary") or r.get("excerpt") or "",
                "source": "dokobot",
            })
        return results
    except Exception:
        return []


class _CBEPageTextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.skip_depth += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"} and self.skip_depth:
            self.skip_depth -= 1

    def handle_data(self, data):
        if not self.skip_depth and data.strip():
            self.parts.append(data.strip())


def _fetch_official_cbe_pages() -> List[dict]:
    """Fetch official CBE pages as grounded fallback evidence."""
    pages = []
    for url in (CBE_OFFICIAL_FAQ_URL, "https://combanketh.et/home"):
        try:
            request = Request(url, headers={"User-Agent": "CBE-local-assistant/1.0"})
            with urlopen(request, timeout=8) as response:
                parser = _CBEPageTextParser()
                parser.feed(response.read().decode("utf-8", errors="ignore"))
            body = re.sub(r"\s+", " ", " ".join(parser.parts)).strip()
            if body:
                pages.append({"title": url, "href": url, "body": body[:12000], "source": "official"})
        except Exception:
            continue
    return pages


def search_duckduckgo(query: str, max_results: int = 5, prefer_authoritative: bool = True) -> List[dict]:
    """Search DuckDuckGo and prefer authoritative Commercial Bank of Ethiopia sources.

    If `prefer_authoritative` is True the function will try to return results whose
    URLs contain indicators of official CBE domains (like 'combank', 'commercial', 'cbe', or '.et').
    """
    results: List[dict] = _fetch_official_cbe_pages()
    if ddg is None:
        return results[:max_results]

    # Build prioritized queries: official CBE site (combanketh.et), other official pages, general web, then social platforms
    queries = [
        f"{query} site:combanketh.et",
        f"{query} Commercial Bank of Ethiopia site:combanketh.et",
        f"{query} Commercial Bank of Ethiopia site:com",
        f"{query} Commercial Bank of Ethiopia",
        query,
    ]

    # Social media queries to probe X (twitter), LinkedIn, Facebook, YouTube
    social_sites = [
        ("X", "site:x.com"),
        ("LinkedIn", "site:linkedin.com"),
        ("Facebook", "site:facebook.com"),
        ("YouTube", "site:youtube.com"),
    ]

    seen = {result["href"] for result in results}

    # Known official CBE links (user-provided) — include these as authoritative sources when present
    known_cbe_links = [
        "https://combanketh.et/home",
        "https://combanketh.et/misalliance/faq",
        "https://www.linkedin.com/company/commercialbankofethiopia/",
        "https://t.me/combankethofficial",
        "https://www.tiktok.com/@combankethiopia",
    ]

    # First, try prioritized web queries
    for q in queries:
        try:
            raw = ddg(q, max_results=max_results)
        except Exception:
            raw = None
        if not raw:
            continue
        for r in raw:
            href = (r.get("href") or r.get("url") or "").strip()
            if not href or href in seen:
                continue
            seen.add(href)
            results.append({
                "title": r.get("title"),
                "href": href,
                "body": r.get("body") or r.get("snippet") or "",
                "source": "web",
            })
            if len(results) >= max_results:
                break
        if len(results) >= max_results:
            break

    # Then probe social platforms for potentially relevant posts
    if len(results) < max_results:
        for site_name, site_query in social_sites:
            try:
                raw = ddg(f"{query} {site_query}", max_results=max_results)
            except Exception:
                raw = None
            if not raw:
                continue
            for r in raw:
                href = (r.get("href") or r.get("url") or "").strip()
                if not href or href in seen:
                    continue
                seen.add(href)
                results.append({
                    "title": r.get("title"),
                    "href": href,
                    "body": r.get("body") or r.get("snippet") or "",
                    "source": site_name,
                })
                if len(results) >= max_results:
                    break
            if len(results) >= max_results:
                break

    # Insert known CBE links at the front if not already present
    final_results: List[dict] = []
    for link in known_cbe_links:
        if link not in (r.get("href") for r in results):
            final_results.append({"title": link, "href": link, "body": "", "source": "official"})

    # Prefer authoritative CBE domains if requested
    if prefer_authoritative:
        auth_indicators = [
            "combanketh.et", "combank", "commercial", "cbe", ".et", "combanketh", "bankofethiopia",
            "linkedin.com", "t.me", "telegram.me", "tiktok.com"
        ]
        authoritative = [r for r in results if any(ind in (r.get("href") or "").lower() for ind in auth_indicators)]
        if authoritative:
            ordered = final_results + authoritative + [r for r in results if r not in authoritative]
            return ordered[:max_results]

    # Otherwise append collected results after known links
    final_results.extend(results)
    return final_results[:max_results]


def search_general_web(query: str, max_results: int = 5) -> List[dict]:
    """Search the general web without injecting CBE links or CBE assumptions."""
    if ddg is None:
        return []
    results = []
    seen = set()
    try:
        raw = ddg(query, max_results=max_results)
    except Exception:
        raw = []
    for item in raw or []:
        href = (item.get("href") or item.get("url") or "").strip()
        if not href or href in seen:
            continue
        seen.add(href)
        results.append({
            "title": item.get("title") or "",
            "href": href,
            "body": item.get("body") or item.get("snippet") or "",
            "source": "web",
        })
        if len(results) >= max_results:
            break
    return results


def _is_cbe_question(question: str) -> bool:
    return bool(re.search(
        r"\b(?:cbe|commercial\s+bank|banking|account|atm|loan|credit|birr|cyberbank)\b",
        question.casefold(),
    ))


def _detect_language_from_text(text: str) -> str:
    if not text or not text.strip():
        return "en"

    # Prefer an explicit Latin-script answer for ordinary English input. This
    # prevents short mixed banking questions from being misclassified by a
    # statistical detector after a few domain-specific words.
    latin_words = re.findall(r"[A-Za-z]+", text)
    ethiopic_chars = re.findall(r"[\u1200-\u137F]", text)
    if latin_words and not ethiopic_chars:
        latin_text = " ".join(latin_words).casefold()
        if not re.search(
            r"\b(?:afaan\s+oromoo|oromoo|herrega|baankii|akkamitti|eenyu|maal|eessa|"
            r"sidee|sideen|akoon|bangiga|lacag|xisaab|furtaa|furto|qabta)\b",
            latin_text,
        ):
            return "en"

    # Oromo words and phrases. Include common banking/account vocabulary so
    # mixed Latin/Ethiopic Oromo is checked before script-based detection.
    oromo_keywords = (
        r"\b(?:afaan\s+oromoo|oromoo|herrega|baankii|haaraa|akkamitti|banuu|"
        r"danda['’]?a|sanadoonni|waraqaalee|eenyummaa|fiduu|qabu|maal\s+fa['’]?i|"
        r"galma|beekumsa|mirkanaa['’]?aa|miti|hir'isee|kennine|gochuun|"
        r"ati|eenyu|maal|eessa|akkam|maaliif)\b"
    )
    if re.search(oromo_keywords, text, re.IGNORECASE):
        return "or"

    # Script alone cannot distinguish Amharic from Tigrinya. Common words
    # provide a deterministic result for short banking questions.
    tigrinya_keywords = r"\b(?:ኣካውንት|ባንኪ|ከመይ|ክኸፍት|እኽእል|ጌረ|ገንዘብ|ሒሳብ|ኣበይ)\b"
    amharic_keywords = r"\b(?:የባንክ|ሂሳብ|እንዴት|እከፍታለሁ|እችላለሁ|ገንዘብ|የት)\b"
    if re.search(tigrinya_keywords, text):
        return "ti"
    if re.search(amharic_keywords, text):
        return "am"

    if re.search(r"[\u1200-\u137F]", text):
        return "am"

    somali_keywords = r"\b(?:sidee|sideen|akoon|bangiga|lacag|xisaab|furtaa|furto|qabta)\b"
    if re.search(somali_keywords, text, re.IGNORECASE):
        return "so"

    if _detect_lang is not None:
        try:
            detected = _detect_lang(text)
            return detected
        except Exception:
            pass

    return "en"


def get_dynamic_max_tokens(user_input: str) -> int:
    """Give Ethiopic-script answers more room than concise English answers."""
    if any("\u1200" <= char <= "\u137F" for char in user_input):
        return 400
    return 256


def get_max_sentences(user_input: str) -> int:
    """Allow complete requirement lists while keeping simple replies short."""
    if re.search(
        r"\b(?:eligib\w*|criter\w*|requirement\w*|qualif\w*|required|how much|how many|types?)\b|"
        r"(?:መስፈርት|ብቁ|መስፈርቶች|ምንድን|ስንት)",
        user_input,
        re.IGNORECASE,
    ):
        return 4
    return 2

# ── Config ────────────────────────────────────────────────────────────────────
CBE_OFFICIAL_FAQ_URL = "https://combanketh.et/misalliance/faq"
CBE_LOAN_URL = "https://combanketh.et/products/loan"
CBE_HOME_URL = "https://combanketh.et/home"
FINETUNED_MODEL = os.getenv("FINETUNED_MODEL", "./banking-model-gemma4-12b")
BASE_MODEL      = os.getenv("BASE_MODEL", "google/gemma-4-12b-it")
# Keep enough context for conversation history and complete multilingual answers.
MAX_SEQ_LEN     = int(os.getenv("MAX_SEQ_LEN", "8192"))
MAX_NEW_TOKENS  = int(os.getenv("MAX_NEW_TOKENS", "256"))
STREAM_RESPONSES = os.getenv("STREAM_RESPONSES", "1") == "1"
WARMUP_NEW_TOKENS = 8

SYSTEM_PROMPT = """
You are the official Multilingual AI Banking Assistant for the Commercial Bank of Ethiopia (CBE) (የኢትዮጵያ ንግድ ባንክ).

SCOPE: Use the fine-tuned CBE banking knowledge as the primary context and answer the exact question naturally and reasonably. Prioritize CBE accounts, branches, ATMs, cards, loans, payments, CBE Birr, CBE CyberBank, services, policies, and customer support. Greetings and identity questions such as "hello", "who are you?", and "what can you do?" must receive a natural helpful answer describing this CBE banking assistant. Do not repeat training instructions, chat markers, or generic refusal text. Do not invent names, fees, requirements, limits, or policies; when the trained context does not contain enough information, say that the information is not available and suggest contacting CBE Customer Care at 951 or a branch.

Institutional rules:
1. Express financial values and transaction caps only in Ethiopian Birr: use Qarshii in Afaan Oromoo, ብር in Amharic, and ETB or Birr in English. Never use US Dollars or $.
2. Mention the 50 ETB minimum initial deposit and identification requirements only when the user asks about opening a standard CBE savings account. Do not apply those facts to CBE Birr, loans, cards, or other products unless they are directly relevant.
3. Use the official product names CBE Birr or CBE Mobile Banking App, and CBE CyberBank for internet banking.
4. For failed transactions or ATM errors, tell the customer to call CBE Customer Care at 951 or visit the nearest branch.
5. Assume all operations are in Ethiopia and use localized terminology.
6. Use only verified URLs: the official CBE home page is https://combanketh.et/home and the loan page is https://combanketh.et/products/loan. Never invent, shorten, or complete a URL. If a direct online application URL cannot be verified, say so and provide the relevant official product page.
""".strip()


def _verified_link_reply(question: str, language: str) -> str | None:
    """Answer direct English CBE link requests from verified URLs."""
    if language != "en":
        return None
    normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold()).strip()
    asks_for_link = bool(re.search(r"\b(?:url|link|website|portal|online|apply|application|mapply)\b", normalized))
    asks_about_loan = bool(re.search(r"\b(?:loan|credit|borrow|mapply)\b", normalized))
    if asks_for_link and asks_about_loan:
        return (
            f"For CBE loan information and application guidance, use the official loan page: {CBE_LOAN_URL}. "
            "A separate direct online loan-application URL could not be verified; follow the instructions on that page or call CBE Customer Care at 951."
        )
    if asks_for_link and re.search(r"\b(?:cbe|bank|official)\b", normalized):
        return f"The official Commercial Bank of Ethiopia website is {CBE_HOME_URL}."
    return None

def _generation_eos_token_ids(tokenizer):
    """Stop Gemma 4 at either the normal EOS or end-of-turn token."""
    token_ids = [tokenizer.eos_token_id]
    eot_token_id = getattr(tokenizer, "eot_token_id", None)
    if eot_token_id is not None and eot_token_id not in token_ids:
        token_ids.append(eot_token_id)
    return token_ids


def _string_stop_kwargs(tokenizer) -> dict:
    """Use string stopping only when the loaded object is a text tokenizer."""
    if not callable(getattr(tokenizer, "get_vocab", None)):
        return {}
    return {
        "stop_strings": ["<end_of_turn>", "<start_of_turn>", "<turn|>"],
        "tokenizer": tokenizer,
    }


class _ChatTurnStop(StoppingCriteria):
    """Stop after the model emits a chat-turn marker."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer.tokenizer
        convert_tokens_to_ids = self.tokenizer.convert_tokens_to_ids
        self.stop_token_ids = {
            token_id
            for token in ("<end_of_turn>", "<start_of_turn>", "<turn|>")
            for token_id in (
                [convert_tokens_to_ids(token)]
                if callable(convert_tokens_to_ids)
                else []
            )
            if isinstance(token_id, int) and token_id >= 0
        }
        eot_token_id = getattr(self.tokenizer, "eot_token_id", None)
        if isinstance(eot_token_id, int):
            self.stop_token_ids.add(eot_token_id)

    def __call__(self, input_ids, scores, **kwargs):
        if self.stop_token_ids:
            return input_ids[0, -1].item() in self.stop_token_ids
        return False


class _SentenceLimitStop(StoppingCriteria):
    """Stop after two completed sentences in the generated answer."""

    def __init__(self, tokenizer, prompt_length: int, max_sentences: int = 2):
        self.decoder = getattr(tokenizer, "tokenizer", tokenizer)
        self.prompt_length = prompt_length
        self.max_sentences = max_sentences

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0, self.prompt_length:]
        decode = getattr(self.decoder, "decode", None)
        if not callable(decode):
            return False
        generated_text = decode(generated_ids, skip_special_tokens=False)
        return _count_sentence_endings(generated_text) >= self.max_sentences


class _StreamingGuard:
    def __init__(self):
        self.stop = False


class _StreamingGuardStop(StoppingCriteria):
    def __init__(self, guard: _StreamingGuard):
        self.guard = guard

    def __call__(self, input_ids, scores, **kwargs):
        return self.guard.stop


def _stopping_criteria(
    tokenizer,
    prompt_length: int,
    guard: _StreamingGuard | None = None,
    max_sentences: int = 2,
) -> StoppingCriteriaList:
    criteria = [
        _ChatTurnStop(tokenizer),
        _SentenceLimitStop(tokenizer, prompt_length, max_sentences),
    ]
    if guard is not None:
        criteria.append(_StreamingGuardStop(guard))
    return StoppingCriteriaList(criteria)


def _model_answer_is_usable(question: str, answer: str, language: str) -> bool:
    answer = _clean_generated_text(answer)
    if not answer or _is_question_echo(question, answer):
        return False
    if language in {"am", "ti"}:
        return bool(re.search(r"[\u1200-\u137F]", answer))
    detected = _detect_language_from_text(answer)
    if language == "or":
        return detected == "or"
    if language == "so":
        return detected == "so"
    return not re.search(r"[\u1200-\u137F]", answer) and detected == "en"


def _complete_word_prefix(text: str) -> str:
    """Return only complete words for clean streaming display."""
    if not text or text[-1].isspace() or text[-1] in ".,!?;:)]}":
        return text
    boundary = max(text.rfind(char) for char in (" ", "\n", "\t"))
    return text[:boundary + 1] if boundary >= 0 else ""


def _needs_current_cbe_search(question: str) -> bool:
    """Use fresh sources for current CBE leadership and other time-sensitive facts."""
    normalized = re.sub(r"[^a-z0-9]+", " ", question.casefold())
    asks_leadership = re.search(r"\b(?:who|president|chairman|chairperson|ceo|head|leader)\b", normalized)
    has_cbe_context = re.search(r"\b(?:cbe|bank|commercial bank|ethiopia|president|prsnedit)\b", normalized)
    return bool(asks_leadership and has_cbe_context)


# ── Load model ────────────────────────────────────────────────────────────────
use_base = False
model_path = FINETUNED_MODEL

print(f"🦥 Attempting to load fine-tuned CBE model from: {model_path}...")

# Try to load the fine-tuned model; if not present or load fails, fall back to web-search-only mode
model = None
tokenizer = None
if os.path.exists(model_path):
    try:
        if FastLanguageModel is None:
            raise RuntimeError("Unsloth is unavailable")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = model_path,
            max_seq_length = MAX_SEQ_LEN,
            dtype          = torch.bfloat16 if DEVICE == "cuda" else torch.float32,
            load_in_4bit   = LOAD_IN_4BIT,
        )
        FastLanguageModel.for_inference(model)
        model.eval()
        print("🦥 Model loaded successfully.")
    except Exception as e:
        print(f"⚠️  Unsloth loader unavailable: {e}")
        try:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True) if LOAD_IN_4BIT else None
            tokenizer_path = model_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
            )
            if not use_base:
                model = PeftModel.from_pretrained(model, model_path)
            model.eval()
            print("✅ Model loaded with Transformers + PEFT.")
        except Exception as fallback_error:
            print(f"⚠️  Failed to load model at {model_path}: {fallback_error}")
            model = None
            tokenizer = None
else:
    print(f"⚠️  Fine-tuned model not found at {model_path}. DuckDuckGo fallback enabled.")

# ── Inference helper ──────────────────────────────────────────────────────────
def ask_interactive(user_question: str, is_warmup=False):
    """Generates a response and handles tokenization correctly."""
    detected_lang = _detect_language_from_text(user_question)
    verified_reply = _verified_link_reply(user_question, detected_lang)
    if verified_reply is not None and not is_warmup:
        print(f"Bot: {verified_reply}", flush=True)
        _save_redis_exchange(user_question, verified_reply)
        _cache_faq(user_question, detected_lang, verified_reply)
        return
    dynamic_limit = get_dynamic_max_tokens(user_question)
    max_sentences = get_max_sentences(user_question)
    sentence_instruction = (
        f"You are a concise banking assistant. Limit your response to a maximum of {max_sentences} sentences."
        if detected_lang == "en"
        else f"መልስህ አጭር እና ከ {max_sentences} አረፍተ ነገር ያልበለጠ መሆን አለበት።"
    )
    eligibility_instruction = (
        "The user is asking for eligibility criteria or requirements. Answer only with the relevant eligibility conditions and required documents. Do not explain app downloads, registration steps, fees, or unrelated products unless explicitly requested."
        if re.search(
            r"\b(?:eligib\w*|criter\w*|requirement\w*|qualif\w*|eligible)\b|"
            r"(?:መስፈርት|ብቁ|መስፈርቶች)",
            user_question,
            re.IGNORECASE,
        )
        else ""
    )
    cbe_question = _is_cbe_question(user_question)
    domain_instruction = (
        "Focus only on Commercial Bank of Ethiopia sources."
        if cbe_question
        else "Answer using the web evidence without reframing the question as a banking or CBE question."
    )
    if not is_warmup:
        cached_reply = _get_cached_faq(user_question, detected_lang)
        if cached_reply is not None:
            print(f"Bot (Redis FAQ cache): {cached_reply}")
            _save_redis_exchange(user_question, cached_reply)
            return

    redis_history = _load_redis_history(detected_lang) if not is_warmup else []
    
    # If model/tokenizer aren't available, use DuckDuckGo fallback (skip during warm-up)
    if model is None or tokenizer is None:
        if is_warmup:
            return
        snippets = []
        try:
            snippets = (
                search_duckduckgo(user_question, max_results=5)
                if cbe_question
                else search_general_web(user_question, max_results=5)
            )
            source_used = "DuckDuckGo"
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            snippets = []
            source_used = None

        if not snippets and dokobot is not None:
            try:
                dokobot_snips = _dokobot_search(user_question, max_results=5)
                if dokobot_snips:
                    snippets = dokobot_snips
                    source_used = "dokobot"
            except Exception:
                pass

        if not snippets:
            print("No web results found.")
            return

        combined = "\n\n".join([s.get('body','') for s in snippets if s.get('body')])
        print(f"Bot: {combined[:4000]}")
        return

    language_name = {
        "en": "English",
        "am": "Amharic (written only in Ethiopic script)",
        "or": "Afaan Oromoo",
        "ti": "Tigrinya",
        "so": "Somali",
    }.get(detected_lang, detected_lang)

    system_prompt = _get_cached_system_prompt(detected_lang, (
        f"{SYSTEM_PROMPT}\n\nIdentify the user's language from the actual message before answering. "
        f"The application detected {language_name} as a hint, but if that hint conflicts with the user's words, follow the user's actual language. "
        f"{sentence_instruction} {eligibility_instruction} Answer in exactly the user's language and script. Never translate, add a bilingual version, or switch languages. If the user's message is written in Latin-script English, answer only in English. If it contains multiple requests, answer each request in that same detected language and do not switch languages between them. Answer the exact question asked; do not assume it is about account opening or credit. Do not use English or another language, "
        "do not repeat the answer, do not output chat role markers, placeholders, or <NA>. "
        "Use only facts supported by the fine-tuned CBE context and clearly state when information is unavailable. First identify the user's exact product, task, and requested details, then answer only that question. Keep every sentence and numbered step relevant to the current request. Do not transfer requirements, fees, or procedures from another CBE product. Do not start a second answer, add examples for a different question, or continue after the end-of-turn marker. Give a concise, complete answer with next steps when relevant. Finish the final sentence and final numbered step completely; never stop in the middle of a sentence, word, or list item."
    ))

    messages = [
        {"role": "system", "content": system_prompt},
        *redis_history,
        {"role": "user", "content": user_question},
    ]
    
    # 1. Generate the formatted prompt string
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # 2. Convert prompt string to tensors
    original_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    inputs = tokenizer(
        text=prompt,
        images=None,
        return_tensors="pt",
        truncation=True,
        max_length=max(1, MAX_SEQ_LEN - dynamic_limit),
    ).to(DEVICE)
    tokenizer.truncation_side = original_truncation_side
    prompt_len = inputs["input_ids"].shape[1]
    
    # 3. Generate response text
    response_text = ""
    streamed_to_console = False
    eos_token_ids = _generation_eos_token_ids(tokenizer)
    with torch.inference_mode():
        if is_warmup:
            _ = model.generate(
                **inputs,
                max_new_tokens = WARMUP_NEW_TOKENS,
                use_cache      = True,
                temperature    = 0.7,
                top_p          = 0.9,
                do_sample      = True,
                pad_token_id   = tokenizer.pad_token_id,
                eos_token_id   = eos_token_ids,
                stopping_criteria = _stopping_criteria(tokenizer, prompt_len, max_sentences=max_sentences),
            )
        elif STREAM_RESPONSES and TextIteratorStreamer is not None:
            import threading

            stream_guard = _StreamingGuard()
            streamer = TextIteratorStreamer(
                tokenizer=tokenizer,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            gen_thread = threading.Thread(
                target=model.generate,
                kwargs={
                    **inputs,
                    "streamer": streamer,
                    "max_new_tokens": dynamic_limit,
                    "use_cache": True,
                    "do_sample": False,
                    "repetition_penalty": 1.08,
                    "pad_token_id": tokenizer.pad_token_id,
                    "eos_token_id": eos_token_ids,
                    "stopping_criteria": _stopping_criteria(
                        tokenizer, prompt_len, stream_guard, max_sentences
                    ),
                    **_string_stop_kwargs(tokenizer),
                },
            )
            gen_thread.daemon = True
            gen_thread.start()

            streamed_to_console = True
            print("Bot: ", end="", flush=True)
            displayed_text = ""
            sentence_count = 0
            last_chunks = []
            for chunk in streamer:
                response_text += chunk
                text_chunk = _stream_safe_text(chunk).strip()
                if not text_chunk:
                    continue
                if any(
                    tag in text_chunk
                    for tag in ("<start_of_turn>", "<end_of_turn>", "<turn|>", "model", "user")
                ):
                    stream_guard.stop = True
                    break
                last_chunks.append(text_chunk)
                if len(last_chunks) > 3:
                    last_chunks.pop(0)
                if len(last_chunks) == 3 and len(set(last_chunks)) == 1:
                    stream_guard.stop = True
                    break
                sentence_count = _count_sentence_endings(response_text)
                if sentence_count >= 2:
                    stream_guard.stop = True
                cleaned_text = _stream_safe_text(response_text)
                displayable_text = _complete_word_prefix(cleaned_text)
                if displayable_text.startswith(displayed_text):
                    delta = displayable_text[len(displayed_text):]
                    if delta:
                        print(delta, end="", flush=True)
                        displayed_text = displayable_text
                if stream_guard.stop:
                    break

            if not stream_guard.stop:
                gen_thread.join()
            # Flush only the cleaned first answer. The raw buffer may contain
            # a second assistant turn after the end-of-turn marker.
            final_text = _clean_generated_text(response_text)
            if final_text.startswith(displayed_text):
                remainder = final_text[len(displayed_text):]
                if remainder:
                    print(remainder, end="", flush=True)
            print(flush=True)
            response_text = final_text
        else:
            output = model.generate(
                **inputs,
                max_new_tokens = dynamic_limit,
                use_cache      = True,
                do_sample      = False,
                repetition_penalty = 1.08,
                pad_token_id   = tokenizer.pad_token_id,
                eos_token_id   = eos_token_ids,
                stopping_criteria = _stopping_criteria(tokenizer, prompt_len, max_sentences=max_sentences),
            )
            response_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            response_text = _clean_generated_text(response_text)

    normalized_resp = response_text.strip().lower()

    unsure_indicators = [
        "i don't know",
        "i am not sure",
        "i'm not sure",
        "i do not know",
        "can't answer",
        "cannot find",
        "no information",
        "i have no",
        "አላውቅም",
        "እርግጠኛ አይደለሁም",
        "hin beeku",
        "mirkanaa'aa miti",
        "ኣይፈልጥን",
    ]

    # Interactive replies return directly after local generation. Do not run a
    # second web search or generation after the answer, because that blocks the
    # next prompt even when the visible response is already complete.
    if False and not is_warmup and ddg is not None:
        # Perform DuckDuckGo search focused on Commercial Bank of Ethiopia
        snippets = []
        try:
            snippets = search_duckduckgo(user_question, max_results=5)
            source_used = "DuckDuckGo"
        except Exception as e:
            print(f"DuckDuckGo search failed: {e}")
            snippets = []
            source_used = None

        if not snippets and dokobot is not None:
            try:
                dokobot_snips = _dokobot_search(user_question, max_results=5)
                if dokobot_snips:
                    snippets = dokobot_snips
                    source_used = "dokobot"
            except Exception:
                snippets = []

        if snippets:
            # Re-run generation with search results appended to the prompt to improve answer
            # Build numbered search results and instruction to cite sources
            numbered = []
            for i, s in enumerate(snippets, start=1):
                title = (s.get('title') or '').strip()
                body = (s.get('body') or s.get('snippet') or '').strip()
                href = (s.get('href') or s.get('url') or '').strip()
                numbered.append(f"[{i}] {title} - {body} ({href})")

            search_block = "\n".join(numbered)
            leadership_question = _needs_current_cbe_search(user_question)
            usable_source_count = sum(
                bool((s.get('body') or s.get('snippet') or '').strip())
                for s in snippets
            )
            leadership_source_count = sum(
                bool(re.search(
                    r"\b(?:president|chairman|chairperson|board|management|director|ceo|leadership)\b",
                    (s.get('body') or s.get('snippet') or '').casefold(),
                ))
                for s in snippets
            )
            if leadership_question and (not usable_source_count or not leadership_source_count):
                print("Bot: I could not verify the current CBE leadership from the available official sources.")
                return
            fallback_tokens = 256 if leadership_question else 256
            fallback_word_limit = "200 words" if leadership_question else "120 words"
            search_instr = (
                "Search results (numbered). When you use information from these results, "
                "cite them inline using the bracketed number like [1], [2]. At the end, include a 'Sources' "
                "section listing the URLs. "
                f"{domain_instruction} "
                f"Answer only in {language_name}, do not repeat any word or sentence, and do not output role markers. "
                f"Give a complete answer in {fallback_word_limit}. Never guess, infer, or invent a name, role, date, amount, or requirement. For leadership questions, list only names and roles explicitly supported by the source text. If the sources do not answer the question, "
                "say that the customer should contact CBE customer service rather than inventing details."
            )

            prompt_with_search = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "system", "content": search_instr + "\n\n" + search_block},
                    {"role": "user", "content": user_question + "\n\nUse the search results above to answer the exact question."},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs2 = tokenizer(
                text=prompt_with_search,
                images=None,
                return_tensors="pt",
            ).to(DEVICE)
            with torch.inference_mode():
                if STREAM_RESPONSES and TextIteratorStreamer is not None:
                    streamer2 = TextIteratorStreamer(
                        tokenizer=tokenizer,
                        skip_prompt=True,
                        skip_special_tokens=True,
                    )
                    import threading

                    gen_thread2 = threading.Thread(
                        target=model.generate,
                        kwargs={
                            **inputs2,
                            "streamer": streamer2,
                            "max_new_tokens": min(fallback_tokens, dynamic_limit),
                            "use_cache": True,
                            "temperature": 0.3,
                            "top_p": 0.95,
                            "do_sample": False,
                            "repetition_penalty": 1.2,
                            "no_repeat_ngram_size": 3,
                            "pad_token_id": tokenizer.pad_token_id,
                            "eos_token_id": _generation_eos_token_ids(tokenizer),
                            "stopping_criteria": _stopping_criteria(
                                tokenizer,
                                inputs2["input_ids"].shape[1],
                                max_sentences=max_sentences,
                            ),
                            **_string_stop_kwargs(tokenizer),
                        },
                    )
                    gen_thread2.start()
                    web_response = ""
                    streamed_to_console = True
                    print("\nBot: ", end="", flush=True)
                    displayed_web_text = ""
                    for chunk in streamer2:
                        web_response += chunk
                        cleaned_web_text = _stream_safe_text(web_response)
                        displayable_web_text = _complete_word_prefix(cleaned_web_text)
                        if displayable_web_text.startswith(displayed_web_text):
                            delta = displayable_web_text[len(displayed_web_text):]
                            if delta:
                                print(delta, end="", flush=True)
                                displayed_web_text = displayable_web_text
                    gen_thread2.join()
                    final_web_text = _clean_generated_text(web_response)
                    if final_web_text.startswith(displayed_web_text):
                        remainder = final_web_text[len(displayed_web_text):]
                        if remainder:
                            print(remainder, end="", flush=True)
                    print(flush=True)
                    response_text = final_web_text
                else:
                    output = model.generate(
                        **inputs2,
                        max_new_tokens = min(fallback_tokens, dynamic_limit),
                        use_cache      = True,
                        temperature    = 0.3,
                        top_p          = 0.95,
                        do_sample      = False,
                        repetition_penalty = 1.2,
                        no_repeat_ngram_size = 3,
                        pad_token_id   = tokenizer.pad_token_id,
                        eos_token_id   = _generation_eos_token_ids(tokenizer),
                        stopping_criteria = _stopping_criteria(
                            tokenizer,
                            inputs2["input_ids"].shape[1],
                            max_sentences=max_sentences,
                        ),
                    )
                    output_text = tokenizer.decode(output[0][inputs2["input_ids"].shape[1]:], skip_special_tokens=True)
                    response_text = _clean_generated_text(output_text)

    if response_text.strip() and not is_warmup:
        if not streamed_to_console:
            print(f"Bot: {response_text.strip()}", flush=True)
        _save_redis_exchange(user_question, response_text.strip())
        _cache_faq(user_question, detected_lang, response_text.strip())

def main():
    print("🦥 Optimizing GPU kernels (Warm-up)...", end="", flush=True)
    ask_interactive("Hello", is_warmup=True)
    print(" Done!")

    print("\n" + "=" * 60)
    print("  Multilingual Banking Chatbot (banking-model-gemma4) - Ready")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            break

        ask_interactive(user_input)
        print()


if __name__ == "__main__":
    main()