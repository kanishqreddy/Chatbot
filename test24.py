import re
import os
import textwrap
import math
from collections import Counter

import streamlit as st
import psutil
import requests

# ---------------------------
# Basic config
# ---------------------------

st.set_page_config(page_title="Asha – Portfolio Chatbot", layout="centered")

CHARACTER_NAME = "Asha"

INTRO_MESSAGE = (
    "Hey, I’m Asha 👋\n\n"
    "I’m here to help you explore this portfolio and answer questions "
    "about Kanishq Reddy, his experience, skills, and projects — based only "
    "on what’s written here."
)

STOPWORDS = {
    "the", "and", "a", "an", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "it", "this", "that", "as", "at",
    "by", "from", "or", "be", "about", "my", "i", "you", "your",
    "me", "we", "our", "they", "their", "he", "she", "his", "her",
    "them"
}

# URL of the live portfolio site – used internally only
DEFAULT_SITE_URL = "https://kanishqreddy.github.io/"
SITE_URL = os.environ.get("PORTFOLIO_URL", DEFAULT_SITE_URL)

# ---------------------------
# HTML / text utilities
# ---------------------------

def clean_html_text(html: str) -> str:
    """Strip tags and script/style from HTML and normalise whitespace."""
    if not html:
        return ""
    html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)
    html = re.sub(
        r"<script[^>]*>.*?</script>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    html = re.sub(
        r"<style[^>]*>.*?</style>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    html = re.sub(r"<[^>]+>", " ", html)  # strip remaining tags
    html = re.sub(r"&nbsp;?", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def split_sentences(text: str):
    """Very small sentence splitter."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def tokenize(text: str):
    """Lowercase word tokens, minus stopwords / very short tokens."""
    return [
        w.lower()
        for w in re.findall(r"\b\w+\b", text)
        if len(w) > 2 and w.lower() not in STOPWORDS
    ]

# ---------------------------
# Load & index remote page + build TF-IDF model
# ---------------------------

@st.cache_resource
def load_site_knowledge():
    debug = []
    html_text = None

    # 1) Try downloading from the live portfolio URL
    try:
        resp = requests.get(SITE_URL, timeout=5)
        debug.append(f"SITE_URL status={resp.status_code}")
        if resp.status_code == 200:
            html_text = resp.text
    except Exception as e:
        debug.append(f"Error fetching SITE_URL: {e}")

    # 2) Fallback: try a local index.html (for local dev)
    if html_text is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        for candidate in [
            os.path.join(base_dir, "index.html"),
            os.path.join(base_dir, "public", "index.html"),
            os.path.join(base_dir, "static", "index.html"),
        ]:
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        html_text = f.read()
                    break
                except Exception as e:
                    debug.append(f"Error reading {candidate}: {e}")

    if html_text is None:
        # Nothing worked – return empty skeleton + internal debug info
        return {
            "sections": {},
            "all_text": "",
            "idf": {},
            "doc_vecs": {},
            "doc_norms": {},
            "debug": debug,
        }

    # Extract <section id="..."> blocks
    section_pattern = re.compile(
        r'<section\s+[^>]*id="([^"]+)"[^>]*>(.*?)</section>',
        re.DOTALL | re.IGNORECASE,
    )
    sections = {}
    for sec_id, sec_html in section_pattern.findall(html_text):
        clean_text = clean_html_text(sec_html)
        sentences = split_sentences(clean_text)
        heading_match = re.search(
            r"<h2[^>]*>(.*?)</h2>",
            sec_html,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if heading_match:
            heading = clean_html_text(heading_match.group(1))
        else:
            heading = sec_id.replace("-", " ").title()

        sections[sec_id] = {
            "id": sec_id,
            "heading": heading,
            "text": clean_text,
            "sentences": sentences,
        }

    # ---- Build TF-IDF model over sections ----
    docs_tokens = {}
    df = Counter()
    for sec_id, sec in sections.items():
        tokens = tokenize(sec["text"])
        docs_tokens[sec_id] = tokens
        for t in set(tokens):
            df[t] += 1

    N = max(len(sections), 1)
    idf = {
        term: math.log((1 + N) / (1 + freq)) + 1.0
        for term, freq in df.items()
    }

    doc_vecs = {}
    doc_norms = {}
    for sec_id, tokens in docs_tokens.items():
        tf = Counter(tokens)
        vec = {}
        for term, f in tf.items():
            if term in idf:
                vec[term] = (f / len(tokens)) * idf[term]
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        doc_vecs[sec_id] = vec
        doc_norms[sec_id] = norm

    all_text = clean_html_text(html_text)

    debug.append(f"Loaded {len(sections)} sections")

    return {
        "sections": sections,
        "all_text": all_text,
        "idf": idf,
        "doc_vecs": doc_vecs,
        "doc_norms": doc_norms,
        "debug": debug,
    }


SITE = load_site_knowledge()

# Human-friendly mapping for some section IDs
SECTION_ALIASES = {
    "about": {"about", "summary", "intro", "introduction", "profile"},
    "projects": {
        "project", "projects", "experience", "work", "job", "internship",
        "role", "roles",
    },
    "interactive-apps": {"app", "apps", "application", "dashboard", "chatbot"},
    "skills": {"skill", "skills", "tools", "stack", "tech", "technologies"},
    "education": {"education", "degree", "college", "university", "btech", "b.tech"},
    "contact": {"contact", "email", "phone", "reach", "connect"},
    "hero": {"hero", "top", "intro", "header"},
}

# ---------------------------
# TF-IDF scoring utilities
# ---------------------------

def build_query_vector(question: str):
    tokens = tokenize(question)
    if not tokens:
        return {}, 1.0
    tf = Counter(tokens)
    idf = SITE["idf"]
    vec = {}
    for term, f in tf.items():
        if term in idf:
            vec[term] = (f / len(tokens)) * idf[term]
    if not vec:
        return {}, 1.0
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return vec, norm


def cosine_sim(vec_q, norm_q, vec_d, norm_d):
    if not vec_q or not vec_d:
        return 0.0
    dot = 0.0
    for term, wq in vec_q.items():
        wd = vec_d.get(term)
        if wd is not None:
            dot += wq * wd
    return dot / (norm_q * norm_d)


def score_section(question: str, sec_id: str, sec: dict) -> float:
    vec_q, norm_q = build_query_vector(question)
    vec_d = SITE["doc_vecs"].get(sec_id, {})
    norm_d = SITE["doc_norms"].get(sec_id, 1.0)
    base = cosine_sim(vec_q, norm_q, vec_d, norm_d)

    # alias boost: e.g. user says "projects" and this is the projects section
    q_tokens = set(tokenize(question))
    aliases = SECTION_ALIASES.get(sec_id, set())
    if q_tokens & aliases:
        base += 0.2

    return base


def find_best_sections(question: str, top_k: int = 2):
    if not SITE["sections"]:
        return []
    scored = []
    for sec_id, sec in SITE["sections"].items():
        s = score_section(question, sec_id, sec)
        if s > 0:
            scored.append((s, sec))
    if not scored:
        return []
    scored.sort(key=lambda x: x[0], reverse=True)
    return [sec for _, sec in scored[:top_k]]

# ---------------------------
# Section summarisation helpers
# ---------------------------

def section_kind(sec_id: str, heading: str) -> str:
    """Classify a section into a simple kind for nicer wording."""
    h = heading.lower()
    if sec_id == "skills" or "skill" in h:
        return "skills"
    if sec_id == "projects" or "experience" in h or "projects" in h:
        return "projects"
    if sec_id == "education" or "education" in h:
        return "education"
    if sec_id == "contact" or "contact" in h:
        return "contact"
    if sec_id == "interactive-apps" or "apps" in h or "interactive" in h:
        return "apps"
    if sec_id == "about" or "about" in h:
        return "about"
    if sec_id == "hero":
        return "hero"
    return "other"


def summarize_section_text(sec: dict, max_chars: int = 260) -> str:
    sentences = sec.get("sentences") or [sec.get("text", "")]
    text = " ".join(sentences[:3])
    return textwrap.shorten(text, width=max_chars, placeholder="…")


def build_section_answer(sec: dict, primary: bool) -> str:
    """Generate a short, explanation-style answer for one section."""
    sid = sec["id"]
    heading = sec["heading"]
    kind = section_kind(sid, heading)
    summary = summarize_section_text(sec)

    if kind == "skills":
        prefix = (
            "Here’s a short summary of his skills and tools:\n"
            if primary else
            "Also, regarding his skills:\n"
        )
    elif kind == "projects":
        prefix = (
            "Here’s a quick summary of his projects and experience:\n"
            if primary else
            "Also, from his projects and experience:\n"
        )
    elif kind == "education":
        prefix = (
            "Here’s a summary of his education background:\n"
            if primary else
            "Also, about his education:\n"
        )
    elif kind == "contact":
        prefix = (
            "Here’s how you can reach him:\n"
            if primary else
            "Also, for contacting him:\n"
        )
    elif kind == "apps":
        prefix = (
            "Here’s a quick overview of his interactive apps:\n"
            if primary else
            "Also, from the interactive apps:\n"
        )
    elif kind == "about":
        prefix = (
            "Here’s a short summary from the About section:\n"
            if primary else
            "Also, from the About section:\n"
        )
    elif kind == "hero":
        prefix = (
            "At a glance, this is what the page highlights:\n"
            if primary else
            "Also, from the top section:\n"
        )
    else:
        prefix = (
            f"Here’s a short summary from the {heading} section:\n"
            if primary else
            f"Also, from the {heading} section:\n"
        )

    tail = "\n\nIf you’d like more detail on this, just ask me to expand on it 🙂"
    return prefix + summary + (tail if primary else "")

# ---------------------------
# Answer builders
# ---------------------------

def build_answer_from_sections(question: str) -> str:
    if not SITE["sections"]:
        return (
            "I’m supposed to answer using the content of this portfolio, "
            "but I couldn’t load the page text right now."
        )

    best_secs = find_best_sections(question, top_k=2)
    if not best_secs:
        return (
            "I’m designed to answer questions about this portfolio only. "
            "I couldn’t find anything related to that in the page content."
        )

    # Avoid adding hero as a noisy second section
    if len(best_secs) > 1 and best_secs[0]["id"] != "hero":
        filtered = [best_secs[0]] + [
            sec for sec in best_secs[1:] if sec["id"] != "hero"
        ]
        best_secs = filtered[:2]

    # Remember which sections we used, so we can expand later
    try:
        st.session_state["last_sections"] = [sec["id"] for sec in best_secs]
    except Exception:
        pass

    main = build_section_answer(best_secs[0], primary=True)
    if len(best_secs) == 1:
        return main

    extra = build_section_answer(best_secs[1], primary=False)
    return main + "\n\n" + extra


def build_overview_answer() -> str:
    """Give a high-level summary of the whole portfolio page."""
    if not SITE["sections"]:
        return (
            "I’m supposed to answer using the content of this portfolio, "
            "but I couldn’t load the page text right now."
        )

    parts = []
    order = ["hero", "about", "projects", "skills", "education", "contact"]

    def add_sec(sec_id, label):
        sec = SITE["sections"].get(sec_id)
        if not sec:
            return
        short = summarize_section_text(sec, max_chars=260)
        parts.append(f"{label}: {short}")

    add_sec("hero", "At a glance")
    add_sec("about", "About Kanishq")
    add_sec("projects", "Projects & experience")
    add_sec("skills", "Skills & tools")
    add_sec("education", "Education")
    add_sec("contact", "How to contact")

    # Remember overview sections as context for "tell me more"
    try:
        st.session_state["last_sections"] = [sid for sid in order if sid in SITE["sections"]]
    except Exception:
        pass

    if not parts:
        text = textwrap.shorten(SITE["all_text"], width=480, placeholder="…")
        return f"Here’s a quick overview of this portfolio:\n\n{text}"

    return "Here’s a quick overview of this portfolio:\n\n" + "\n\n".join(parts)


def build_expand_answer() -> str:
    """Give more detail about the last sections Asha talked about."""
    if not SITE["sections"]:
        return (
            "I’m supposed to answer using this portfolio, "
            "but I couldn’t load the page text right now."
        )

    try:
        sec_ids = st.session_state.get("last_sections")
    except Exception:
        sec_ids = None

    if not sec_ids:
        # No previous context → fall back to a general overview
        return build_overview_answer()

    blocks = []
    for sid in sec_ids[:2]:  # expand at most two sections
        sec = SITE["sections"].get(sid)
        if not sec:
            continue

        heading = sec["heading"]
        kind = section_kind(sid, heading)
        sentences = sec.get("sentences") or [sec.get("text", "")]
        text = " ".join(sentences[:6])
        text = textwrap.shorten(text, width=520, placeholder="…")

        if kind == "skills":
            label = "Here’s more detail on his skills:"
        elif kind == "projects":
            label = "Here’s more detail on his projects and experience:"
        elif kind == "education":
            label = "Here’s more detail on his education:"
        elif kind == "contact":
            label = "Here’s more detail on how to contact him:"
        elif kind == "apps":
            label = "Here’s more detail on his interactive apps:"
        elif kind == "about":
            label = "Here’s more detail from the About section:"
        else:
            label = f"Here’s more detail from {heading}:"

        blocks.append(f"{label}\n{text}")

    if not blocks:
        return build_overview_answer()

    return "\n\n".join(blocks)

# ---------------------------
# Phrase matching helper
# ---------------------------

def contains_any_phrase(text: str, phrases) -> bool:
    """
    Check if `text` contains any phrase from `phrases`.

    Single-word phrases must match whole words (using word boundaries),
    multi-word phrases can be simple substring matches.
    """
    for p in phrases:
        p = p.lower()
        if " " in p:
            if p in text:
                return True
        else:
            if re.search(r"\b" + re.escape(p) + r"\b", text):
                return True
    return False

# ---------------------------
# Main reply router
# ---------------------------

def generate_asha_reply(user_text: str) -> str:
    """Top-level reply generator using our TF-IDF retrieval model."""
    if not user_text or not user_text.strip():
        return "You can ask me about Kanishq’s skills, projects, education, or how to contact him 😊"

    low = user_text.strip().lower()

    # Greetings
    greeting_phrases = [
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening"
    ]
    if contains_any_phrase(low, greeting_phrases) and len(low) <= 40:
        return (
            "Hey there 😊\n\n"
            "I’m Asha. I can tell you about his skills, projects, education, "
            "or how to contact him. What would you like to know first?"
        )

    # Thanks
    thanks_phrases = ["thank you", "thanks", "thx", "tysm", "ty"]
    if contains_any_phrase(low, thanks_phrases):
        return "You’re very welcome 🤍 If you want to know more about anything on this portfolio, just ask."

    # Farewell
    bye_phrases = ["bye", "goodbye", "see you", "see ya", "good night"]
    if contains_any_phrase(low, bye_phrases):
        return "Thanks for dropping by! If you have more questions about his work or skills, you can always come back 😊"

    # Small talk / meta
    if any(x in low for x in ["who are you", "what are you", "your name"]):
        return (
            f"I’m {CHARACTER_NAME}, a small chatbot built into this portfolio. "
            "I answer questions using the content that’s written here."
        )

    if "what can you do" in low or "help me" in low:
        return (
            "I can explain different parts of this site — things like Kanishq’s background, "
            "projects, skills, education, and how to contact him — all based on the page content."
        )

    # Follow-up: "tell me more", "go deeper", etc.
    expand_phrases = [
        "tell me more",
        "can you explain more",
        "explain more",
        "more details",
        "give me more details",
        "more detail",
        "expand this",
        "expand on that",
        "go deeper",
        "elaborate",
        "elaborate more",
    ]
    if contains_any_phrase(low, expand_phrases):
        return build_expand_answer()

    # Generic “overview / everything / summary” questions
    generic_overview_phrases = [
        "tell everything",
        "tell me everything",
        "everything about",
        "give me everything",
        "overview",
        "summary",
        "summarise",
        "summarize",
        "what do you do",
        "what does he do",
        "about you",
        "about kanishq",
        "what is this website about",
        "what is this site about",
        "explain this website",
        "describe this website",
    ]
    if contains_any_phrase(low, generic_overview_phrases):
        return build_overview_answer()

    # Otherwise, answer from page content using our TF-IDF model
    return build_answer_from_sections(user_text)

# ---------------------------
# Session state & UI
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Show intro once at the very beginning
    st.session_state.messages.append(
        {"role": "assistant", "content": INTRO_MESSAGE}
    )

st.title("🤖 Asha – Portfolio Chatbot")
st.caption("Ask questions about this portfolio and I’ll answer using only what’s written here.")

# Render chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Sidebar: status + live RAM stats
with st.sidebar:
    if not SITE["sections"]:
        st.error("I’m having trouble loading the page content right now.")
        if SITE.get("debug"):
            with st.expander("Debug info (developer only)"):
                for line in SITE["debug"]:
                    st.write(f"- {line}")
    else:
        st.markdown("**Asha is online and ready to chat.**")

    proc = psutil.Process(os.getpid())
    mem_used = proc.memory_info().rss / (1024 ** 2)
    st.write(f"Approx. memory in use: **{mem_used:.1f} MB**")

# ---------------------------
# Suggestion buttons + chat input
# ---------------------------

suggestion_clicked = None

suggestions = [
    "Give me an overview",
    "What are his skills?",
    "Show me his projects",
    "Tell me about his education",
    "How can I contact him?",
]

st.markdown("##### Not sure what to ask? Try one of these:")

cols = st.columns(len(suggestions))
for col, text in zip(cols, suggestions):
    if col.button(text, use_container_width=True):
        suggestion_clicked = text

# Chat input is ALWAYS rendered, even when a button is clicked
typed = st.chat_input("Ask something about this portfolio…")

if typed:
    user_input = typed
elif suggestion_clicked is not None:
    user_input = suggestion_clicked
else:
    user_input = None

# ---------------------------
# Handle user input (typed or clicked)
# ---------------------------

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate reply with our TF-IDF model
    reply = generate_asha_reply(user_input)

    # Add assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    # Trim long histories just in case
    if len(st.session_state.messages) > 40:
        st.session_state.messages = st.session_state.messages[-40:]
