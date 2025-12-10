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
    "I’m here to help you explore this portfolio site and answer questions "
    "about Kanishq Reddy, his experience, skills, and projects — based only "
    "on what’s written on this page."
)

STOPWORDS = {
    "the", "and", "a", "an", "of", "to", "in", "on", "for", "with",
    "is", "are", "was", "were", "it", "this", "that", "as", "at",
    "by", "from", "or", "be", "about", "my", "i", "you", "your",
    "me", "we", "our", "they", "their", "he", "she", "his", "her",
    "them"
}

# URL of the live portfolio site (index.html)
# You can change this to the raw GitHub URL if you want:
#   https://raw.githubusercontent.com/kanishqreddy/kanishqreddy.github.io/main/index.html
SITE_URL = os.environ.get("PORTFOLIO_URL", "https://kanishqreddy.github.io/")


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
# Load & index remote index.html + build TF-IDF model
# ---------------------------

@st.cache_resource
def load_site_knowledge():
    debug = []
    html_text = None
    html_source = None

    # 1) Try downloading from the live portfolio URL
    try:
        debug.append(f"Trying SITE_URL={SITE_URL}")
        resp = requests.get(SITE_URL, timeout=5)
        debug.append(f"SITE_URL status={resp.status_code}")
        if resp.status_code == 200:
            html_text = resp.text
            html_source = SITE_URL
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
            debug.append(f"Try local {candidate} exists={os.path.exists(candidate)}")
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as f:
                        html_text = f.read()
                    html_source = candidate
                    break
                except Exception as e:
                    debug.append(f"Error reading {candidate}: {e}")

    if html_text is None:
        # Nothing worked – return empty skeleton + debug info
        return {
            "source": None,
            "sections": {},
            "all_text": "",
            "idf": {},
            "doc_vecs": {},
            "doc_norms": {},
            "debug": debug,
        }

    # Extract <section id="..."> blocks to keep structure
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

    debug.append(f"Loaded {len(sections)} sections from {html_source}")

    return {
        "source": html_source,
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


def build_answer_from_sections(question: str):
    if not SITE["sections"]:
        return (
            "I’m supposed to answer using the content of the portfolio website, "
            "but I couldn’t load the index.html either from the live URL or locally. "
            "Please check that the site is reachable at "
            f"{SITE_URL!r} and that Render allows outbound HTTP."
        )

    best_secs = find_best_sections(question, top_k=2)
    if not best_secs:
        return (
            "I’m designed to answer questions about this portfolio site only. "
            "I couldn’t find anything on this page related to that question."
        )

    parts = []
    for sec in best_secs:
        sentences = sec["sentences"] or [sec["text"]]
        text = " ".join(sentences[:4])
        text = textwrap.shorten(text, width=420, placeholder="…")
        sid = sec["id"]

        if sid == "about":
            prefix = "From the About section: "
        elif sid == "projects":
            prefix = "From the Projects & Experience section: "
        elif sid == "skills":
            prefix = "From the Skills & Tools section: "
        elif sid == "education":
            prefix = "From the Education section: "
        elif sid == "contact":
            prefix = "From the Contact section: "
        elif sid == "interactive-apps":
            prefix = "From the Interactive Apps section: "
        elif sid == "hero":
            prefix = "From the top hero section: "
        else:
            prefix = f"From the {sec['heading']} section: "

        parts.append(prefix + text)

    if len(parts) == 1:
        return parts[0]
    return parts[0] + "\n\nAlso, " + parts[1]


def generate_asha_reply(user_text: str) -> str:
    """Top-level reply generator using our TF-IDF retrieval model."""
    if not user_text or not user_text.strip():
        return "You can ask me about Kanishq’s skills, projects, education, or how to contact him 😊"

    low = user_text.strip().lower()
    if "who are you" in low or "what are you" in low or "your name" in low:
        return (
            f"I’m {CHARACTER_NAME}, a small chatbot built into this portfolio. "
            "I answer questions using the content that’s written on this page."
        )

    if "what can you do" in low or "help me" in low:
        return (
            "I can explain sections of this site — things like Kanishq’s background, "
            "projects, skills, education, and contact details — all based on the page content."
        )

    # Otherwise, answer from website content using our TF-IDF model
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
st.caption("Ask questions about this website and I’ll answer using only what’s written here.")

# Render chat history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Sidebar info
st.sidebar.header("Site status")
if SITE["source"]:
    st.sidebar.write(f"Loaded from: `{SITE['source']}`")
    st.sidebar.write(f"Sections indexed: {len(SITE['sections'])}")
else:
    st.sidebar.error("Could not load index.html from remote URL or local file.")
    # Show debug info so you can see what went wrong on Render
    if "debug" in SITE:
        st.sidebar.write("Debug info:")
        for line in SITE["debug"]:
            st.sidebar.write(f"- {line}")

# Memory usage (for reassurance)
proc = psutil.Process(os.getpid())
mem_used = proc.memory_info().rss / (1024 ** 2)
st.sidebar.write(f"Approx. memory in use: {mem_used:.1f} MB (limit ~500MB, no heavy ML libs)")

# User input
user_input = st.chat_input("Ask something about this portfolio…")

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
