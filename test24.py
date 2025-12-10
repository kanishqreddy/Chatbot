import re
import os
import textwrap

import streamlit as st
import psutil
import torch
from transformers import AutoTokenizer, AutoModel


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

# Tiny encoder model (very small)
MODEL_NAME = "prajjwal1/bert-tiny"
DEVICE = torch.device("cpu")


# ---------------------------
# HTML / text utilities
# ---------------------------

def clean_html_text(html: str) -> str:
    """Strip tags and script/style from HTML and normalise whitespace."""
    if not html:
        return ""
    # comments, scripts, styles
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
    # tags
    html = re.sub(r"<[^>]+>", " ", html)
    # entities + whitespace
    html = re.sub(r"&nbsp;?", " ", html, flags=re.IGNORECASE)
    html = re.sub(r"\s+", " ", html)
    return html.strip()


def split_sentences(text: str):
    """Very small sentence splitter."""
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def extract_keywords(text: str):
    words = re.findall(r"\b\w+\b", text.lower())
    return {w for w in words if len(w) > 2 and w not in STOPWORDS}


# ---------------------------
# Load model (tiny encoder)
# ---------------------------

@st.cache_resource
def load_model():
    """
    Load a very small transformer encoder (bert-tiny) to stay under
    the 500MB RAM limit. This is used for semantic similarity only,
    not for text generation.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    except Exception as e:
        # If anything goes wrong, fall back to no-model mode
        st.sidebar.error(f"Could not load model ({MODEL_NAME}): {e}")
        return None, None


def encode_text(text: str, tokenizer, model):
    """Get a single vector embedding for a piece of text using mean pooling."""
    if not text:
        return None
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    encoded = {k: v.to(DEVICE) for k, v in encoded.items()}
    with torch.no_grad():
        out = model(**encoded)
    token_emb = out.last_hidden_state  # (1, L, H)
    mask = encoded["attention_mask"].unsqueeze(-1)  # (1, L, 1)
    masked = token_emb * mask
    summed = masked.sum(dim=1)          # (1, H)
    counts = mask.sum(dim=1).clamp(min=1)  # (1, 1)
    mean = summed / counts
    return mean[0].cpu()  # (H,)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a is None or b is None:
        return 0.0
    a_norm = a / (a.norm() + 1e-8)
    b_norm = b / (b.norm() + 1e-8)
    return float((a_norm * b_norm).sum().item())


# ---------------------------
# Load & index index.html
# ---------------------------

@st.cache_resource
def load_site_knowledge():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "index.html"),
        os.path.join(base_dir, "public", "index.html"),
        os.path.join(base_dir, "static", "index.html"),
    ]

    html_text = None
    html_path = None
    for path in candidates:
        if os.path.exists(path):
            html_path = path
            try:
                with open(path, "r", encoding="utf-8") as f:
                    html_text = f.read()
                break
            except Exception:
                continue

    if html_text is None:
        return {
            "path": None,
            "sections": {},
            "all_text": "",
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
            # Fallback to id as human label
            heading = sec_id.replace("-", " ").title()

        sections[sec_id] = {
            "id": sec_id,
            "heading": heading,
            "text": clean_text,
            "sentences": sentences,
            "keywords": extract_keywords(clean_text),
        }

    all_text = clean_html_text(html_text)

    return {
        "path": html_path,
        "sections": sections,
        "all_text": all_text,
    }


SITE = load_site_knowledge()
TOKENIZER, MODEL = load_model()


# ---------------------------
# Retrieval-based answering with model
# ---------------------------

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
}


def lexical_score(question_keywords, section):
    """Simple overlap-based score."""
    if not section["keywords"]:
        return 0.0
    overlap = question_keywords & section["keywords"]
    return float(len(overlap))


def score_section(question: str, section) -> float:
    """
    Combined lexical + semantic similarity score.
    If the model is unavailable, fall back to lexical only.
    """
    q_keywords = extract_keywords(question)
    lex = lexical_score(q_keywords, section)

    # Alias boost: if question mentions a typical word for this section, bump lexical
    for sec_id, alias_words in SECTION_ALIASES.items():
        if section["id"] == sec_id and q_keywords & alias_words:
            lex += 1.5
            break

    if TOKENIZER is None or MODEL is None:
        return lex

    # Semantic similarity using tiny BERT
    section_for_model = " ".join(section["sentences"][:4]) or section["text"]
    section_for_model = section_for_model[:512]

    try:
        q_vec = encode_text(question, TOKENIZER, MODEL)
        s_vec = encode_text(section_for_model, TOKENIZER, MODEL)
        sem = cosine_similarity(q_vec, s_vec)  # in [-1, 1]
        # Map to roughly 0–2 and combine with lexical
        sem = max(sem, 0.0) * 2.0
    except Exception:
        sem = 0.0

    return lex * 0.7 + sem * 1.3


def find_best_sections(question: str, top_k: int = 2):
    if not SITE["sections"]:
        return []

    scored = []
    for sec in SITE["sections"].values():
        s = score_section(question, sec)
        if s > 0:
            scored.append((s, sec))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    return [sec for _, sec in scored[:top_k]]


def build_answer_from_sections(question: str):
    # If no site loaded, short message
    if not SITE["sections"]:
        return (
            "I’m supposed to answer using the content of this website, "
            "but I couldn’t load the page on the server. "
            "You might want to refresh or contact the site owner."
        )

    best_sections = find_best_sections(question, top_k=2)

    if not best_sections:
        return (
            "I’m designed to answer questions about this portfolio site only. "
            "I couldn’t find anything on this page related to that question."
        )

    # Build a concise answer from 1–2 sections
    parts = []
    for sec in best_sections:
        text = " ".join(sec["sentences"][:4])
        # Limit length
        text = textwrap.shorten(text, width=420, placeholder="…")
        if sec["id"] == "about":
            prefix = "From the About section: "
        elif sec["id"] == "projects":
            prefix = "From the Projects & Experience section: "
        elif sec["id"] == "skills":
            prefix = "From the Skills & Tools section: "
        elif sec["id"] == "education":
            prefix = "From the Education section: "
        elif sec["id"] == "contact":
            prefix = "From the Contact section: "
        elif sec["id"] == "interactive-apps":
            prefix = "From the Interactive Apps section: "
        else:
            prefix = f"From the {sec['heading']} section: "

        parts.append(prefix + text)

    if len(parts) == 1:
        return parts[0]
    else:
        return parts[0] + "\n\nAlso, " + parts[1]


def generate_asha_reply(user_text: str) -> str:
    """Top-level reply generator using a tiny model for retrieval."""
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

    # Otherwise, answer from website content using model-powered retrieval
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
if SITE["path"]:
    st.sidebar.write(f"Page loaded from: `{os.path.basename(SITE['path'])}`")
    st.sidebar.write(f"Sections indexed: {len(SITE['sections'])}")
else:
    st.sidebar.error("index.html could not be found on the server.")

if TOKENIZER is not None and MODEL is not None:
    st.sidebar.write(f"Model: `{MODEL_NAME}` (tiny encoder for retrieval)")
else:
    st.sidebar.write("Model: unavailable, using lexical matching only.")

# Memory usage (for reassurance)
proc = psutil.Process(os.getpid())
mem_used = proc.memory_info().rss / (1024 ** 2)
st.sidebar.write(f"Approx. memory in use: {mem_used:.1f} MB (limit ~500MB)")

# User input
user_input = st.chat_input("Ask something about this portfolio…")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Generate reply (model-assisted retrieval)
    reply = generate_asha_reply(user_input)

    # Add assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    # Optionally trim very long histories
    if len(st.session_state.messages) > 40:
        st.session_state.messages = st.session_state.messages[-40:]
