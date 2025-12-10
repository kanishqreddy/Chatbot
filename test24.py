import re
import os
import psutil
import torch
import streamlit as st
from transformers import BlenderbotSmallTokenizer, BlenderbotSmallForConditionalGeneration

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="Asha – Website Guide", layout="centered")

# ---------------------------
# Character / Personality
# ---------------------------
CHARACTER_NAME = "Asha"
CHARACTER_PERSONALITY = (
    f"{CHARACTER_NAME} is a friendly, knowledgeable guide for Kanishq Reddy’s portfolio website. "
    "She answers questions ONLY using the information that is actually on the portfolio page. "
    "If something is not in the website, she clearly says she doesn’t know instead of guessing. "
    "Her tone is warm, concise, and professional, like a helpful teammate walking you through the site. "
    "She can use light emojis sometimes (😊, 📊, 💻) but keeps the focus on clarity."
)

# ---------------------------
# Load and index website content
# ---------------------------

STOPWORDS = {
    "the","and","a","an","of","to","in","on","for","with","is","are","was","were",
    "it","this","that","as","at","by","from","or","be","about","my","i","you",
    "your","me","we","our","they","their","he","she","his","her","them"
}

@st.cache_resource
def load_site_knowledge():
    """Load index.html from disk and build a simple text index.

    The model will only be allowed to answer using this content.
    """  # noqa: D401
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
        # Fallback: no file found – return empty index
        return {
            "path": None,
            "sentences": [],
            "indexed": [],
            "raw": ""
        }

    # Remove comments, scripts, styles
    cleaned = re.sub(r"<!--.*?-->", " ", html_text, flags=re.DOTALL)
    cleaned = re.sub(
        r"<script[^>]*>.*?</script>",
        " ",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(
        r"<style[^>]*>.*?</style>",
        " ",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )
    # Strip tags
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"&nbsp;?", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Split into sentence-ish chunks
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    sentences = [p.strip() for p in parts if len(p.strip()) > 25]

    indexed = []
    for s in sentences:
        words = set(w.lower() for w in re.findall(r"\b\w+\b", s))
        indexed.append(
            {
                "text": s,
                "words": words,
            }
        )

    return {
        "path": html_path,
        "sentences": sentences,
        "indexed": indexed,
        "raw": cleaned,
    }


SITE_KNOWLEDGE = load_site_knowledge()

def find_relevant_context(question: str, max_sentences: int = 8):
    """Pick the most relevant sentences from the website for a question."""  # noqa: D401
    if not SITE_KNOWLEDGE["indexed"]:
        return "", 0

    q_words = set(
        w.lower()
        for w in re.findall(r"\b\w+\b", question)
    )
    q_words = {w for w in q_words if w not in STOPWORDS}

    if not q_words:
        # Nothing meaningful to match – just return a short site summary
        core = SITE_KNOWLEDGE["sentences"][:max_sentences]
        return "\n".join(core), 0

    scored = []
    for item in SITE_KNOWLEDGE["indexed"]:
        overlap = q_words & (item["words"] - STOPWORDS)
        score = len(overlap)
        if score > 0:
            scored.append((score, item["text"]))

    if not scored:
        # No overlapping words – very likely out-of-scope question
        return "", 0

    scored.sort(key=lambda x: x[0], reverse=True)
    top = [text for _, text in scored[:max_sentences]]
    best_score = scored[0][0]
    context = "\n".join(top)
    # Also trim extremely long context just in case
    if len(context) > 2500:
        context = context[:2500]
    return context, best_score

# ---------------------------
# Model loading (small / memory friendly)
# ---------------------------

MODEL_NAME = "facebook/blenderbot_small-90M"
DEVICE = torch.device("cpu")  # Render free tier is CPU-only

@st.cache_resource
def load_model():
    tokenizer = BlenderbotSmallTokenizer.from_pretrained(MODEL_NAME)
    model = BlenderbotSmallForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------
# Session state
# ---------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------
# Helpers
# ---------------------------

def sanitize_model_output(text: str):
    text = text.strip()

    # Strip speaker labels
    for p in [r"Asha:\s*", r"Assistant:\s*", r"Bot:\s*"]:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            text = text[m.end():].strip()
            break

    # If the model starts writing another user turn, cut it off
    for tag in ["\nUser:", "\nHuman:"]:
        if tag in text:
            text = text.split(tag)[0].strip()

    # Remove bracketed directions like [laughs]
    text = re.sub(r"\[[^\]]+\]", "", text)

    # Avoid repeating the last assistant message exactly
    last_assistant = next(
        (m["content"] for m in reversed(st.session_state.messages)
         if m["role"] == "assistant"),
        ""
    )
    if text == last_assistant:
        return None

    return re.sub(r"\s+\n", "\n", text).strip()

def build_prompt(user_text: str, context: str, has_good_context: bool) -> str:
    # Keep only last 6 messages for flavour (not for facts)
    history = st.session_state.messages[-6:]
    history_lines = [
        f"{'User' if m['role']=='user' else CHARACTER_NAME}: {m['content'].strip()}"
        for m in history
    ]
    history_section = "\n".join(history_lines)

    scope_instruction = (
        "You must answer ONLY using the information from the WEBSITE CONTEXT. "
        "If the answer is not clearly supported by that context, say you don’t know "
        "and remind the user you only know the content of the portfolio page."
    )

    if not SITE_KNOWLEDGE["indexed"]:
        context_block = (
            "WEBSITE CONTEXT:\n"
            "(No website content could be loaded on the server – explain this briefly "
            "and ask the user to contact the site owner.)"
        )
    elif not has_good_context:
        # User probably asked something unrelated to the site content
        first_sentence = SITE_KNOWLEDGE["sentences"][0] if SITE_KNOWLEDGE["sentences"] else ""
        context_block = (
            "WEBSITE CONTEXT:\n"
            f"{first_sentence}\n\n"
            "The question seems unrelated to the portfolio content."
        )
        scope_instruction = (
            "The user's question does NOT match the website content. "
            "Politely explain that you only answer questions about the portfolio "
            "website (experience, skills, projects, contact info, education, etc.). "
            "Do NOT hallucinate information."
        )
    else:
        context_block = f"WEBSITE CONTEXT (from the portfolio page):\n{context}"

    instructions = f"""ROLE:
{CHARACTER_PERSONALITY}

SCOPE:
{scope_instruction}

STYLE:
- Speak as {CHARACTER_NAME}, a helpful guide to Kanishq Reddy's portfolio.
- Be clear and concise (1–4 sentences per reply).
- When referring to Kanishq, use third person ("he") unless the user speaks as Kanishq.
- If the user asks where something is on the page, describe the section (e.g., About, Skills, Projects, Certifications, Contact).
"""

    prompt_parts = [
        instructions,
        context_block,
        "RECENT CHAT:",
        history_section,
        "Now answer the user's latest question based ONLY on the website context.",
        f"User: {user_text}",
        f"{CHARACTER_NAME}:",
    ]

    prompt = "\n\n".join(p for p in prompt_parts if p)
    # keep prompt short for memory / speed
    return prompt[:4000]

def generate_reply(user_text: str):
    context, best_score = find_relevant_context(user_text)
    has_good_context = best_score > 0

    prompt = build_prompt(user_text, context, has_good_context)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=96,
        do_sample=True,
        top_p=0.90,
        temperature=0.75,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id,
    )

    for _ in range(3):
        try:
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
            decoded = tokenizer.decode(output[0], skip_special_tokens=True)
            reply = sanitize_model_output(decoded)
            if reply:
                return reply
        except Exception:
            continue

    return (
        "I’m having a small issue reading the website content right now 😅. "
        "Please try again in a moment, or check the portfolio directly."
    )

def maybe_trim_history(max_messages: int = 20):
    """Trim chat history if memory usage creeps up."""  # noqa: D401
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 ** 2)

    if mem_mb > 450:
        # Keep only the last few turns
        st.session_state.messages = st.session_state.messages[-max_messages:]


# ---------------------------
# UI
# ---------------------------

st.title("🤖 Asha – Portfolio Chatbot")
st.caption("Ask me anything about this website and Kanishq Reddy’s portfolio.")

# Show previous messages
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Display basic site status in sidebar
st.sidebar.header("Site Knowledge")
if SITE_KNOWLEDGE["path"]:
    st.sidebar.write(f"Loaded from: `{os.path.basename(SITE_KNOWLEDGE['path'])}`")
    st.sidebar.write(f"Indexed sentences: {len(SITE_KNOWLEDGE['sentences'])}")
else:
    st.sidebar.warning(
        "Couldn't load index.html on the server. "
        "Asha may only be able to answer in very general terms."
    )

# Memory info (lightweight)
proc = psutil.Process(os.getpid())
mem_used = proc.memory_info().rss / (1024 ** 2)
st.sidebar.write(f"Memory in use: {mem_used:.2f} MB (Render free tier ≈ 500 MB)")

# Input
user_input = st.chat_input("Ask something about this portfolio website…")

# ---------------------------
# Handle User Input
# ---------------------------

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    try:
        reply = generate_reply(user_input)
    except Exception as e:
        st.error(f"Error while generating reply: {e}")
        reply = (
            "Something went wrong on my side 😅. "
            "Please refresh and try asking again."
        )

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)

    # Update sidebar resource usage & maybe trim
    proc = psutil.Process(os.getpid())
    mem_used = proc.memory_info().rss / (1024 ** 2)
    st.sidebar.write(f"Memory in use: {mem_used:.2f} MB")
    maybe_trim_history()
