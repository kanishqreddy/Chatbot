import re
import os
import psutil
import torch
import streamlit as st
from packaging import version
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration

# ---------------------------
# Streamlit config
# ---------------------------
st.set_page_config(page_title="Asha – Chatbot", layout="centered")

# ---------------------------
# Character / Personality
# ---------------------------
CHARACTER_NAME = "Asha"
CHARACTER_PERSONALITY = (
    f"{CHARACTER_NAME} is warm, empathetic, and naturally conversational. "
    "She replies like a caring friend — concise, human, occasionally playful, and curious. "
    "She NEVER mentions being an AI, system details, or constraints. "
    "Her tone adapts to the user's emotional state. "
    "She uses emojis naturally (😊, 🤍, 😅) depending on context."
)

FEW_SHOT = """
User: hi
Asha: Hey — nice to hear from you 😊 How’s your day going?

User: I'm stressed about exams.
Asha: That does sound heavy… want to tell me what part feels most overwhelming?

User: I'm a student, not working.
Asha: Makes sense — being a student is a full-time thing on its own. What are you studying?
"""

# ---------------------------
# Model options & UI
# ---------------------------
MODEL_OPTIONS = {
    "Fast (400M)": "facebook/blenderbot-400M-distill",
    "Balanced (1B)": "facebook/blenderbot-1B-distill",
    "Best Quality (3B)": "facebook/blenderbot-3B",
}

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.radio("Choose model:", list(MODEL_OPTIONS.keys()), index=1)
auto_switch = st.sidebar.checkbox("Auto-switch model if needed", value=True)
max_history_messages = st.sidebar.number_input(
    "Max visible messages context", min_value=3, max_value=12, value=6
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Model loader
# ---------------------------
@st.cache_resource
def load_model(model_name):
    if version.parse(torch.__version__) < version.parse("2.6.0"):
        raise RuntimeError("PyTorch >= 2.6.0 required.")
    tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    try:
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name, use_safetensors=True)
    except:
        model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ---------------------------
# Session state
# ---------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "tokenizer" not in st.session_state or "model" not in st.session_state:
    st.session_state.tokenizer, st.session_state.model = load_model(MODEL_OPTIONS[model_choice])
    st.session_state.active_model = model_choice

# ---------------------------
# UI
# ---------------------------
st.title("💬 Chat with Asha")
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])
user_input = st.chat_input("Say something to Asha…")

# ---------------------------
# Helpers
# ---------------------------
def detect_emotion(user_text):
    text = user_text.lower()
    if any(w in text for w in ["stressed", "anxious", "worried", "nervous"]):
        return "stressed"
    if any(w in text for w in ["happy", "glad", "excited", "good", "awesome"]):
        return "happy"
    if any(w in text for w in ["bored", "lonely", "nothing to do", "tired"]):
        return "bored"
    if any(w in text for w in ["sad", "down", "depressed", "unhappy"]):
        return "sad"
    return "neutral"

def sanitize_model_output(text):
    text = text.strip()
    for p in [r"Asha:\s*", r"Assistant:\s*", r"Bot:\s*"]:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            text = text[m.end():].strip()
            break
    stop_tags = ["\nUser:", "\nHuman:"]
    for tag in stop_tags:
        if tag in text:
            text = text.split(tag)[0].strip()
    text = re.sub(r"\[[^\]]+\]", "", text)
    last_assistant = next((m["content"] for m in reversed(st.session_state.messages) if m["role"]=="assistant"), "")
    # Only reject if identical to last reply
    if text == last_assistant:
        return None
    return re.sub(r"\s+\n", "\n", text).strip()

def build_prompt(user_text, emotion="neutral"):
    user_text = " ".join(user_text.split()[-50:])
    recent = st.session_state.messages[-max_history_messages:]
    history_lines = [
        f"{'User' if m['role']=='user' else CHARACTER_NAME}: {m['content'].strip()}"
        for m in recent
    ]
    history_section = "\n".join(history_lines)
    tones = {
        "stressed": "Be calm, soothing, and reassuring. Use gentle emojis.",
        "happy": "Be cheerful, upbeat, and playful. Add light emojis if appropriate.",
        "bored": "Be curious, engaging, suggest fun activities or questions. Light humor okay.",
        "sad": "Be empathetic, supportive, and comforting. Avoid jokes.",
        "neutral": "Be warm, friendly, and naturally conversational."
    }
    tone = tones.get(emotion, "neutral")
    instruction = f"""
INSTRUCTIONS (for {CHARACTER_NAME}):
- Reply naturally to the user's latest message.
- Always reply with warmth and curiosity, even if the user says very little.
- Never just say "tell me more". Expand naturally with reflections, questions, or observations.
- Avoid repeating the same phrasing across turns.
- Keep responses 1–4 sentences unless more detail is clearly needed.
- Respond with authenticity. {tone}
- Never mention being an AI or system constraints.
"""
    prompt_parts = [
        CHARACTER_PERSONALITY,
        "STYLE EXAMPLES:",
        FEW_SHOT,
        "VISIBLE CONVERSATION:",
        history_section,
        instruction,
        f"User: {user_text}",
        f"{CHARACTER_NAME}:"
    ]
    return "\n\n".join([p for p in prompt_parts if p])

def generate_reply(user_text):
    tokenizer = st.session_state.tokenizer
    model = st.session_state.model
    active_model = st.session_state.active_model
    emotion = detect_emotion(user_text)

    prompt = build_prompt(user_text, emotion=emotion)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # Auto model upgrade
    input_len = inputs["input_ids"].shape[1]
    if auto_switch:
        try:
            if active_model == "Fast (400M)" and input_len > 120:
                st.session_state.tokenizer, st.session_state.model = load_model(MODEL_OPTIONS["Balanced (1B)"])
                st.session_state.active_model = "Balanced (1B)"
            elif active_model == "Balanced (1B)" and input_len > 650:
                st.session_state.tokenizer, st.session_state.model = load_model(MODEL_OPTIONS["Best Quality (3B)"])
                st.session_state.active_model = "Best Quality (3B)"
        except Exception:
            pass

    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=200,
        do_sample=True,
        top_p=0.92,
        temperature=0.78,
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        pad_token_id=tokenizer.eos_token_id
    )

    for _ in range(3):
        try:
            with torch.no_grad():
                output = model.generate(**gen_kwargs)
            reply = sanitize_model_output(tokenizer.decode(output[0], skip_special_tokens=True))
            if reply:
                return reply, st.session_state.active_model
        except Exception:
            continue

    # Only fallback if generation fails completely
    return "I’m here with you 🤍 — could you share a little more?", st.session_state.active_model

# ---------------------------
# Handle User Input
# ---------------------------
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)
    try:
        reply, active_model = generate_reply(user_input)
    except Exception as e:
        st.error(f"Error generating reply: {e}")
        reply = "I’m having trouble understanding — could you say it slightly differently?"
        active_model = st.session_state.active_model

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(f"{reply}\n\n---\n⚡ Model: **{active_model}**")
    st.sidebar.markdown(f"### Active Model: **{active_model}**")

    if torch.cuda.is_available():
        st.sidebar.markdown(
            f"**GPU:** {torch.cuda.memory_allocated()/(1024**2):.2f}MB / "
            f"{torch.cuda.memory_reserved()/(1024**2):.2f}MB"
        )
    else:
        st.sidebar.markdown("**Running on CPU**")

    proc = psutil.Process(os.getpid())
    st.sidebar.markdown(
        f"**CPU Usage:** {proc.cpu_percent(interval=0.1):.2f}%\n"
        f"**Memory:** {proc.memory_info().rss/(1024**2):.2f}MB"
    )
