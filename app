import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pickle
import numpy as np
import os
import gdown
import json

st.set_page_config(page_title="MoodVibe", page_icon="😊", layout="wide")

model_path = "./stress_model"
drive_folder_id = "1KpwcYgQcNxwns6sMyl5bBzIGtZtE0kH5"

# ------------------------------
# Download model if missing
# ------------------------------



def download_model_from_drive():
    required_files = [
        "label_encoder.pkl", "tokenizer_config.json", "model.safetensors", "config.json"
    ]
    if not os.path.exists(model_path) or not all(
        os.path.exists(os.path.join(model_path, f)) for f in required_files
    ):
        os.makedirs(model_path, exist_ok=True)
        st.info("📥 Downloading model files from Google Drive...")
        try:
            gdown.download_folder(
                id=drive_folder_id, output=model_path, quiet=False, use_cookies=False
            )
            st.success("✅ Model downloaded successfully.")
        except Exception as e:
            st.error(f"❌ Failed to download model: {str(e)}")
            raise

download_model_from_drive()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Cached resources
# ------------------------------
@st.cache_resource
def load_label_encoder():
    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_tokenizer():
    return BertTokenizer.from_pretrained(model_path, local_files_only=True)

@st.cache_resource
def load_model():
    # Load label encoder inside the function (not as param)
    with open(f"{model_path}/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)

    model = BertForSequenceClassification.from_pretrained(
        model_path,
        local_files_only=True,
        num_labels=len(le.classes_),
        id2label={i: label for i, label in enumerate(le.classes_)},
        label2id={label: i for i, label in enumerate(le.classes_)},
        ignore_mismatched_sizes=True
    )
    model.to(device)
    model.eval()
    return model, le

# Load everything once
tokenizer = load_tokenizer()
model, label_encoder = load_model()


# ------------------------------
# Debug Info: Model + Encoder
# ------------------------------
try:
    with open(f"{model_path}/config.json") as f:
        cfg = json.load(f)
    st.sidebar.write("🔧 Model num_labels:", cfg.get("num_labels"))
except Exception:
    st.sidebar.write("⚠️ Could not read config.json")

st.sidebar.write("🔧 Label encoder classes:", list(label_encoder.classes_))

# ------------------------------
# Prediction function
# ------------------------------
def predict_sentiment(post, max_length=128):
    inputs = tokenizer(
        post,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # Debug raw probabilities
    st.write("🔍 Raw probabilities:", {cls: float(p) for cls, p in zip(label_encoder.classes_, probs)})

    predicted_class = np.argmax(probs)
    confidence = probs[predicted_class]
    return label_encoder.inverse_transform([predicted_class])[0], confidence

# ------------------------------
# Sidebar
# ------------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    max_length = st.slider("Max Tokens for Analysis", 32, 512, 128, 8)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.1)
    debug_mode = st.checkbox("Show Confidence Scores", value=True)
    char_limit_enabled = st.checkbox("Enable Character Limit", value=True)
    char_limit = st.number_input(
        "Character Limit", min_value=100, max_value=10000, value=280, step=10
    ) if char_limit_enabled else float("inf")
    st.markdown("---")
    st.markdown("😊 MoodVibe v1.0")

# ------------------------------
# Main app
# ------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.title("😊 MoodVibe")
st.markdown("Discover the emotions behind social media posts with AI-driven sentiment analysis.")

col1, col2 = st.columns([3, 2])
with col1:
    user_input = st.text_area(
        "Enter Post", placeholder="Type a post (e.g., 'Feeling overwhelmed...')", height=150
    )

    if char_limit_enabled:
        token_count = len(tokenizer.tokenize(user_input)) if user_input.strip() else 0
        st.markdown(
            f"<span id='char-count'>{len(user_input)}/{char_limit}</span> | Tokens: {token_count}",
            unsafe_allow_html=True,
        )

    if st.button("Analyze"):
        if not user_input.strip():
            st.error("Please enter a post to analyze.")
        elif char_limit_enabled and len(user_input) > char_limit:
            st.error(f"Post exceeds {char_limit} characters. Please shorten it or disable the limit.")
        else:
            with st.spinner("Analyzing..."):
                sentiment, confidence = predict_sentiment(user_input, max_length)
                if confidence >= confidence_threshold:
                    st.session_state.history.append(
                        {"post": user_input, "sentiment": sentiment, "confidence": confidence}
                    )
                    st.markdown(
                        f"""
                        <div class='card'>
                            <h3>Sentiment Prediction</h3>
                            <p><b>Post:</b> {user_input}</p>
                            <p><b>Sentiment:</b> {sentiment}</p>
                            {f"<p><b>Confidence:</b> {confidence:.2%}</p>" if debug_mode else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.warning(f"Prediction confidence ({confidence:.2%}) is below threshold ({confidence_threshold:.2%}).")

with col2:
    with st.expander("Recent Predictions", expanded=True):
        if st.session_state.history:
            sentiment_filter = st.selectbox("Filter by Sentiment", ["All"] + list(label_encoder.classes_))
            for item in st.session_state.history[-5:][::-1]:
                if sentiment_filter == "All" or item["sentiment"] == sentiment_filter:
                    st.markdown(
                        f"""
                        <div class='history-item'>
                            <p><b>Post:</b> {item['post'][:50]}{'...' if len(item['post']) > 50 else ''}</p>
                            <p><b>Sentiment:</b> {item['sentiment']}</p>
                            {f"<p><b>Confidence:</b> {item['confidence']:.2%}</p>" if debug_mode else ""}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown("No predictions yet.")
