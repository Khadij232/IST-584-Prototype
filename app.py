import streamlit as st
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import easyocr
import numpy as np
import re
import datetime

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Privacy Policy Analyzer", layout="wide", page_icon="🔐")

# ---------------------------
# MODELS — cached so they only load ONCE per session (Bug Fix #1)
# ---------------------------
model_path = "./model_folder"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

tokenizer, model = load_model()
reader = load_ocr()

# ---------------------------
# CONSTANTS
# ---------------------------
labels = [
    "DATA_COLLECTION",
    "THIRD_PARTY_SHARING",
    "LOCATION_DATA",
    "ADVERTISING_DATA",
    "OPT_OUT"
]

THRESHOLD = 0.3
HIGH_RISK_THRESHOLD = 0.6

# Weights for overall privacy score (higher = more impactful label)
label_weights = {
    "DATA_COLLECTION":    0.25,
    "THIRD_PARTY_SHARING": 0.25,
    "LOCATION_DATA":      0.20,
    "ADVERTISING_DATA":   0.20,
    "OPT_OUT":            0.10,
}

label_display_names = {
    "DATA_COLLECTION":    "Data Collection",
    "THIRD_PARTY_SHARING": "Third Party Sharing",
    "LOCATION_DATA":      "Location Data",
    "ADVERTISING_DATA":   "Advertising Data",
    "OPT_OUT":            "Opt Out"
}

label_descriptions = {
    "DATA_COLLECTION":    "Collects personal data such as name, email, device info, and usage behavior.",
    "THIRD_PARTY_SHARING": "Shares user data with external companies like advertisers or analytics providers.",
    "LOCATION_DATA":      "Tracks physical location using GPS or network-based methods.",
    "ADVERTISING_DATA":   "Uses user behavior to deliver targeted or personalized advertisements.",
    "OPT_OUT":            "Privacy settings are enabled by default, requiring users to manually opt out of data sharing."
}

screenshot_recommendations = {
    "DATA_COLLECTION":    "🔴 Turn OFF data collection to reduce tracking of your personal information.",
    "THIRD_PARTY_SHARING": "🔴 Disable third-party sharing to keep your data from being sold or shared.",
    "LOCATION_DATA":      "🔴 Turn OFF location access unless the app absolutely requires it.",
    "ADVERTISING_DATA":   "🔴 Disable ad personalization to reduce behavioral tracking.",
    "OPT_OUT":            "🔴 Manually opt out — privacy-invasive settings are enabled by default."
}

explain_keywords = {
    "DATA_COLLECTION":    ["collect", "data", "information", "store", "gather", "retain"],
    "THIRD_PARTY_SHARING": ["share", "third party", "third-party", "partners", "vendors", "disclose"],
    "LOCATION_DATA":      ["location", "gps", "track", "geolocation", "whereabouts"],
    "ADVERTISING_DATA":   ["ads", "advertising", "tracking", "personalized", "targeted", "behavioral"],
    "OPT_OUT":            ["default", "automatic", "opt-out", "opt out", "enabled", "unless you"]
}

# Screenshot keyword patterns: checks for label keyword near an "on" indicator (Bug Fix #6)
screenshot_patterns = {
    "DATA_COLLECTION":    (["data collection", "collect info", "usage data"], ["on", "enabled", "active"]),
    "THIRD_PARTY_SHARING": (["third-party", "third party", "share data", "data sharing"], ["on", "enabled", "active"]),
    "LOCATION_DATA":      (["location", "track location", "gps"], ["on", "enabled", "active", "always", "while using"]),
    "ADVERTISING_DATA":   (["ads", "personalized ads", "ad tracking", "advertising"], ["on", "enabled", "active"]),
    "OPT_OUT":            (["opt out", "opt-out", "data sharing", "marketing"], ["on", "enabled", "active"])  # Bug Fix #4
}

# ---------------------------
# CHUNK + CLASSIFY (Bug Fix #2 & Improvement #5)
# ---------------------------
def chunk_text(text, max_tokens=400, overlap=50):
    """Split text into overlapping token-safe chunks."""
    words = text.split()
    chunks = []
    step = max_tokens - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + max_tokens])
        chunks.append(chunk)
        if i + max_tokens >= len(words):
            break
    return chunks

def classify_chunk(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=256, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0].detach().numpy()
    return probs

def classify_text(text):
    """Classify full text by chunking and taking max prob per label."""
    chunks = chunk_text(text)
    all_probs = np.array([classify_chunk(chunk) for chunk in chunks])
    # Take the max probability across all chunks for each label
    max_probs = all_probs.max(axis=0)

    return {
        label: {
            "prob": float(prob),
            "risk": get_risk_level(prob),
            "explanation": explain_label(text, label)
        }
        for label, prob in zip(labels, max_probs)
    }

def get_risk_level(prob):
    if prob > HIGH_RISK_THRESHOLD:
        return "🔴 High Risk"
    elif prob > THRESHOLD:
        return "🟠 Medium Risk"
    else:
        return "🟢 Low Risk"

# ---------------------------
# EXPLANATION ENGINE
# ---------------------------
def explain_label(text, label):
    text_lower = text.lower()
    hits = [kw for kw in explain_keywords[label] if kw in text_lower]
    if hits:
        return f"Keywords detected: {', '.join(hits)}"
    return "No strong keyword evidence found; model inference used."

# ---------------------------
def generate_policy_summary(policy_text, results, score):
    """
    Summarize the policy using three targeted sentence searches:
    - What data is collected (sentences that START with "we collect / we automatically collect")
    - Who data is shared with (sentences that START with "we share / we disclose")
    - How data is used for advertising (sentences containing advertising-specific language)
    Scores candidates with BERT to pick the most substantive one per section.
    """

    noise_phrases = [
        "if you choose", "you can", "you may be able", "please be aware",
        "connecticut", "california resident", "click here", "contact us",
        "last updated", "terms of service", "instructions for deactivation",
        "we recommend that you review", "we will not retaliate",
        "we do not accept", "we will respond", "we will verify",
        "pre-uploading", "clipboard", "web beacons", "flash cookies"
    ]

    def is_noise(sent):
        return any(p in sent.lower() for p in noise_phrases)

    # Sentences must START with these patterns to qualify for each section
    # (uses regex to enforce sentence-start matching)
    section_patterns = {
        "collection": [
            r"^we (automatically )?collect",
            r"^we gather",
            r"^we may collect (biometric|precise|certain information from you)",
            r"^we automatically assign",
        ],
        "sharing": [
            r"^we share",
            r"^we may share",
            r"^we disclose",
            r"^advertisers.{0,30}partners",
        ],
        "advertising": [
            r"^advertising partners",
            r"^we use your activity",
            r"^we use (your|cookies|behavioral)",
            r"^ad (networks?|partners?)",
        ],
    }

    sentences = split_sentences(policy_text)

    def needs_continuation(sent):
        return sent.strip().endswith(":") or len(sent.strip().split()) < 12

    best = {"collection": None, "sharing": None, "advertising": None}
    best_scores = {"collection": 0.0, "sharing": 0.0, "advertising": 0.0}

    seen = set()
    for i, sent in enumerate(sentences):
        chunk = sent + " " + sentences[i + 1] if needs_continuation(sent) and i + 1 < len(sentences) else sent
        if chunk in seen or is_noise(chunk):
            continue
        seen.add(chunk)

        chunk_lower = chunk.strip().lower()
        probs = classify_chunk(chunk)
        bert_score = float(probs.max())

        if bert_score < THRESHOLD:
            continue

        for section, patterns in section_patterns.items():
            if any(re.match(p, chunk_lower) for p in patterns):
                if bert_score > best_scores[section]:
                    best_scores[section] = bert_score
                    text = chunk.strip()
                    if len(text) > 280:
                        text = text[:280].rsplit(" ", 1)[0] + "..."
                    best[section] = text

    # Opening verdict
    if score >= 60:
        opener = "This privacy policy raises significant privacy concerns."
    elif score >= 35:
        opener = "This privacy policy has moderate privacy implications worth being aware of."
    else:
        opener = "This privacy policy appears relatively low risk."

    p1_parts = [s for s in [best["collection"], best["sharing"]] if s]
    para1 = opener + (" " + " ".join(p1_parts) if p1_parts else "")
    para2 = best["advertising"] or ""

    if not p1_parts and not para2:
        return opener + " No specific substantive clauses could be extracted from this policy."

    return para1 + ("\n\n" + para2 if para2 else "")

# ---------------------------
# OVERALL PRIVACY SCORE (New Feature #8)
# ---------------------------
def compute_privacy_score(results):
    """Weighted average of label probs, scaled to 0–100 risk score."""
    score = sum(results[label]["prob"] * label_weights[label] for label in labels)
    return round(score * 100)

def score_color(score):
    if score >= 60:
        return "#ef4444"   # red
    elif score >= 35:
        return "#f97316"   # orange
    else:
        return "#22c55e"   # green

def score_label(score):
    if score >= 60:
        return "High Privacy Risk"
    elif score >= 35:
        return "Moderate Privacy Risk"
    else:
        return "Low Privacy Risk"

# ---------------------------
# OCR TEXT EXTRACTION
# ---------------------------
def extract_text(image):
    img = np.array(image)
    result = reader.readtext(img, detail=0)
    return " ".join(result).lower()

# ---------------------------
# SCREENSHOT ANALYSIS (Bug Fix #6: proximity-aware toggle detection)
# ---------------------------
def detect_toggle_state(ocr_text, label_keywords, on_indicators):
    """
    Look for a label keyword appearing near an ON indicator within 60 chars.
    Returns 'ON', 'OFF', or 'UNKNOWN'.
    """
    for kw in label_keywords:
        match = re.search(re.escape(kw), ocr_text)
        if match:
            start = max(0, match.start() - 60)
            end = min(len(ocr_text), match.end() + 60)
            surrounding = ocr_text[start:end]
            for indicator in on_indicators:
                if indicator in surrounding:
                    return "ON"
            # Found the keyword but no ON indicator nearby → likely OFF
            return "OFF"
    return "UNKNOWN"

def analyze_screenshot(image_file):
    image_file.seek(0)                          # Bug Fix #3: reset buffer
    image = Image.open(image_file).convert("RGB")
    ocr_text = extract_text(image)

    results = {}
    for label, (label_kws, on_indicators) in screenshot_patterns.items():
        results[label] = detect_toggle_state(ocr_text, label_kws, on_indicators)

    return results, ocr_text

# ---------------------------
# SENTENCE-LEVEL HIGHLIGHT (New Feature #9)
# ---------------------------
def split_sentences(text):
    """
    Robust sentence splitter for legal/privacy policy text.
    Splits on sentence-ending punctuation AND newlines, then filters
    out fragments that are too short to be meaningful.
    """
    # Normalize line breaks
    text = re.sub(r'\n+', ' \n ', text)
    # Split on: end-of-sentence punctuation, newlines, or semicolons before a capital
    raw = re.split(r'(?<=[.!?])\s+|(?<=;)\s+(?=[A-Z])|(?<=\n)\s*', text)
    sentences = []
    for s in raw:
        s = s.strip()
        if len(s.split()) >= 8:   # skip short fragments
            sentences.append(s)
    return sentences


def get_risky_sentences(text, results):
    """
    Classify sentences individually, but if a sentence looks like an
    incomplete intro clause (ends with ":" or is under 12 words),
    append the next sentence for context before classifying.
    """
    from collections import defaultdict
    sentences = split_sentences(text)

    def needs_continuation(sent):
        """True if sentence is an intro clause that needs the next for context."""
        stripped = sent.strip()
        return stripped.endswith(":") or len(stripped.split()) < 12

    by_label = defaultdict(list)
    seen = defaultdict(set)

    for i, sent in enumerate(sentences):
        if needs_continuation(sent) and i + 1 < len(sentences):
            chunk = sent + " " + sentences[i + 1]
        else:
            chunk = sent

        probs = classify_chunk(chunk)
        for j, label in enumerate(labels):
            prob = float(probs[j])
            if prob > THRESHOLD and chunk not in seen[label]:
                by_label[label].append((chunk, prob, label))
                seen[label].add(chunk)

    results_out = []
    for label in labels:
        top = sorted(by_label[label], key=lambda x: x[1], reverse=True)
        results_out.extend(top)

    results_out.sort(key=lambda x: x[1], reverse=True)
    return results_out
# ---------------------------
# REPORT GENERATOR (New Feature #10)
# ---------------------------
def build_report(policy_results, screenshot_results, score, ocr_text=""):
    lines = []
    lines.append("=" * 60)
    lines.append("       PRIVACY POLICY ANALYSIS REPORT")
    lines.append(f"       Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("=" * 60)

    if policy_results:
        lines.append(f"\n📊 OVERALL PRIVACY RISK SCORE: {score}/100  ({score_label(score)})\n")
        lines.append("── POLICY CLASSIFICATION RESULTS ──")
        for label, data in policy_results.items():
            name = label_display_names[label]
            lines.append(f"  {name}: {data['risk']} (confidence: {data['prob']:.2f})")
            lines.append(f"    Evidence: {data['explanation']}")

    if screenshot_results:
        lines.append("\n── SCREENSHOT SETTING RECOMMENDATIONS ──")
        for label, state in screenshot_results.items():
            name = label_display_names[label]
            if state == "ON":
                lines.append(f"  ⚠️  {name}: ENABLED → {screenshot_recommendations[label]}")
            elif state == "OFF":
                lines.append(f"  ✅  {name}: OFF (no action needed)")
            else:
                lines.append(f"  ❓  {name}: Could not determine toggle state from screenshot.")

    lines.append("\n" + "=" * 60)
    lines.append("This report was generated by the Privacy Policy Analyzer.")
    lines.append("=" * 60)
    return "\n".join(lines)

# ---------------------------
# CROSS-REFERENCE POLICY + SCREENSHOT (Improvement #7)
# ---------------------------
def get_cross_reference_alerts(policy_results, screenshot_results):
    """Flag labels where policy is High Risk AND setting is ON."""
    alerts = []
    for label in labels:
        if label not in screenshot_results:
            continue
        policy_high = policy_results.get(label, {}).get("prob", 0) > HIGH_RISK_THRESHOLD
        setting_on = screenshot_results.get(label) == "ON"
        if policy_high and setting_on:
            alerts.append(label)
    return alerts

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("🔐 Privacy Policy Analyzer")
st.write("AI-powered privacy policy + device screenshot analysis tool")
st.divider()

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    policy_text = st.text_area("📄 Paste Privacy Policy Text", height=250,
                               placeholder="Paste the full privacy policy here...")

with col_right:
    uploaded_image = st.file_uploader("📷 Upload Settings Screenshot",
                                      type=["png", "jpg", "jpeg"])
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Screenshot", use_container_width=True)

st.divider()
analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

# ---------------------------
# MAIN ANALYSIS
# ---------------------------
if analyze_btn:

    if not policy_text and not uploaded_image:
        st.warning("Please provide at least a policy text or screenshot.")
        st.stop()

    policy_results = None
    screenshot_results = None
    ocr_text = ""
    privacy_score = None

    # ── POLICY ANALYSIS ──
    if policy_text:
        with st.spinner("Analyzing policy..."):
            policy_results = classify_text(policy_text)
            privacy_score = compute_privacy_score(policy_results)

        # Overall score banner
        color = score_color(privacy_score)
        st.markdown(
            f"""
            <div style="background:{color}22; border-left: 5px solid {color};
                        padding: 16px 20px; border-radius: 8px; margin-bottom: 20px;">
                <span style="font-size:2rem; font-weight:800; color:{color};">
                    {privacy_score}/100
                </span>
                <span style="font-size:1.1rem; color:{color}; margin-left:12px;">
                    {score_label(privacy_score)}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("## 📊 Detected Data Practices")

        for label, data in policy_results.items():
            name = label_display_names[label]
            col1, col2 = st.columns([4, 1])
            with col1:
                st.markdown(f"**{name}:** {data['risk']} &nbsp; `{data['prob']:.2f}`")
            with col2:
                with st.popover("ℹ️ Details"):
                    st.markdown(f"### {name}")
                    st.write(label_descriptions[label])
                    st.markdown("---")
                    st.markdown("**🧠 Model Explanation:**")
                    st.write(data["explanation"])

        # Sentence-level highlighting — grouped by label
        with st.expander("🔍 View Riskiest Policy Sentences"):
            with st.spinner("Scanning individual sentences..."):
                risky = get_risky_sentences(policy_text, policy_results)
            if risky:
                from collections import defaultdict
                grouped = defaultdict(list)
                for sent, prob, lbl in risky:
                    grouped[lbl].append((sent, prob))

                label_colors = {
                    "DATA_COLLECTION":     ("#ef4444", "#fef2f2"),
                    "THIRD_PARTY_SHARING": ("#f97316", "#fff7ed"),
                    "LOCATION_DATA":       ("#eab308", "#fefce8"),
                    "ADVERTISING_DATA":    ("#8b5cf6", "#f5f3ff"),
                    "OPT_OUT":             ("#3b82f6", "#eff6ff"),
                }

                for lbl in labels:
                    if lbl not in grouped:
                        continue
                    border, bg = label_colors[lbl]
                    name = label_display_names[lbl]
                    count = len(grouped[lbl])
                    with st.expander(f"{name} — {count} sentence(s) flagged"):
                        for sent, prob in grouped[lbl]:
                            st.markdown(
                                f"<div style='border-left:4px solid {border}; padding:8px 12px; "
                                f"margin:4px 0 10px 0; border-radius:4px; background:{bg};'>"
                                f"<span style='font-size:0.75rem; color:{border}; font-weight:600;'>"
                                f"confidence {prob:.2f}</span><br>{sent}</div>",
                                unsafe_allow_html=True
                            )
            else:
                st.info("No risky sentences detected.")

    # ── SCREENSHOT ANALYSIS ──
    if uploaded_image:
        with st.spinner("Analyzing screenshot..."): 
            screenshot_results, ocr_text = analyze_screenshot(uploaded_image)

        st.markdown("## 🖼️ Screenshot Setting Recommendations")

        enabled_labels = [label for label, state in screenshot_results.items() if state == "ON"]

        if enabled_labels:
            for label in enabled_labels:
                name = label_display_names[label]
                st.error(f"⚠️ **{name}** is ENABLED — {screenshot_recommendations[label]}")
        else:
            st.success("No clearly enabled privacy risks detected in your current settings.")


    # ── CROSS-REFERENCE ALERTS ──
    if policy_results and screenshot_results:
        alerts = get_cross_reference_alerts(policy_results, screenshot_results)
        if alerts:
            st.markdown("## 🚨 Priority Alerts")
            st.markdown(
                "These settings are **both flagged as High Risk in the policy AND currently enabled** "
                "on your device — these are your most urgent privacy concerns:"
            )
            for label in alerts:
                st.error(
                    f"🚨 **{label_display_names[label]}**: The policy is high-risk for this category "
                    f"AND the setting is currently ON. Disable immediately."
                )
