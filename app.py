#!/usr/bin/env python3
# ════════════════════════════════════════════════════════════
# 🚀 APP.PY - WORLD'S BEST SENTIMENT ANALYZER
# Run: streamlit run app.py
# ════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
from collections import Counter
import time
import os
import random
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════════════════════════════════
# ⚙️  PAGE CONFIG
# ════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="🐦 Sentiment140 Analyzer",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════════════════════
# 🎨 CUSTOM CSS
# ════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Inter:wght@300;400;600;700&display=swap');

#MainMenu {visibility:hidden;}
footer    {visibility:hidden;}
header    {visibility:hidden;}
.stDeployButton {display:none;}

.stApp {
    background: linear-gradient(135deg,#0A0E1A 0%,#0D1B2A 50%,#0A0E1A 100%);
    background-attachment: fixed;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0D1117 0%,#161B27 100%);
    border-right: 1px solid #7B68EE44;
}
[data-testid="stSidebar"] * { color:#E0E0E0 !important; }

.main-title {
    font-family:'Orbitron',monospace;
    font-size:2.8rem;
    font-weight:900;
    text-align:center;
    background:linear-gradient(90deg,#00D4AA,#7B68EE,#FF6B6B,#FFD700,#00D4AA);
    background-size:300% 300%;
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
    background-clip:text;
    animation:gradientFlow 4s ease infinite;
    letter-spacing:3px;
    margin-bottom:0.2rem;
}
@keyframes gradientFlow {
    0%   {background-position:0% 50%;}
    50%  {background-position:100% 50%;}
    100% {background-position:0% 50%;}
}

.subtitle {
    font-family:'Rajdhani',sans-serif;
    font-size:1.1rem;
    text-align:center;
    color:#00FFFF;
    letter-spacing:4px;
    opacity:0.8;
    margin-bottom:2rem;
}

.metric-card {
    background:linear-gradient(135deg,#1A1F35,#1E2540);
    border:1px solid;
    border-radius:16px;
    padding:1.5rem;
    text-align:center;
    transition:transform 0.3s ease,box-shadow 0.3s ease;
}
.metric-card:hover {
    transform:translateY(-5px);
    box-shadow:0 20px 40px rgba(0,0,0,0.4);
}
.metric-value {
    font-family:'Orbitron',monospace;
    font-size:2.2rem;
    font-weight:900;
    margin:0.3rem 0;
}
.metric-label {
    font-family:'Rajdhani',sans-serif;
    font-size:0.85rem;
    letter-spacing:2px;
    opacity:0.7;
    text-transform:uppercase;
}
.metric-icon {font-size:2rem;margin-bottom:0.5rem;}

.section-header {
    font-family:'Orbitron',monospace;
    font-size:1.3rem;
    font-weight:700;
    color:#FFD700;
    border-left:4px solid #7B68EE;
    padding-left:1rem;
    margin:2rem 0 1rem 0;
    letter-spacing:2px;
}

.tweet-card {
    background:linear-gradient(135deg,#1A1F35,#1E2540);
    border-radius:12px;
    padding:1rem 1.2rem;
    margin:0.5rem 0;
    border-left:4px solid;
    transition:all 0.3s ease;
}
.tweet-card:hover {
    transform:translateX(5px);
    box-shadow:0 8px 25px rgba(0,0,0,0.3);
}
.tweet-pos {border-color:#00D4AA;}
.tweet-neg {border-color:#FF6B6B;}

/* ── CHAT STYLES ─────────────────────────────────────────── */
.live-badge {
    display:inline-block;
    background:rgba(255,0,0,0.15);
    border:1px solid #FF4444;
    color:#FF4444;
    border-radius:20px;
    padding:0.15rem 0.8rem;
    font-size:0.75rem;
    font-family:'Rajdhani';
    letter-spacing:2px;
    animation:blink 1.5s infinite;
}
@keyframes blink {
    0%,100% {opacity:1;}
    50%      {opacity:0.4;}
}

.user-bubble-wrap {
    display:flex;
    justify-content:flex-end;
    align-items:flex-end;
    margin:0.8rem 0;
    gap:0.5rem;
    animation:slideRight 0.3s ease;
}
@keyframes slideRight {
    from {opacity:0;transform:translateX(20px);}
    to   {opacity:1;transform:translateX(0);}
}

.bot-bubble-wrap {
    display:flex;
    justify-content:flex-start;
    align-items:flex-end;
    margin:0.8rem 0;
    gap:0.5rem;
    animation:slideLeft 0.3s ease;
}
@keyframes slideLeft {
    from {opacity:0;transform:translateX(-20px);}
    to   {opacity:1;transform:translateX(0);}
}

.user-msg {
    background:linear-gradient(135deg,#7B68EE,#5B48CE);
    border-radius:20px 20px 4px 20px;
    padding:0.8rem 1.3rem;
    max-width:65%;
    color:white;
    font-family:'Inter',sans-serif;
    font-size:0.95rem;
    line-height:1.5;
    box-shadow:0 4px 20px rgba(123,104,238,0.35);
    word-break:break-word;
}

.bot-msg-pos {
    background:linear-gradient(135deg,rgba(0,212,170,0.12),rgba(0,212,170,0.04));
    border:1.5px solid #00D4AA;
    border-radius:20px 20px 20px 4px;
    padding:1rem 1.3rem;
    max-width:72%;
    box-shadow:0 4px 20px rgba(0,212,170,0.15);
    word-break:break-word;
}
.bot-msg-neg {
    background:linear-gradient(135deg,rgba(255,107,107,0.12),rgba(255,107,107,0.04));
    border:1.5px solid #FF6B6B;
    border-radius:20px 20px 20px 4px;
    padding:1rem 1.3rem;
    max-width:72%;
    box-shadow:0 4px 20px rgba(255,107,107,0.15);
    word-break:break-word;
}

.score-pill {
    display:inline-block;
    padding:0.15rem 0.7rem;
    border-radius:20px;
    font-size:0.75rem;
    font-family:'Rajdhani';
    font-weight:600;
    letter-spacing:1px;
    margin:0.15rem;
}

.chat-input-container {
    background:linear-gradient(135deg,#1A1F35,#1E2540);
    border:1px solid #7B68EE55;
    border-radius:16px;
    padding:1rem;
    margin-top:1rem;
}

.stButton>button {
    background:linear-gradient(135deg,#7B68EE,#00D4AA) !important;
    color:white !important;
    border:none !important;
    border-radius:12px !important;
    font-family:'Rajdhani',sans-serif !important;
    font-size:1rem !important;
    font-weight:700 !important;
    letter-spacing:2px !important;
    transition:all 0.3s ease !important;
    text-transform:uppercase !important;
}
.stButton>button:hover {
    transform:translateY(-2px) !important;
    box-shadow:0 10px 30px rgba(123,104,238,0.4) !important;
}

.stTextArea textarea,.stTextInput>div>div>input {
    background:#1A1F35 !important;
    border:1.5px solid #7B68EE55 !important;
    border-radius:14px !important;
    color:#E0E0E0 !important;
    font-family:'Inter',sans-serif !important;
    font-size:1rem !important;
    padding:0.8rem 1rem !important;
}
.stTextInput>div>div>input:focus {
    border-color:#7B68EE !important;
    box-shadow:0 0 25px rgba(123,104,238,0.25) !important;
}
.stTextInput>div>div>input::placeholder {
    color:#555 !important;
}

.stTabs [data-baseweb="tab-list"] {
    background:#1A1F35 !important;
    border-radius:12px !important;
    padding:0.3rem !important;
    gap:0.3rem !important;
}
.stTabs [data-baseweb="tab"] {
    font-family:'Rajdhani',sans-serif !important;
    font-weight:600 !important;
    letter-spacing:1px !important;
    color:#9090A0 !important;
    border-radius:10px !important;
    padding:0.5rem 1.5rem !important;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#7B68EE,#00D4AA) !important;
    color:white !important;
}

.stProgress>div>div {
    background:linear-gradient(90deg,#7B68EE,#00D4AA) !important;
    border-radius:10px !important;
}

hr {
    border:none !important;
    height:1px !important;
    background:linear-gradient(90deg,transparent,#7B68EE88,transparent) !important;
    margin:2rem 0 !important;
}

::-webkit-scrollbar       {width:6px;height:6px;}
::-webkit-scrollbar-track {background:#0A0E1A;}
::-webkit-scrollbar-thumb {
    background:linear-gradient(#7B68EE,#00D4AA);
    border-radius:3px;
}
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 🎨 COLORS & PLOTLY LAYOUT
# ════════════════════════════════════════════════════════════
C = {
    'bg'     : '#0A0E1A',
    'surface': '#1A1F35',
    'pos'    : '#00D4AA',
    'neg'    : '#FF6B6B',
    'gold'   : '#FFD700',
    'accent' : '#7B68EE',
    'cyan'   : '#00FFFF',
}

PL = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,31,53,0.8)',
    font=dict(color='#E0E0E0', family='Rajdhani, sans-serif'),
    margin=dict(l=20, r=20, t=40, b=20),
    xaxis=dict(gridcolor='#2A2F45', zerolinecolor='#2A2F45'),
    yaxis=dict(gridcolor='#2A2F45', zerolinecolor='#2A2F45'),
)

def rgba(hex_col, alpha=0.3):
    h = hex_col.lstrip('#')
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return f'rgba({r},{g},{b},{alpha})'

# ════════════════════════════════════════════════════════════
# 📦 CACHED — train ONCE, reuse everywhere
# ════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_and_train(filepath, sample_size):
    """
    Returns (trainer, df_clean).
    trainer.tfidf  ← the ONLY vectoriser to use for prediction
    trainer.best_model ← the ONLY model to use
    """
    from load_data    import load_sentiment140
    from preprocess   import Sentiment140Preprocessor
    from train_models import Sentiment140Trainer

    df      = load_sentiment140(filepath, sample_size=sample_size)
    prep    = Sentiment140Preprocessor()
    df_c    = prep.process_dataframe(df.copy(), text_col='text')
    trainer = Sentiment140Trainer(df_c)
    trainer.prepare().train()
    return trainer, df_c

# ════════════════════════════════════════════════════════════
# 🔮 PREDICT — always use trainer.tfidf (fixes the 6862 error)
# ════════════════════════════════════════════════════════════
def predict_sentiment(text, trainer):
    """
    Uses the SAME tfidf that was fitted during training.
    Fixes: 'X has 6862 features, but model expects 15000'
    """
    from preprocess import Sentiment140Preprocessor
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    from textblob import TextBlob

    prep  = Sentiment140Preprocessor()
    vader = SentimentIntensityAnalyzer()

    cleaned = prep.clean(text)                        # clean text
    feat    = trainer.tfidf.transform([cleaned])      # use SAME tfidf ✅
    pred    = trainer.best_model.predict(feat)        # predict
    label   = trainer.le.inverse_transform(pred)[0]  # decode label

    vs = vader.polarity_scores(text)['compound']
    tb = TextBlob(text).sentiment.polarity

    return label, vs, tb

def get_insight(label, vs, tb):
    if label == 'Positive':
        if vs > 0.6 : return "🔥 Extremely positive! Overflowing with joy!"
        if vs > 0.3 : return "✨ Strong positive energy. Great vibes!"
        if vs > 0.05: return "🌱 Mildly positive. Pleasant and hopeful."
        return "💫 Positive overall despite some neutral phrasing."
    else:
        if vs < -0.6 : return "⚠️ Very strong negative emotion. Hope things improve! 💙"
        if vs < -0.3 : return "😟 Clear negative sentiment. Frustration detected."
        if vs < -0.05: return "🌧️ Mildly negative. A hint of disappointment."
        return "🤔 Slightly negative despite some neutral words."

# ════════════════════════════════════════════════════════════
# 🎨 UI COMPONENTS
# ════════════════════════════════════════════════════════════
def metric_card(icon, label, value, color):
    st.markdown(f"""
    <div class="metric-card" style="border-color:{color}44">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="color:{color}">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def section_header(icon, title):
    st.markdown(
        f'<div class="section-header">{icon} {title}</div>',
        unsafe_allow_html=True)

def tweet_card(text, sentiment, score=None):
    cls   = 'tweet-pos' if sentiment=='Positive' else 'tweet-neg'
    emoji = '😊' if sentiment=='Positive' else '😢'
    color = C['pos'] if sentiment=='Positive' else C['neg']
    sc    = f'VADER: {score:.3f}' if score is not None else ''
    st.markdown(f"""
    <div class="tweet-card {cls}">
        <div style="color:#E0E0E0;font-size:0.9rem;line-height:1.5;margin-bottom:0.4rem">
            {emoji} {text[:120]}{'...' if len(text)>120 else ''}
        </div>
        <div style="font-size:0.75rem;opacity:0.6;font-family:Rajdhani;letter-spacing:1px">
            <span style="color:{color};font-weight:600">{sentiment}</span>
            {'&nbsp;·&nbsp;'+sc if sc else ''}
        </div>
    </div>""", unsafe_allow_html=True)

def make_wc(text, bg, colors):
    if not text.strip(): return None
    wc = WordCloud(
        width=1000, height=500, background_color=bg, max_words=120,
        color_func=lambda *a,**k: random.choice(colors),
        prefer_horizontal=0.7, min_font_size=8, max_font_size=120,
        stopwords=STOPWORDS, collocations=False,
    ).generate(text)
    fig, ax = plt.subplots(figsize=(10,5))
    fig.patch.set_facecolor(bg)
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# ════════════════════════════════════════════════════════════
# 💬 RENDER CHAT HISTORY
# ════════════════════════════════════════════════════════════
def render_chat(trainer):
    history = st.session_state.chat_history

    if not history:
        st.markdown("""
        <div style="text-align:center;padding:3rem 2rem;
                    background:linear-gradient(135deg,#1A1F35,#1E2540);
                    border-radius:20px;border:1px dashed #7B68EE44;margin-bottom:1rem">
            <div style="font-size:3rem;margin-bottom:1rem">🤖</div>
            <div style="font-family:Orbitron;color:#7B68EE;font-size:1rem;letter-spacing:2px">
                AI SENTIMENT ANALYZER
            </div>
            <div style="color:#9090A0;font-family:Rajdhani;margin-top:0.5rem">
                Type any message below to analyze sentiment instantly!
            </div>
            <div style="margin-top:1rem;color:#555;font-size:0.85rem">
                Try: "I love this!" &nbsp;·&nbsp; "This is terrible..." &nbsp;·&nbsp; "Best day ever!"
            </div>
        </div>""", unsafe_allow_html=True)
        return

    for idx, msg in enumerate(history):
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="user-bubble-wrap">
                <div class="user-msg">
                    <div style="font-size:0.7rem;color:rgba(255,255,255,0.45);
                                margin-bottom:0.3rem;font-family:Rajdhani;letter-spacing:1px">YOU</div>
                    {msg['text']}
                </div>
                <div style="font-size:1.8rem;flex-shrink:0">👤</div>
            </div>""", unsafe_allow_html=True)

        else:
            color = C['pos'] if msg['sentiment']=='Positive' else C['neg']
            emoji = '😊' if msg['sentiment']=='Positive' else '😢'
            cls   = 'bot-msg-pos' if msg['sentiment']=='Positive' else 'bot-msg-neg'
            stars = '⭐⭐⭐⭐⭐' if msg['sentiment']=='Positive' else '💔💔💔💔💔'

            vs       = msg['vader']
            bar_len  = int((vs + 1) / 2 * 12)
            bar_str  = '█'*bar_len + '░'*(12-bar_len)

            st.markdown(f"""
            <div class="bot-bubble-wrap">
                <div style="font-size:1.8rem;flex-shrink:0">🤖</div>
                <div class="{cls}">
                    <div style="font-size:0.7rem;color:rgba(255,255,255,0.45);
                                margin-bottom:0.6rem;font-family:Rajdhani;letter-spacing:1px">
                        AI ANALYSIS
                    </div>

                    <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem">
                        <span style="font-size:2rem">{emoji}</span>
                        <div>
                            <div style="font-family:Orbitron;font-size:1.3rem;font-weight:900;
                                        color:{color};letter-spacing:2px">
                                {msg['sentiment'].upper()}
                            </div>
                            <div style="color:#9090A0;font-size:0.8rem;margin-top:0.1rem">
                                {stars}
                            </div>
                        </div>
                    </div>

                    <div style="background:rgba(0,0,0,0.25);border-radius:8px;
                                padding:0.5rem 0.8rem;margin:0.5rem 0;font-family:monospace;
                                font-size:0.78rem">
                        <span style="color:#9090A0">VADER </span>
                        <span style="color:{color}">[{bar_str}] {vs:+.3f}</span>
                    </div>

                    <div style="margin:0.5rem 0;display:flex;flex-wrap:wrap;gap:0.2rem">
                        <span class="score-pill" style="background:rgba(0,212,170,0.12);
                              border:1px solid #00D4AA55;color:#00D4AA">
                            🧠 ML: {msg['sentiment']}
                        </span>
                        <span class="score-pill" style="background:rgba(123,104,238,0.12);
                              border:1px solid #7B68EE55;color:#7B68EE">
                            ⚡ VADER: {vs:+.3f}
                        </span>
                        <span class="score-pill" style="background:rgba(255,215,0,0.12);
                              border:1px solid #FFD70055;color:#FFD700">
                            📝 TextBlob: {msg['textblob']:+.3f}
                        </span>
                    </div>

                    <div style="color:#C0C0C0;font-size:0.82rem;font-family:Inter,sans-serif;
                                padding:0.5rem 0.8rem;background:rgba(255,255,255,0.03);
                                border-radius:8px;border-left:3px solid {color};
                                margin-top:0.5rem;line-height:1.5">
                        {msg['insight']}
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Reaction buttons
            r_cols = st.columns([1,1,1,1,1,5])
            reacts = ["👍","❤️","😮","😂","🔥"]
            for ri, (rc, react) in enumerate(zip(r_cols[:5], reacts)):
                with rc:
                    key   = f"{idx}_{react}"
                    count = st.session_state.reactions.get(key, 0)
                    label = f"{react}{count}" if count else react
                    if st.button(label, key=f"rb_{idx}_{ri}"):
                        st.session_state.reactions[key] = count + 1
                        st.rerun()

            st.markdown("<div style='margin-bottom:0.3rem'></div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# 🚀 MAIN
# ════════════════════════════════════════════════════════════
def main():

    # ── HEADER ────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem 0">
        <div class="main-title">🐦 SENTIMENT140 ANALYZER</div>
        <div class="subtitle">◆ NLP · MACHINE LEARNING · REAL-TIME PREDICTION ◆</div>
    </div>
    <div style="height:3px;background:linear-gradient(90deg,
        transparent,#7B68EE,#00D4AA,#FF6B6B,#FFD700,transparent);
        border-radius:3px;margin-bottom:2rem"></div>
    """, unsafe_allow_html=True)

    # ── SIDEBAR ────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0">
            <div style="font-family:Orbitron,monospace;font-size:1.2rem;
                        color:#FFD700;letter-spacing:2px">⚙️ CONTROLS</div>
        </div>""", unsafe_allow_html=True)
        st.markdown("---")

        st.markdown('<p style="color:#7B68EE;font-weight:600;letter-spacing:1px">📂 DATASET PATH</p>',
                    unsafe_allow_html=True)
        filepath = st.text_input("CSV", value='data/training.1600000.processed.noemoticon.csv',
                                 label_visibility='collapsed')

        st.markdown("---")
        st.markdown('<p style="color:#7B68EE;font-weight:600;letter-spacing:1px">📊 SAMPLE SIZE</p>',
                    unsafe_allow_html=True)
        sample_size = st.select_slider("Size",
            options=[5000,10000,20000,30000,50000,100000], value=50000,
            label_visibility='collapsed')
        st.caption(f"Using {sample_size:,} tweets")

        st.markdown("---")
        load_btn = st.button("🚀  LOAD & TRAIN", use_container_width=True)
        st.markdown("---")

        st.markdown("""
        <div style="background:#1A1F35;border-radius:12px;padding:1rem;
                    border:1px solid #7B68EE33;font-family:Rajdhani,sans-serif;
                    font-size:0.85rem;color:#9090A0">
            <b style="color:#FFD700">📌 Dataset</b><br><br>
            🗂️ Sentiment140<br>📝 1.6M tweets<br>🏷️ Binary labels<br><br>
            <b style="color:#FFD700">🤖 Models</b><br><br>
            🧠 Logistic Regression<br>📊 Naive Bayes<br>
            ⚡ Linear SVM<br>🌲 Random Forest
        </div>""", unsafe_allow_html=True)

    # ── INIT SESSION ──────────────────────────────────────────
    for key, val in [
        ('data_loaded',    False),
        ('chat_history',   []),
        ('reactions',      {}),
        ('total_analyzed', 0),
        ('pos_chat',       0),
        ('neg_chat',       0),
        ('pending_msg',    ''),
    ]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── LOAD & TRAIN ───────────────────────────────────────────
    if load_btn or not st.session_state.data_loaded:
        if not os.path.exists(filepath):
            st.error(f"❌ File not found: `{filepath}`")
            st.stop()

        ph = st.empty()
        ph.markdown("""
        <div style="text-align:center;padding:3rem">
            <div style="font-family:Orbitron,monospace;font-size:1.4rem;
                        background:linear-gradient(90deg,#7B68EE,#00D4AA);
                        -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                        margin-bottom:1rem">⚡ INITIALIZING AI ENGINE...</div>
            <div style="color:#9090A0;font-family:Rajdhani">
                Loading · Preprocessing · Training · Ready
            </div>
        </div>""", unsafe_allow_html=True)

        prog = st.progress(0)
        prog.progress(15)
        trainer, df_clean = load_and_train(filepath, sample_size)
        prog.progress(100)
        time.sleep(0.3)
        ph.empty(); prog.empty()

        st.session_state.data_loaded = True
        st.session_state.trainer     = trainer
        st.session_state.df          = df_clean
        st.success("✅ All systems ready!")

    if not st.session_state.data_loaded:
        st.markdown("""
        <div style="text-align:center;padding:5rem;color:#444">
            <div style="font-size:4rem">🚀</div>
            <div style="font-family:Orbitron;color:#7B68EE;font-size:1.2rem;margin-top:1rem">
                Click LOAD & TRAIN to begin
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    trainer = st.session_state.trainer
    df      = st.session_state.df
    results = trainer.results
    best    = trainer.best_name

    # ── KPI METRICS ────────────────────────────────────────────
    section_header("🎯","KEY PERFORMANCE INDICATORS")
    counts   = df['sentiment'].value_counts()
    pos_cnt  = counts.get('Positive',0)
    neg_cnt  = counts.get('Negative',0)
    best_acc = results[best]['accuracy']
    best_auc = results[best]['roc_auc']

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    with c1: metric_card("🐦","Total Tweets",f"{len(df):,}",C['accent'])
    with c2: metric_card("😊","Positive",f"{pos_cnt:,}",C['pos'])
    with c3: metric_card("😢","Negative",f"{neg_cnt:,}",C['neg'])
    with c4: metric_card("🎯","Best Accuracy",f"{best_acc:.2%}",C['gold'])
    with c5: metric_card("📈","ROC AUC",f"{best_auc:.4f}",C['cyan'])
    with c6: metric_card("🏆","Best Model","Log.Reg",C['accent'])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ───────────────────────────────────────────────────
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        "📊  Overview","🔬  Analytics","☁️  Word Clouds","🤖  Models","🔮  Predict & Chat"
    ])

    # ══════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        cl, cr = st.columns(2)

        with cl:
            section_header("📊","SENTIMENT DISTRIBUTION")
            fig = go.Figure(go.Pie(
                labels=counts.index.tolist(), values=counts.values.tolist(),
                hole=0.65,
                marker=dict(colors=[C['pos'],C['neg']], line=dict(color=C['bg'],width=4)),
                textinfo='label+percent',
                textfont=dict(size=14,color='white',family='Rajdhani'),
                hovertemplate="<b>%{label}</b><br>Count: %{value:,}<extra></extra>",
                pull=[0.05,0.05],
            ))
            fig.add_annotation(
                text=f'<b>{len(df):,}</b><br><span style="font-size:11px">Tweets</span>',
                x=0.5,y=0.5,showarrow=False,
                font=dict(size=20,color=C['gold'],family='Orbitron'))
            fig.update_layout(**PL, height=380,
                legend=dict(font=dict(color='white',size=13),bgcolor='rgba(0,0,0,0)',
                            orientation='h',y=-0.1,x=0.5,xanchor='center'))
            st.plotly_chart(fig, use_container_width=True)

        with cr:
            section_header("📏","TWEET LENGTH ANALYSIS")
            fig = go.Figure()
            for s, col, fill in [
                ('Positive',C['pos'],rgba(C['pos'],0.2)),
                ('Negative',C['neg'],rgba(C['neg'],0.2)),
            ]:
                fig.add_trace(go.Box(
                    y=df[df['sentiment']==s]['text_length'], name=s,
                    marker_color=col, line_color=col, fillcolor=fill,
                    boxmean='sd',
                    hovertemplate=f"<b>{s}</b><br>Length: %{{y}}<extra></extra>",
                ))
            fig.update_layout(**PL, height=380, yaxis_title='Characters',
                legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        cl2, cr2 = st.columns([3,2])

        with cl2:
            section_header("🕐","HOURLY TREND")
            if 'hour' in df.columns:
                h = df.groupby(['hour','sentiment']).size().reset_index(name='count')
                fig = px.area(h, x='hour', y='count', color='sentiment',
                    color_discrete_map={'Positive':C['pos'],'Negative':C['neg']},
                    template='none')
                fig.update_traces(line_width=2.5,opacity=0.7)
                fig.update_layout(**PL, height=320,
                    xaxis_title='Hour of Day', yaxis_title='Tweet Count',
                    legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig, use_container_width=True)

        with cr2:
            section_header("📝","WORD COUNT")
            fig = go.Figure()
            for s, col in [('Positive',C['pos']),('Negative',C['neg'])]:
                fig.add_trace(go.Histogram(
                    x=df[df['sentiment']==s]['word_count'],
                    name=s, marker_color=col, opacity=0.7, nbinsx=20))
            fig.update_layout(**PL, height=320, barmode='overlay',
                xaxis_title='Words', yaxis_title='Frequency',
                legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        section_header("🐦","SAMPLE TWEETS")
        cp, cn = st.columns(2)
        with cp:
            st.markdown('<p style="color:#00D4AA;font-weight:600;font-family:Rajdhani;'
                        'letter-spacing:2px">😊 POSITIVE</p>', unsafe_allow_html=True)
            for _, row in df[df['sentiment']=='Positive'].sample(5,random_state=42).iterrows():
                tweet_card(row['text'],'Positive',row['vader_compound'])
        with cn:
            st.markdown('<p style="color:#FF6B6B;font-weight:600;font-family:Rajdhani;'
                        'letter-spacing:2px">😢 NEGATIVE</p>', unsafe_allow_html=True)
            for _, row in df[df['sentiment']=='Negative'].sample(5,random_state=42).iterrows():
                tweet_card(row['text'],'Negative',row['vader_compound'])

    # ══════════════════════════════════════════════════════════
    # TAB 2 — ANALYTICS
    # ══════════════════════════════════════════════════════════
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            section_header("🎻","VADER DISTRIBUTION")
            fig = go.Figure()
            for s, col in [('Positive',C['pos']),('Negative',C['neg'])]:
                fig.add_trace(go.Violin(
                    y=df[df['sentiment']==s]['vader_compound'], name=s,
                    fillcolor=rgba(col,0.35), line_color=col,
                    meanline_visible=True, meanline_color=C['gold'],
                    box_visible=True, box_fillcolor=rgba(col,0.2),
                    hovertemplate=f"<b>{s}</b><br>Score: %{{y:.3f}}<extra></extra>",
                ))
            fig.update_layout(**PL, height=400, yaxis_title='VADER Compound',
                violingap=0.3, violinmode='overlay',
                legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            section_header("🔬","POLARITY vs SUBJECTIVITY")
            sdf = df.sample(min(2000,len(df)),random_state=42)
            fig = px.scatter(sdf, x='tb_polarity', y='tb_subjectivity', color='sentiment',
                color_discrete_map={'Positive':C['pos'],'Negative':C['neg']},
                opacity=0.5, template='none', hover_data={'vader_compound':':.3f'})
            fig.add_vline(x=0,line_dash='dash',line_color='white',opacity=0.3)
            fig.add_hline(y=0.5,line_dash='dash',line_color='white',opacity=0.3)
            fig.update_layout(**PL, height=400,
                xaxis_title='Polarity', yaxis_title='Subjectivity',
                legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        c3, c4 = st.columns(2)

        with c3:
            section_header("🔥","CORRELATION HEATMAP")
            nc = [c for c in ['text_length','word_count','n_hashtags','n_exclaim',
                'n_question','caps_ratio','tb_polarity','tb_subjectivity',
                'vader_compound','vader_pos','vader_neg'] if c in df.columns]
            corr = df[nc].corr()
            fig  = px.imshow(corr, color_continuous_scale='RdBu_r',
                zmin=-1,zmax=1,aspect='auto',text_auto='.2f',template='none')
            fig.update_traces(textfont=dict(size=9,color='white'))
            fig.update_layout(**PL, height=420,
                coloraxis_colorbar=dict(
                    tickfont=dict(color='white'),
                    title=dict(text='Corr',font=dict(color='white'))))
            st.plotly_chart(fig, use_container_width=True)

        with c4:
            section_header("🔤","TOP WORDS")
            nw = st.slider("Words",5,25,15,key='tw_sl')
            def gtw(s,n):
                words = ' '.join(df[df['sentiment']==s]['cleaned_text'].values).split()
                return Counter(w for w in words if len(w)>3).most_common(n)
            pw = gtw('Positive',nw); nww = gtw('Negative',nw)
            fig = make_subplots(rows=1,cols=2,subplot_titles=['😊 Positive','😢 Negative'])
            if pw:
                w,f=zip(*pw)
                fig.add_trace(go.Bar(y=list(w)[::-1],x=list(f)[::-1],orientation='h',
                    marker_color=C['pos'],marker_line_width=0,name='Positive'),row=1,col=1)
            if nww:
                w,f=zip(*nww)
                fig.add_trace(go.Bar(y=list(w)[::-1],x=list(f)[::-1],orientation='h',
                    marker_color=C['neg'],marker_line_width=0,name='Negative'),row=1,col=2)
            fig.update_layout(**PL,height=420,showlegend=False)
            fig.update_annotations(font=dict(color=C['gold'],size=13))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        section_header("☀️","VADER vs ML AGREEMENT")
        df['vader_label']=df['vader_compound'].apply(
            lambda x:'Positive' if x>0.05 else 'Negative' if x<-0.05 else 'Neutral')
        ag = df.groupby(['sentiment','vader_label']).size().reset_index(name='count')
        fig = px.sunburst(ag,path=['sentiment','vader_label'],values='count',color='sentiment',
            color_discrete_map={'Positive':C['pos'],'Negative':C['neg']},template='none')
        fig.update_layout(**PL,height=400)
        fig.update_traces(textfont=dict(color='white',size=13),insidetextorientation='radial')
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 3 — WORD CLOUDS
    # ══════════════════════════════════════════════════════════
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("☁️","SENTIMENT WORD CLOUDS")
        w1, w2 = st.columns(2)

        with w1:
            st.markdown('<p style="color:#00D4AA;font-family:Orbitron;font-size:1rem;'
                'text-align:center;letter-spacing:3px">😊 POSITIVE</p>',unsafe_allow_html=True)
            pt = ' '.join(df[df['sentiment']=='Positive']['cleaned_text'].dropna())
            f  = make_wc(pt,'#041A0F',['#00FF7F','#00FA9A','#7FFF00','#ADFF2F','#00D4AA','#50C878'])
            if f: st.pyplot(f,use_container_width=True); plt.close()

        with w2:
            st.markdown('<p style="color:#FF6B6B;font-family:Orbitron;font-size:1rem;'
                'text-align:center;letter-spacing:3px">😢 NEGATIVE</p>',unsafe_allow_html=True)
            nt = ' '.join(df[df['sentiment']=='Negative']['cleaned_text'].dropna())
            f  = make_wc(nt,'#1A0404',['#FF4500','#FF6347','#FF0000','#DC143C','#FF69B4','#FF8C00'])
            if f: st.pyplot(f,use_container_width=True); plt.close()

        st.markdown("---")
        section_header("📊","TOP WORDS COMPARISON")
        def gf(s,n=30):
            words = ' '.join(df[df['sentiment']==s]['cleaned_text'].values).split()
            return dict(Counter(w for w in words if len(w)>3).most_common(n))
        pf = gf('Positive'); nf = gf('Negative')
        aw = sorted(set(list(pf.keys())[:15]+list(nf.keys())[:15]))
        fig = go.Figure()
        fig.add_trace(go.Bar(name='😊 Positive',x=aw,y=[pf.get(w,0) for w in aw],
            marker_color=C['pos'],opacity=0.85))
        fig.add_trace(go.Bar(name='😢 Negative',x=aw,y=[nf.get(w,0) for w in aw],
            marker_color=C['neg'],opacity=0.85))
        fig.update_layout(**PL,height=380,barmode='group',xaxis_tickangle=-35,
            legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
        st.plotly_chart(fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 4 — MODELS
    # ══════════════════════════════════════════════════════════
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        section_header("🏆","MODEL LEADERBOARD")

        short  = {'🧠 Logistic Regression':'Log.Reg','📊 Naive Bayes':'Naive Bayes',
                  '⚡ Linear SVM':'Linear SVM','🌲 Random Forest':'Rand.Forest'}
        medals = ['🥇','🥈','🥉','4️⃣']
        mc     = ['#00D4AA','#FF9F43','#7B68EE','#54A0FF']
        sr     = sorted(results.items(),key=lambda x:x[1]['accuracy'],reverse=True)

        st.dataframe(pd.DataFrame([{
            'Rank':medals[i],'Model':short.get(n,n),
            'Accuracy':f"{r['accuracy']:.4f}",'F1':f"{r['f1']:.4f}",
            'CV':f"{r['cv_mean']:.4f}±{r['cv_std']:.4f}",
            'AUC':f"{r['roc_auc']:.4f}",'Time':f"{r['time']:.2f}s",
        } for i,(n,r) in enumerate(sr)]),use_container_width=True,hide_index=True)

        st.markdown("---")
        cm1,cm2 = st.columns(2)

        with cm1:
            section_header("🎯","ACCURACY")
            mn=[short.get(n,n) for n,_ in sr]; ac=[r['accuracy'] for _,r in sr]
            fig=go.Figure(go.Bar(x=mn,y=ac,marker_color=mc,marker_line_width=0,
                text=[f'{a:.4f}' for a in ac],textposition='outside',
                textfont=dict(color='white',size=12),
                hovertemplate='<b>%{x}</b><br>Acc: %{y:.4f}<extra></extra>'))
            fig.add_hline(y=0.5,line_dash='dash',line_color='white',opacity=0.3)
            fig.update_layout(**PL,height=380,yaxis_title='Accuracy',
                yaxis_range=[0,max(ac)*1.12])
            st.plotly_chart(fig,use_container_width=True)

        with cm2:
            section_header("📈","ROC CURVES")
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode='lines',
                line=dict(dash='dash',color='white',width=1),opacity=0.4,name='Random'))
            for (n,r),col in zip(sr,mc):
                if r['fpr'] is not None:
                    fig.add_trace(go.Scatter(x=r['fpr'],y=r['tpr'],mode='lines',
                        line=dict(color=col,width=2.5),
                        name=f"{short.get(n,n)} ({r['roc_auc']:.4f})",
                        fill='tozeroy',fillcolor=rgba(col,0.07)))
            fig.update_layout(**PL,height=380,
                xaxis_title='FPR',yaxis_title='TPR',
                legend=dict(font=dict(color='white',size=10),bgcolor='rgba(0,0,0,0)'))
            st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")
        section_header("🔲","CONFUSION MATRICES")
        cms = st.columns(4)
        for (n,r),ci,col in zip(sr,range(4),mc):
            with cms[ci]:
                cm=r['cm']; cp=cm.astype(float)/cm.sum(axis=1,keepdims=True)
                fig=px.imshow(cp,labels=dict(x='Pred',y='True',color='Rate'),
                    x=list(trainer.le.classes_),y=list(trainer.le.classes_),
                    color_continuous_scale=[[0,'#1A1F35'],[1,col]],
                    zmin=0,zmax=1,text_auto='.0%',template='none')
                fig.update_traces(textfont=dict(color='white',size=14))
                fig.update_layout(**PL,height=260,coloraxis_showscale=False,
                    title=dict(text=f"<b>{short.get(n,n)}</b><br><sup>Acc:{r['accuracy']:.4f}</sup>",
                        font=dict(color=col,size=12),x=0.5))
                st.plotly_chart(fig,use_container_width=True)

        st.markdown("---")
        section_header("🕷️","RADAR COMPARISON")
        cats=['Accuracy','F1','CV Score','ROC AUC','Speed']
        mt=max(r['time'] for _,r in sr) or 1
        fig=go.Figure()
        for (n,r),col in zip(sr,mc):
            vs=[r['accuracy'],r['f1'],r['cv_mean'],r['roc_auc'],1-r['time']/mt]
            vs=[max(0,min(1,v)) for v in vs]+[max(0,min(1,vs[0]))]
            cs=cats+[cats[0]]
            fig.add_trace(go.Scatterpolar(r=vs,theta=cs,fill='toself',
                name=short.get(n,n),line=dict(color=col,width=2.5),
                fillcolor=rgba(col,0.12)))
        fig.update_layout(**PL,height=450,
            polar=dict(bgcolor='rgba(26,31,53,0.8)',
                radialaxis=dict(visible=True,range=[0,1],gridcolor='#2A2F45',
                    tickfont=dict(color='white',size=8)),
                angularaxis=dict(gridcolor='#2A2F45',tickfont=dict(color='white',size=12))),
            legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)',
                orientation='h',y=-0.1,x=0.5,xanchor='center'))
        st.plotly_chart(fig,use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # TAB 5 — PREDICT & CHAT  ← THE MAIN FIX IS HERE
    # ══════════════════════════════════════════════════════════
    with tab5:
        st.markdown("<br>", unsafe_allow_html=True)

        # ── LIVE BANNER ────────────────────────────────────────
        st.markdown("""
        <div style="background:linear-gradient(135deg,#1A1F35,#1E2540);
                    border:1px solid #7B68EE33;border-radius:16px;
                    padding:0.8rem 1.5rem;margin-bottom:1.5rem;
                    display:flex;align-items:center;gap:1rem">
            <span class="live-badge">● LIVE</span>
            <span style="color:#9090A0;font-family:Rajdhani;letter-spacing:1px;font-size:0.9rem">
                Real-time Sentiment Analysis Engine
            </span>
        </div>""", unsafe_allow_html=True)

        # ── CHAT STATS ─────────────────────────────────────────
        if st.session_state.total_analyzed > 0:
            total = st.session_state.total_analyzed
            pos_c = st.session_state.pos_chat
            neg_c = st.session_state.neg_chat
            s1,s2,s3,s4 = st.columns(4)
            with s1: metric_card("💬","Analyzed",str(total),C['accent'])
            with s2: metric_card("😊","Positive",str(pos_c),C['pos'])
            with s3: metric_card("😢","Negative",str(neg_c),C['neg'])
            with s4: metric_card("📊","Positivity",
                f"{pos_c/total*100:.0f}%" if total else "—", C['gold'])
            st.markdown("<br>", unsafe_allow_html=True)

        # ── 2-COLUMN LAYOUT ────────────────────────────────────
        main_col, side_col = st.columns([3,1])

        with main_col:
            section_header("💬","SENTIMENT CHAT — TYPE ANY MESSAGE")
            st.markdown(
                '<p style="color:#9090A0;font-family:Rajdhani;letter-spacing:1px;'
                'font-size:0.95rem;margin-bottom:1rem">Type any tweet, sentence, or message.'
                ' Our AI will instantly analyze the sentiment!</p>',
                unsafe_allow_html=True)

            # ── CHAT HISTORY ───────────────────────────────────
            render_chat(trainer)

            # ── MESSAGE INPUT BOX ──────────────────────────────
            # Uses st.form so pressing Enter also submits
            st.markdown("<br>", unsafe_allow_html=True)
            with st.form("chat_form", clear_on_submit=True):
                inp_col, btn_col = st.columns([5,1])
                with inp_col:
                    user_input = st.text_input(
                        "msg",
                        placeholder="💬 Type your tweet or message here...",
                        label_visibility="collapsed",
                    )
                with btn_col:
                    submitted = st.form_submit_button(
                        "Send 🚀", use_container_width=True)

            # ── QUICK EXAMPLES ─────────────────────────────────
            st.markdown(
                '<p style="color:#9090A0;font-family:Rajdhani;font-size:0.8rem;'
                'letter-spacing:1px;margin-top:0.3rem">💡 QUICK EXAMPLES:</p>',
                unsafe_allow_html=True)

            examples = [
                "I love this so much! 😊",
                "This is terrible 😢",
                "Best day of my life!",
                "I'm so sad and lonely",
            ]
            ex_cols = st.columns(4)
            for ec, ex in zip(ex_cols, examples):
                with ec:
                    if st.button(ex, key=f"ex_{ex[:8]}"):
                        st.session_state.pending_msg = ex
                        st.rerun()

            # ── PROCESS pending_msg (from quick example) ───────
            if st.session_state.pending_msg:
                user_input = st.session_state.pending_msg
                st.session_state.pending_msg = ''
                submitted = True

            # ══════════════════════════════════════════════════
            # ✅ FIX: use trainer.tfidf — same as training
            # ══════════════════════════════════════════════════
            if submitted and user_input and user_input.strip():
                text = user_input.strip()

                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'text': text,
                })

                # Predict using THE SAME tfidf from training ✅
                label, vs, tb = predict_sentiment(text, trainer)
                insight       = get_insight(label, vs, tb)

                # Add bot response
                st.session_state.chat_history.append({
                    'role'     : 'bot',
                    'sentiment': label,
                    'vader'    : vs,
                    'textblob' : tb,
                    'insight'  : insight,
                    'text'     : text,
                })

                # Update stats
                st.session_state.total_analyzed += 1
                if label == 'Positive':
                    st.session_state.pos_chat += 1
                else:
                    st.session_state.neg_chat += 1

                st.rerun()

        # ── SIDE PANEL ─────────────────────────────────────────
        with side_col:
            section_header("📊","HOW IT WORKS")
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1A1F35,#1E2540);
                        border:1px solid #7B68EE33;border-radius:16px;padding:1.2rem;
                        font-family:Inter,sans-serif">
                <div style="color:#FFD700;font-family:Orbitron;font-size:0.8rem;
                            letter-spacing:1px;margin-bottom:1rem">🔬 PIPELINE</div>
                {"".join([
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.7rem 0">'
                    f'<div style="background:{bg};border-radius:50%;width:22px;height:22px;'
                    f'display:flex;align-items:center;justify-content:center;'
                    f'font-size:0.65rem;color:white;font-weight:bold;flex-shrink:0">{n}</div>'
                    f'<div style="color:#E0E0E0;font-size:0.8rem">{title}<br>'
                    f'<span style="color:#9090A0;font-size:0.72rem">{sub}</span></div></div>'
                    for n,bg,title,sub in [
                        ('1','#7B68EE','Preprocess','Clean + Normalize'),
                        ('2','#00D4AA','TF-IDF','15K features, bigrams'),
                        ('3','#FF9F43','ML Predict','Best trained model'),
                        ('4','#FF6B6B','VADER+TextBlob','Lexicon verify'),
                    ]
                ])}
                <div style="margin-top:1rem;padding-top:1rem;border-top:1px solid #7B68EE22">
                    <div style="color:#9090A0;font-size:0.78rem;line-height:1.8">
                        🎯 Trained on <b style="color:#FFD700">1.6M tweets</b><br>
                        ⚡ Response <b style="color:#00D4AA">&lt;100ms</b><br>
                        🏆 Accuracy <b style="color:#00D4AA">{best_acc:.2%}</b>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Last score gauge
            if st.session_state.chat_history:
                last = next((m for m in reversed(st.session_state.chat_history)
                             if m['role']=='bot'), None)
                if last:
                    st.markdown("<br>", unsafe_allow_html=True)
                    section_header("⚡","LAST SCORE")
                    vs = last['vader']
                    col_g = C['pos'] if vs > 0 else C['neg']
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number", value=vs,
                        gauge=dict(
                            axis=dict(range=[-1,1],tickcolor='white',
                                      tickfont=dict(color='white',size=8)),
                            bar=dict(color=col_g),
                            bgcolor=C['surface'],
                            steps=[
                                dict(range=[-1,-0.05],color=rgba(C['neg'],0.2)),
                                dict(range=[-0.05,0.05],color=rgba(C['accent'],0.2)),
                                dict(range=[0.05,1],color=rgba(C['pos'],0.2)),
                            ],
                        ),
                        title=dict(text='VADER',font=dict(color='white',size=11)),
                        number=dict(font=dict(color='white',size=18),valueformat='.3f'),
                    ))
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                        height=200,margin=dict(l=15,r=15,t=40,b=5),
                        font=dict(color='white'))
                    st.plotly_chart(fig, use_container_width=True)

            # Clear chat
            if st.session_state.chat_history:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("🗑️ Clear Chat", use_container_width=True, key='clear_btn'):
                    st.session_state.chat_history   = []
                    st.session_state.reactions      = {}
                    st.session_state.total_analyzed = 0
                    st.session_state.pos_chat       = 0
                    st.session_state.neg_chat       = 0
                    st.rerun()

        # ── BATCH PREDICTOR ────────────────────────────────────
        st.markdown("---")
        section_header("📋","BATCH PREDICTOR")

        default = """@switchfoot that's a bummer you shoulda got David Carr
is upset that he can't update his Facebook might cry
I dived many times for the ball. Managed to save 50%
my whole body feels itchy and like its on fire
Need a hug right now please
hey long time no see! LOL I'm fine thanks how are you
spring break in plain city its snowing outside blah
I just re-pierced my ears feeling awesome today!
Hollis death scene will hurt me severely to watch
I miss you so much can't believe you are gone"""

        batch_txt = st.text_area("Tweets (one per line)", value=default,
                                 height=200, key='batch_input')
        if st.button("⚡  ANALYZE ALL", use_container_width=True, key='batch_btn'):
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            tweets = [t.strip() for t in batch_txt.split('\n') if t.strip()]
            with st.spinner(f"Analyzing {len(tweets)} tweets..."):
                vader = SentimentIntensityAnalyzer()
                # ✅ Use trainer.tfidf for batch too
                from preprocess import Sentiment140Preprocessor
                prep    = Sentiment140Preprocessor()
                cleaned = [prep.clean(t) for t in tweets]
                feat    = trainer.tfidf.transform(cleaned)
                preds   = trainer.best_model.predict(feat)
                labels  = trainer.le.inverse_transform(preds)

                rows = []
                for tw, lb in zip(tweets, labels):
                    vs = vader.polarity_scores(tw)['compound']
                    vl = 'Positive' if vs>0.05 else 'Negative' if vs<-0.05 else 'Neutral'
                    rows.append({
                        'Tweet'       : tw[:55]+'…' if len(tw)>55 else tw,
                        'ML Sentiment': ('😊 '+lb) if lb=='Positive' else ('😢 '+lb),
                        'VADER Score' : round(vs,3),
                        'VADER Label' : vl,
                        'Match'       : '✅' if lb==vl else '⚠️',
                    })

            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            bc = pd.Series(labels).value_counts()
            b1, b2 = st.columns(2)
            with b1:
                fig=go.Figure(go.Pie(labels=bc.index,values=bc.values,hole=0.5,
                    marker=dict(colors=[C['pos'],C['neg']],line=dict(color=C['bg'],width=3)),
                    textfont=dict(color='white',size=13)))
                fig.update_layout(**PL,height=280,
                    title=dict(text='Results Distribution',
                        font=dict(color=C['gold'],size=14),x=0.5),
                    legend=dict(font=dict(color='white'),bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig,use_container_width=True)
            with b2:
                vs_list=[r['VADER Score'] for r in rows]
                fig=go.Figure(go.Bar(
                    x=list(range(1,len(tweets)+1)),y=vs_list,
                    marker_color=[C['pos'] if v>0 else C['neg'] for v in vs_list],
                    hovertemplate='Tweet %{x}<br>VADER: %{y:.3f}<extra></extra>'))
                fig.add_hline(y=0,line_color='white',opacity=0.3,line_dash='dash')
                fig.update_layout(**PL,height=280,
                    title=dict(text='VADER per Tweet',
                        font=dict(color=C['gold'],size=14),x=0.5),
                    xaxis_title='Tweet #',yaxis_title='Score')
                st.plotly_chart(fig,use_container_width=True)

# ════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()