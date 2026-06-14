# app/streamlit_app.py

import streamlit as st
import numpy as np
import cv2
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import os, sys, time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# ── Must be FIRST Streamlit call ─────────────────────────────────
st.set_page_config(
    page_title="EmotionAI — Facial Emotion Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

from app.utils import (
    load_model, preprocess_image, predict_emotion,
    detect_and_predict, draw_face_boxes, process_webcam_frame,
    EMOTIONS, EMOTION_COLORS, EMOTION_EMOJIS
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #1a1a2e 100%);
    border-right: 1px solid #2a2a4a;
}
.metric-card {
    background: linear-gradient(135deg, #1e1e3a, #2a2a4a);
    border: 1px solid #3a3a6a;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.metric-label {
    font-size: 0.85rem;
    color: #888;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.emotion-badge {
    display: inline-block;
    padding: 10px 24px;
    border-radius: 50px;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 10px 0;
    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
}
.section-header {
    font-size: 1.1rem;
    font-weight: 600;
    color: #a0a0d0;
    border-bottom: 1px solid #3a3a6a;
    padding-bottom: 6px;
    margin: 16px 0 12px 0;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.info-box {
    background: #1e1e3a;
    border-left: 4px solid #667eea;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 8px 0;
    font-size: 0.9rem;
}
.face-card {
    background: linear-gradient(135deg, #1e1e3a, #2a2a4a);
    border: 1px solid #3a3a6a;
    border-radius: 10px;
    padding: 14px;
    margin: 8px 0;
}
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 24px;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.85; }
.webcam-box {
    background: #1a1a2e;
    border: 2px dashed #4a4a8a;
    border-radius: 12px;
    padding: 30px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 20px 0 10px 0;'>
        <div style='font-size:3rem;'>🧠</div>
        <div style='font-size:1.4rem; font-weight:700;
                    background:linear-gradient(90deg,#667eea,#764ba2);
                    -webkit-background-clip:text;
                    -webkit-text-fill-color:transparent;'>
            EmotionAI
        </div>
        <div style='font-size:0.75rem; color:#666; margin-top:4px;'>
            Facial Emotion Detection System
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        ["🏠 Home", "📸 Image Analysis", "📹 Webcam Live", "📊 Model Insights"],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("<div class='section-header'>Model Status</div>",
                unsafe_allow_html=True)
    model = load_model()
    if model is not None:
        st.success("✅ Model Loaded")
        params = model.count_params()
        st.markdown(f"""
        <div class='info-box'>
            🔢 <b>Parameters:</b> {params:,}<br>
            🎯 <b>Classes:</b> 7 Emotions<br>
            📐 <b>Input:</b> 48 × 48 px
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ Model not found")
        st.caption("Run: `python src/train.py`")

    st.divider()
    st.markdown("<div class='section-header'>Detectable Emotions</div>",
                unsafe_allow_html=True)
    for e in EMOTIONS:
        st.markdown(
            f"<span style='color:{EMOTION_COLORS[e]};font-size:1rem;'>"
            f"{EMOTION_EMOJIS[e]} {e}</span>",
            unsafe_allow_html=True
        )
    st.divider()
    st.markdown(
        "<div style='text-align:center;font-size:0.7rem;color:#444;'>"
        "Built with TensorFlow & Streamlit<br>"
        "FER2013 Dataset · CNN Architecture"
        "</div>", unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════
def plot_confidence_chart(all_probs: dict, key: str = "") -> go.Figure:
    emotions = list(all_probs.keys())
    values   = list(all_probs.values())
    colors   = [EMOTION_COLORS[e] for e in emotions]
    fig = go.Figure(go.Bar(
        x=values, y=emotions, orientation='h',
        marker=dict(color=colors,
                    line=dict(color='rgba(255,255,255,0.1)', width=1)),
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Confidence: %{x:.2f}%<extra></extra>'
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#ccc', size=12),
        xaxis=dict(range=[0, 115], showgrid=True,
                   gridcolor='rgba(255,255,255,0.05)',
                   title='Confidence (%)', color='#888'),
        yaxis=dict(autorange='reversed', color='#ccc'),
        height=300, margin=dict(l=10, r=60, t=10, b=30),
        showlegend=False
    )
    return fig


def plot_gauge(confidence: float, emotion: str) -> go.Figure:
    color = EMOTION_COLORS[emotion]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence,
        number={'suffix': '%', 'font': {'size': 32, 'color': color}},
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor='#888',
                      tickfont=dict(color='#888')),
            bar=dict(color=color, thickness=0.3),
            bgcolor='rgba(30,30,60,0.8)', borderwidth=0,
            steps=[
                dict(range=[0,  40], color='rgba(255,255,255,0.03)'),
                dict(range=[40, 70], color='rgba(255,255,255,0.06)'),
                dict(range=[70,100], color='rgba(255,255,255,0.09)')
            ],
            threshold=dict(line=dict(color=color, width=3),
                           thickness=0.8, value=confidence)
        )
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#ccc'),
        height=200, margin=dict(l=20, r=20, t=20, b=10)
    )
    return fig


def render_face_results(results: list, key_prefix: str = ""):
    """Render emotion results for one or multiple faces."""
    if not results:
        st.warning("No faces detected.")
        return

    num_faces = len(results)
    st.markdown(
        f"<div class='section-header'>🎯 {num_faces} Face(s) Detected</div>",
        unsafe_allow_html=True
    )

    for i, r in enumerate(results):
        emotion    = r['emotion']
        confidence = r['confidence']
        all_probs  = r['all_probs']
        color      = EMOTION_COLORS[emotion]
        emoji      = EMOTION_EMOJIS[emotion]

        with st.container():
            if num_faces > 1:
                st.markdown(
                    f"<div style='color:#a0a0d0;font-weight:600;"
                    f"margin-top:12px;'>👤 Face #{i+1}</div>",
                    unsafe_allow_html=True
                )

            # Badge
            st.markdown(
                f"<div style='text-align:center;'>"
                f"<div class='emotion-badge' style='background:{color}22;"
                f"border:2px solid {color};color:{color};'>"
                f"{emoji} {emotion}</div></div>",
                unsafe_allow_html=True
            )

            # Gauge
            st.plotly_chart(
                plot_gauge(confidence, emotion),
                use_container_width=True,
                key=f"{key_prefix}_gauge_{i}"
            )

            # Bar chart
            st.markdown("<div class='section-header'>All Probabilities</div>",
                        unsafe_allow_html=True)
            st.plotly_chart(
                plot_confidence_chart(all_probs),
                use_container_width=True,
                key=f"{key_prefix}_bar_{i}"
            )

            # Raw table
            with st.expander(f"📋 Raw Scores — Face #{i+1}"):
                df = pd.DataFrame(
                    list(all_probs.items()),
                    columns=['Emotion', 'Confidence (%)']
                ).sort_values('Confidence (%)', ascending=False)
                df['Emoji'] = df['Emotion'].map(EMOTION_EMOJIS)
                st.dataframe(
                    df[['Emoji', 'Emotion', 'Confidence (%)']],
                    hide_index=True, use_container_width=True
                )

            if num_faces > 1 and i < num_faces - 1:
                st.divider()


# ════════════════════════════════════════════════════════════════
#  PAGE: HOME
# ════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div style='text-align:center; padding: 40px 0 20px 0;'>
        <div style='font-size:4rem;'>🧠</div>
        <h1 style='font-size:2.8rem; font-weight:800; margin:0;
                   background:linear-gradient(90deg,#667eea,#764ba2,#f093fb);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
            EmotionAI
        </h1>
        <p style='color:#888; font-size:1.1rem; margin-top:8px;'>
            Real-time Facial Emotion Detection · Multi-Face · Webcam Support
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        ["7", "28,709", "Multi-Face", "Live Webcam"],
        ["Emotion Classes", "Training Images", "Detection Mode", "Support"]
    ):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{val}</div>
            <div class='metric-label'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### 🚀 Features")
        for icon, title, desc in [
            ("📸", "Image Upload",     "Analyse emotions from any photo"),
            ("👥", "Multi-Face",       "Detect & classify all faces simultaneously"),
            ("📹", "Live Webcam",      "Real-time emotion detection from camera"),
            ("📊", "Confidence Gauge", "Visual confidence scores for all 7 emotions"),
            ("🗺️", "Face Boxes",      "Colour-coded bounding boxes per emotion"),
        ]:
            st.markdown(
                f"<div class='info-box'>{icon} <b>{title}</b> — {desc}</div>",
                unsafe_allow_html=True
            )

    with col_r:
        st.markdown("### 🎭 Supported Emotions")
        for e in EMOTIONS:
            st.markdown(
                f"<div class='info-box'>"
                f"<span style='color:{EMOTION_COLORS[e]};font-size:1.3rem;'>"
                f"{EMOTION_EMOJIS[e]}</span>  <b>{e}</b></div>",
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Use the sidebar — try **Image Analysis** or **Webcam Live**!")


# ════════════════════════════════════════════════════════════════
#  PAGE: IMAGE ANALYSIS  (multi-face)
# ════════════════════════════════════════════════════════════════
elif page == "📸 Image Analysis":
    st.markdown("## 📸 Image Emotion Analysis")
    st.markdown("Upload a photo — all faces are detected and classified simultaneously.")

    if model is None:
        st.error("⚠️ Model not loaded. Run `python src/train.py` first.")
        st.stop()

    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=['jpg', 'jpeg', 'png', 'webp'],
        help="Best results with clear, front-facing faces"
    )

    if uploaded:
        pil_img = Image.open(uploaded).convert('RGB')
        img_arr = np.array(pil_img)

        col_img, col_res = st.columns([1.2, 1])

        with col_img:
            st.markdown("<div class='section-header'>Input Image</div>",
                        unsafe_allow_html=True)
            with st.spinner("🔍 Detecting faces & predicting emotions..."):
                results = detect_and_predict(model, img_arr)

            if results:
                annotated = draw_face_boxes(img_arr, results)
                st.image(annotated,
                         caption=f"✅ {len(results)} face(s) detected",
                         use_container_width=True)
            else:
                st.image(img_arr, caption="Original image",
                         use_container_width=True)
                st.warning("⚠️ No faces detected — running whole-image analysis...")
                tensor  = preprocess_image(img_arr)
                em, cf, ap = predict_emotion(model, tensor)
                results = [{'bbox': None, 'emotion': em,
                            'confidence': cf, 'all_probs': ap}]

        with col_res:
            render_face_results(results, key_prefix="img")

    else:
        st.markdown("""
        <div style='text-align:center; padding:60px 20px;
                    border:2px dashed #3a3a6a; border-radius:16px;
                    background:#1a1a2e;'>
            <div style='font-size:3rem;'>📤</div>
            <div style='color:#888; margin-top:12px;'>
                Upload an image to begin emotion analysis
            </div>
            <div style='color:#555; font-size:0.85rem; margin-top:8px;'>
                Supports JPG · PNG · WEBP &nbsp;|&nbsp; Multiple faces detected automatically
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: WEBCAM LIVE
# ════════════════════════════════════════════════════════════════
elif page == "📹 Webcam Live":
    st.markdown("## 📹 Live Webcam Emotion Detection")
    st.markdown("Real-time multi-face emotion analysis from your webcam.")

    if model is None:
        st.error("⚠️ Model not loaded. Run `python src/train.py` first.")
        st.stop()

    # ── Controls ───────────────────────────────────────────────
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1, 1, 2])
    with col_ctrl1:
        start_btn = st.button("▶ Start Webcam", key="start")
    with col_ctrl2:
        stop_btn  = st.button("⏹ Stop",         key="stop")
    with col_ctrl3:
        frame_rate = st.slider("Frame delay (ms)", 50, 500, 100, 50,
                               help="Lower = faster, higher = less CPU")

    st.divider()

    # Session state
    if 'webcam_running' not in st.session_state:
        st.session_state.webcam_running = False
    if start_btn:
        st.session_state.webcam_running = True
    if stop_btn:
        st.session_state.webcam_running = False

    # ── Layout ─────────────────────────────────────────────────
    col_feed, col_info = st.columns([1.4, 1])

    with col_feed:
        st.markdown("<div class='section-header'>📡 Live Feed</div>",
                    unsafe_allow_html=True)
        frame_placeholder = st.empty()

    with col_info:
        st.markdown("<div class='section-header'>📊 Live Results</div>",
                    unsafe_allow_html=True)
        results_placeholder = st.empty()
        stats_placeholder   = st.empty()

    # ── Webcam loop ────────────────────────────────────────────
    if st.session_state.webcam_running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("""
            ❌ **Webcam not accessible.**

            **Common fixes:**
            - Allow camera access in your browser/OS settings
            - Close other apps using the camera (Zoom, Teams, etc.)
            - Try a different camera index (built-in vs external)
            """)
            st.session_state.webcam_running = False
        else:
            st.success("🟢 Webcam active — click **Stop** to end")
            frame_count  = 0
            emotion_log  = []

            try:
                while st.session_state.webcam_running:
                    ret, frame_bgr = cap.read()
                    if not ret:
                        st.warning("⚠️ Frame capture failed — retrying...")
                        time.sleep(0.1)
                        continue

                    frame_count += 1

                    # Mirror effect
                    frame_bgr = cv2.flip(frame_bgr, 1)

                    # Process every frame
                    annotated_bgr, results = process_webcam_frame(model, frame_bgr)

                    # Convert BGR → RGB for display
                    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(
                        annotated_rgb,
                        caption=f"Frame #{frame_count} · {len(results)} face(s)",
                        use_container_width=True
                    )

                    # Live results panel
                    if results:
                        emotion_log.extend([r['emotion'] for r in results])
                        # Keep last 50 entries
                        emotion_log = emotion_log[-50:]

                        results_md = ""
                        for i, r in enumerate(results):
                            em  = r['emotion']
                            cf  = r['confidence']
                            col = EMOTION_COLORS[em]
                            emoji = EMOTION_EMOJIS[em]
                            results_md += (
                                f"<div class='face-card'>"
                                f"<span style='color:{col};font-size:1.1rem;"
                                f"font-weight:700;'>{emoji} Face #{i+1}: {em}</span><br>"
                                f"<span style='color:#aaa;font-size:0.9rem;'>"
                                f"Confidence: <b style='color:{col};'>{cf:.1f}%</b></span>"
                                f"</div>"
                            )
                        results_placeholder.markdown(results_md, unsafe_allow_html=True)

                        # Emotion frequency stats
                        if len(emotion_log) >= 5:
                            from collections import Counter
                            counts = Counter(emotion_log)
                            dominant = counts.most_common(1)[0][0]
                            stats_placeholder.markdown(
                                f"<div class='info-box'>"
                                f"🏆 <b>Dominant emotion:</b> "
                                f"<span style='color:{EMOTION_COLORS[dominant]};'>"
                                f"{EMOTION_EMOJIS[dominant]} {dominant}</span><br>"
                                f"📊 <b>Frames analysed:</b> {frame_count}"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        results_placeholder.markdown(
                            "<div class='info-box'>👀 No faces in frame</div>",
                            unsafe_allow_html=True
                        )

                    time.sleep(frame_rate / 1000)

            except Exception as e:
                st.error(f"Webcam error: {e}")
            finally:
                cap.release()
                frame_placeholder.markdown(
                    "<div class='webcam-box'>"
                    "<div style='font-size:2rem;'>📷</div>"
                    "<div style='color:#666;margin-top:8px;'>Webcam stopped</div>"
                    "</div>",
                    unsafe_allow_html=True
                )

    else:
        frame_placeholder.markdown("""
        <div class='webcam-box'>
            <div style='font-size:3rem;'>📹</div>
            <div style='color:#888; margin-top:12px; font-size:1rem;'>
                Click <b>▶ Start Webcam</b> above to begin
            </div>
            <div style='color:#555; font-size:0.85rem; margin-top:8px;'>
                Your camera will be accessed · All faces detected in real-time
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════
elif page == "📊 Model Insights":
    st.markdown("## 📊 Model Insights & Dataset Statistics")

    st.markdown("### 🏗️ CNN Architecture")
    arch_data = {
        'Block':   ['Block 1', 'Block 2', 'Block 3', 'Block 4', 'FC Head'],
        'Layers':  ['Conv64×2 + BN + Pool', 'Conv128×2 + BN + Pool',
                    'Conv256×2 + BN + Pool', 'Conv512×2 + BN + GAP',
                    'Dense512 + Dense256 + Softmax7'],
        'Dropout': ['25%', '25%', '35%', '50%', '50% / 30%'],
        'Purpose': ['Edge detection', 'Low-level features',
                    'Mid-level features', 'High-level features', 'Classification']
    }
    st.dataframe(pd.DataFrame(arch_data), hide_index=True, use_container_width=True)

    st.markdown("### 📊 FER2013 Dataset Distribution")
    dist_data = {
        'Emotion': EMOTIONS,
        'Train':   [3995, 436, 4097, 7215, 4965, 4830, 3171],
        'Test':    [958,  111, 1024, 1774, 1233, 1247,  831]
    }
    df_dist = pd.DataFrame(dist_data)
    fig = px.bar(
        df_dist.melt(id_vars='Emotion', var_name='Split', value_name='Count'),
        x='Emotion', y='Count', color='Split', barmode='group',
        color_discrete_map={'Train': '#667eea', 'Test': '#764ba2'},
        template='plotly_dark'
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=380, legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📈 Training History")
    log_path = 'models/checkpoints/training_log.csv'
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df_log['epoch'], y=df_log['accuracy'],
            name='Train Acc', line=dict(color='#667eea', width=2)
        ))
        fig2.add_trace(go.Scatter(
            x=df_log['epoch'], y=df_log['val_accuracy'],
            name='Val Acc', line=dict(color='#f093fb', width=2, dash='dash')
        ))
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ccc'), height=350,
            xaxis=dict(title='Epoch', gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(title='Accuracy', gridcolor='rgba(255,255,255,0.05)'),
            legend=dict(bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig2, use_container_width=True)

        best_val = df_log['val_accuracy'].max()
        best_ep  = df_log['val_accuracy'].idxmax() + 1
        c1, c2, c3 = st.columns(3)
        c1.metric("Best Val Accuracy", f"{best_val*100:.2f}%")
        c2.metric("Best Epoch",        str(best_ep))
        c3.metric("Total Epochs Run",  str(len(df_log)))
    else:
        st.info("⏳ Training log will appear here once training completes.")

    st.markdown("### 🗺️ Confusion Matrix")
    cm_path = 'models/confusion_matrix.png'
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
    else:
        st.info("⏳ Confusion matrix will appear after training completes.")

#streamlit run app/streamlit_app.py