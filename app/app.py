import streamlit as st
import pandas as pd
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_footer, render_sidebar

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Motor de Micro-Préstamos Inclusivo",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Rutas relativas al root del proyecto ────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fallback por si __file__ no resuelve correctamente en Streamlit
if not os.path.exists(os.path.join(ROOT, "models")):
    ROOT = os.path.dirname(ROOT)
MODELS_PATH = os.path.join(ROOT, "models")
DATA_PATH   = os.path.join(ROOT, "data", "processed")

# ── CSS Global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fuentes ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

/* ── Variables ── */
:root {
    --bg:         #0A0D16;
    --surface:    #111827;
    --surface2:   #1A2235;
    --accent:     #00D4AA;
    --accent2:    #FF6B6B;
    --accent3:    #F59E0B;
    --text:       #F0F4FF;
    --text-muted: #6B7A99;
    --border:     rgba(0, 212, 170, 0.12);
    --font-display: 'Syne', sans-serif;
    --font-body:    'DM Sans', sans-serif;
    --font-mono:    'DM Mono', monospace;
}

/* ── Reset global ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarNav"] a {
    border-radius: 10px !important;
    margin-bottom: 4px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stSidebarNav"] a:hover {
    background: rgba(0,212,170,0.08) !important;
    padding-left: 18px !important;
}

/* ── Bloques principales ── */
[data-testid="block-container"] {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
}

/* ── Métricas ── */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,170,0.08) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 0.78rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stMetricValue"] { font-family: var(--font-display) !important; color: var(--accent) !important; font-size: 2rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Botones ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #00A887) !important;
    color: #0A0D16 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.8rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(0,212,170,0.35) !important;
}

/* ── Divisor ── */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Animación fade-in ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.fade-up { animation: fadeUp 0.6s ease forwards; }
.fade-up-2 { animation: fadeUp 0.6s ease 0.15s forwards; opacity: 0; }
.fade-up-3 { animation: fadeUp 0.6s ease 0.30s forwards; opacity: 0; }
            
[data-testid="block-container"] {
    padding-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Carga de datos con cache ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_summary():
    path = os.path.join(MODELS_PATH, "optimization_summary.json")
    with open(path, "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_shap_summary():
    path = os.path.join(MODELS_PATH, "shap_summary.json")
    with open(path, "r") as f:
        return json.load(f)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem 0;'>
        <div style='font-family: var(--font-display); font-size: 1.1rem; font-weight: 800;
                    color: #00D4AA; letter-spacing: 0.05em;'>
            MICRO-PRÉSTAMOS
        </div>
        <div style='font-size: 0.72rem; color: #6B7A99; letter-spacing: 0.12em;
                    text-transform: uppercase; margin-top: 2px;'>
            Motor Inclusivo · v1.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    #st.markdown("---")

    #st.markdown("""
    #<div style='font-size: 0.7rem; color: #6B7A99; letter-spacing: 0.1em;
    #            text-transform: uppercase; margin-bottom: 0.5rem;'>
    #    Navegación
    #</div>
    #""", unsafe_allow_html=True)

    # Info del modelo
    st.markdown("---")
    st.markdown("""
    <div style='background: rgba(0,212,170,0.06); border: 1px solid rgba(0,212,170,0.15);
                border-radius: 12px; padding: 0.9rem 1rem;'>
        <div style='font-size: 0.68rem; color: #6B7A99; letter-spacing: 0.1em;
                    text-transform: uppercase; margin-bottom: 0.6rem;'>
            Modelo activo
        </div>
        <div style='font-family: "DM Mono", monospace; font-size: 0.78rem; color: #00D4AA;'>
            LightGBM Tuned
        </div>
        <div style='font-size: 0.72rem; color: #F0F4FF; margin-top: 4px;'>
            ROC-AUC <span style='color:#00D4AA; font-weight:600;'>0.7440</span>
        </div>
        <div style='font-size: 0.72rem; color: #F0F4FF; margin-top: 2px;'>
            Threshold <span style='color:#F59E0B; font-weight:600;'>0.35</span>
        </div>
        <div style='font-size: 0.72rem; color: #F0F4FF; margin-top: 2px;'>
            Features <span style='color:#F0F4FF; font-weight:600;'>234</span>
            <span style='color:#00D4AA;'>(7 alt.)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size: 0.65rem; color: #6B7A99; text-align: center; line-height: 1.6;'>
        Dataset: Home Credit Default Risk<br>
        307,511 solicitantes · Kaggle<br><br>
        <span style='color: #00D4AA;'>Abraham Tartalos</span> · 2026
    </div>
    """, unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='fade-up' style='margin-bottom: 0.5rem;'>
    <span style='font-size: 0.72rem; color: #00D4AA; letter-spacing: 0.2em;
                 text-transform: uppercase; font-family: "DM Mono", monospace;'>
        Ciencia de Datos para Inclusión Financiera
    </span>
</div>
<div class='fade-up' style='
    font-family: "Syne", sans-serif;
    font-size: clamp(2.2rem, 5vw, 3.8rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.02em;
    margin-bottom: 1rem;
'>
    Motor de Micro-Préstamos<br>
    <span style='
        background: linear-gradient(135deg, #00D4AA, #00A0FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    '>Inclusivo</span>
</div>
<div class='fade-up-2' style='
    font-size: 1.05rem;
    color: #6B7A99;
    max-width: 580px;
    line-height: 1.7;
    margin-bottom: 2.5rem;
'>
    Sistema de scoring crediticio para los <strong style='color:#F0F4FF;'>7 millones</strong>
    de trabajadores informales en Argentina que el sistema financiero tradicional
    ignora — usando señales alternativas de estabilidad e inclusión.
</div>
""", unsafe_allow_html=True)


# ── KPI CARDS ────────────────────────────────────────────────────────────────
try:
    summary = load_summary()
    bi      = summary["business_impact"]
    fm      = summary["final_metrics"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="ROC-AUC",
            value=f"{fm['roc_auc']:.4f}",
            delta="98.7% del target",
            delta_color="normal",
        )
    with col2:
        st.metric(
            label="Nuevas aprobaciones",
            value=f"{int(bi['additional_approvals']):,}",
            delta=f"+{bi['pct_increase']:.1f}% vs baseline",
        )
    with col3:
        st.metric(
            label="Default rate",
            value=f"{bi['default_rate']:.2f}%",
            delta="Controlado",
            delta_color="off",
        )
    with col4:
        st.metric(
            label="Impacto económico",
            value="$1,104M",
            delta="USD estimado",
            delta_color="off",
        )
except Exception:
    st.info("Cargando métricas...")

st.markdown("<br>", unsafe_allow_html=True)


# ── PROBLEMA / SOLUCIÓN ──────────────────────────────────────────────────────
st.markdown("""
<div class='fade-up-2' style='
    background: linear-gradient(135deg, rgba(0,212,170,0.04), rgba(0,160,255,0.04));
    border: 1px solid rgba(0,212,170,0.12);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
'>
    <div style='font-family:"Syne",sans-serif; font-size:1.25rem; font-weight:700;
                margin-bottom:1.2rem; color:#F0F4FF;'>
        El problema que resolvemos
    </div>
    <div style='display:grid; grid-template-columns: 1fr 1fr; gap: 2rem;'>
        <div>
            <div style='font-size:0.7rem; color:#FF6B6B; letter-spacing:0.15em;
                        text-transform:uppercase; margin-bottom:0.5rem;'>
                Sin este modelo
            </div>
            <div style='font-size:0.95rem; color:#6B7A99; line-height:1.7;'>
                Un trabajador informal sin recibo de sueldo, cuenta bancaria
                o historial crediticio formal es <em style='color:#FF6B6B;'>
                automáticamente rechazado</em> por los sistemas tradicionales,
                sin importar su comportamiento real de pago.
            </div>
        </div>
        <div>
            <div style='font-size:0.7rem; color:#00D4AA; letter-spacing:0.15em;
                        text-transform:uppercase; margin-bottom:0.5rem;'>
                Con este modelo
            </div>
            <div style='font-size:0.95rem; color:#6B7A99; line-height:1.7;'>
                Usamos <strong style='color:#00D4AA;'>7 señales alternativas</strong>
                — estabilidad de ingresos, arraigo domiciliario, inclusión digital —
                para evaluar a quienes el sistema tradicional invisibiliza.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── FEATURES ALTERNATIVOS ────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
            margin-bottom: 1.2rem; color:#F0F4FF;'>
    Las 7 señales alternativas
    <span style='font-size:0.7rem; color:#6B7A99; font-weight:400;
                 font-family:"DM Sans",sans-serif; margin-left:0.5rem;'>
        — el diferenciador del modelo
    </span>
</div>
""", unsafe_allow_html=True)

try:
    shap_s = load_shap_summary()
    alt_ranking = shap_s.get("alternative_features_ranking", [])

    descriptions = {
        "EXT_SOURCE_COMBINED":       ("Score externo combinado",    "Promedio ponderado de 3 fuentes externas de crédito",         "#00D4AA"),
        "FINANCIAL_INCLUSION_SCORE": ("Inclusión financiera",       "Tenencia de móvil, email y documentación completa",           "#00A0FF"),
        "INCOME_STABILITY_SCORE_ADJ":("Estabilidad de ingresos",    "Tipo de empleo + años de antigüedad laboral",                 "#F59E0B"),
        "ADDRESS_TENURE_SCORE":      ("Arraigo domiciliario",       "Tiempo en la misma dirección — señal de estabilidad de vida", "#A78BFA"),
        "CREDIT_BUREAU_SCORE":       ("Historial bureau",           "Inversamente proporcional a consultas al bureau",             "#34D399"),
        "EMPLOYMENT_STABILITY":      ("Estabilidad laboral",        "Años de empleo normalizados sobre 20",                        "#FB923C"),
        "PAYMENT_BURDEN_SCORE":      ("Carga de pagos",             "Relación cuota + crédito respecto al ingreso (invertida)",    "#F472B6"),
    }

    # Fila 1: 4 cards
    row1 = alt_ranking[:4]
    cols1 = st.columns(4)
    for col, feat_data in zip(cols1, row1):
        feat  = feat_data["Feature"]
        rank  = feat_data["Rank"]
        shap  = feat_data["Mean_Abs_SHAP"]
        desc  = descriptions.get(feat, (feat, "", "#00D4AA"))
        color = desc[2]
        with col:
            st.markdown(f"""
            <div style='background:var(--surface2); border:1px solid {color}22;
                        border-left:3px solid {color}; border-radius:14px;
                        padding:1rem 1.1rem; margin-bottom:0.8rem; height:140px;'>
                <div style='font-size:0.62rem; color:{color}; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:0.3rem;'>
                    #{rank} global · SHAP {shap:.4f}
                </div>
                <div style='font-family:"Syne",sans-serif; font-size:0.88rem;
                            font-weight:700; color:#F0F4FF; margin-bottom:0.3rem;'>
                    {desc[0]}
                </div>
                <div style='font-size:0.75rem; color:#6B7A99; line-height:1.5;'>
                    {desc[1]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Fila 2: 3 cards centradas con columnas vacías a los costados
    _, c1, c2, c3, _ = st.columns([0.5, 1, 1, 1, 0.5])
    row2 = alt_ranking[4:]
    for col, feat_data in zip([c1, c2, c3], row2):
        feat  = feat_data["Feature"]
        rank  = feat_data["Rank"]
        shap  = feat_data["Mean_Abs_SHAP"]
        desc  = descriptions.get(feat, (feat, "", "#00D4AA"))
        color = desc[2]
        with col:
            st.markdown(f"""
            <div style='background:var(--surface2); border:1px solid {color}22;
                        border-left:3px solid {color}; border-radius:14px;
                        padding:1rem 1.1rem; margin-bottom:0.8rem; height:140px;'>
                <div style='font-size:0.62rem; color:{color}; letter-spacing:0.12em;
                            text-transform:uppercase; margin-bottom:0.3rem;'>
                    #{rank} global · SHAP {shap:.4f}
                </div>
                <div style='font-family:"Syne",sans-serif; font-size:0.88rem;
                            font-weight:700; color:#F0F4FF; margin-bottom:0.3rem;'>
                    {desc[0]}
                </div>
                <div style='font-size:0.75rem; color:#6B7A99; line-height:1.5;'>
                    {desc[1]}
                </div>
            </div>
            """, unsafe_allow_html=True)

except Exception:
    st.info("Cargando features alternativos...")


# ── IMPACTO SOCIAL ───────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_i1, col_i2, col_i3 = st.columns(3)
impacto_items = [
    ("7M",   "#00D4AA", "trabajadores informales<br>en Argentina",       "45% de la economía argentina<br>es informal"),
    ("45%",  "#F59E0B", "de la economía argentina<br>es informal",       "El mercado informal es enorme<br>e invisible para el crédito"),
    ("182K", "#A78BFA", "nuevos clientes potenciales<br>en el mercado argentino", "Proyección del modelo sobre<br>el mercado total"),
]
for col, (num, color, label, sub) in zip([col_i1, col_i2, col_i3], impacto_items):
    with col:
        st.markdown(f"""
        <div style='background:#1A2235; border:1px solid {color}22;
                    border-top:3px solid {color}; border-radius:16px;
                    padding:1.5rem 1rem; text-align:center;'>
            <div style='font-family:"Syne",sans-serif; font-size:2.8rem;
                        font-weight:800; color:{color}; line-height:1;'>
                {num}
            </div>
            <div style='font-size:0.8rem; color:#6B7A99; margin-top:0.4rem;
                        line-height:1.5;'>{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ── CTA ──────────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_a, col_b, col_c = st.columns([1, 2, 1])
with col_b:
    st.markdown("""
    <div style='text-align:center; margin-bottom:1rem;'>
        <div style='font-family:"Syne",sans-serif; font-size:1rem; font-weight:600;
                    color:#F0F4FF; margin-bottom:0.5rem;'>
            ¿Querés ver el modelo en acción?
        </div>
        <div style='font-size:0.82rem; color:#6B7A99;'>
            Usá el <strong style='color:#00D4AA;'>Simulador</strong> en el menú lateral
            para ingresar el perfil de un solicitante y obtener una decisión explicada.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ───────────────────────────────────────────────────────────────────
render_footer()
