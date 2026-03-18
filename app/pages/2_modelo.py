import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import re
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import streamlit.components.v1 as components
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_footer, render_sidebar

def responsive_height():
    """Devuelve altura según si es mobile o desktop."""
    return 280  # Plotly se encarga del ancho con width='stretch'

# ── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Modelo · Motor Inclusivo",
    page_icon="📊",
    #page_icon_and_name="🏦 Inicio",  # controla el nombre en el menú lateral
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(ROOT, "models")):
    ROOT = os.path.dirname(ROOT)
MODELS_PATH = os.path.join(ROOT, "models")
DATA_PATH   = os.path.join(ROOT, "data", "processed")

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');
:root {
    --bg:#0A0D16; --surface:#111827; --surface2:#1A2235;
    --accent:#00D4AA; --accent2:#FF6B6B; --accent3:#F59E0B;
    --text:#F0F4FF; --text-muted:#6B7A99; --border:rgba(0,212,170,0.12);
    --font-display:'Syne',sans-serif; --font-body:'DM Sans',sans-serif;
    --font-mono:'DM Mono',monospace;
}
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricLabel"] { color:var(--text-muted) !important; font-size:0.75rem !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { font-family:var(--font-display) !important; font-size:1.6rem !important; }

[data-testid="stPlotlyChart"] {
    min-width: 0 !important;
    overflow-x: auto !important;
}
.js-plotly-plot {
    min-width: 280px !important;
}
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
            
/* Gráficos responsivos en mobile */
[data-testid="stPlotlyChart"] > div {
    overflow-x: auto !important;
    -webkit-overflow-scrolling: touch !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly tema base ─────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,34,53,0.6)',
    font=dict(family='DM Sans', color='#F0F4FF', size=12),
    xaxis=dict(gridcolor='rgba(107,122,153,0.15)', zerolinecolor='rgba(107,122,153,0.2)'),
    yaxis=dict(gridcolor='rgba(107,122,153,0.15)', zerolinecolor='rgba(107,122,153,0.2)'),
    margin=dict(l=20, r=20, t=40, b=20),
)

# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_feature_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns]
    return df

THRESHOLD = 0.35
ALTERNATIVE_FEATURES = [
    'EXT_SOURCE_COMBINED', 'FINANCIAL_INCLUSION_SCORE',
    'INCOME_STABILITY_SCORE_ADJ', 'ADDRESS_TENURE_SCORE',
    'CREDIT_BUREAU_SCORE', 'EMPLOYMENT_STABILITY', 'PAYMENT_BURDEN_SCORE',
]

# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_lgbm():
    with open(os.path.join(MODELS_PATH, "lgbm_tuned_v2.pkl"), "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_rf():
    # rf_baseline.pkl está corrupto — usamos valores de Fase 3 directamente
    return None

@st.cache_data(show_spinner=False)
def load_summary():
    with open(os.path.join(MODELS_PATH, "optimization_summary.json"), "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_shap_summary():
    with open(os.path.join(MODELS_PATH, "shap_summary.json"), "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def prepare_test_data():
    """Carga y prepara X_test, y_test con el mismo split de Fase 3."""
    df = pd.read_csv(os.path.join(DATA_PATH, "train_processed_clean.csv"))
    y  = df["TARGET"]
    X  = df.drop(columns=["TARGET"])
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_test_clean = clean_feature_names(X_test).reset_index(drop=True)
    y_test       = y_test.reset_index(drop=True)
    return X_test_clean, y_test

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:1rem 0 1.5rem 0;'>
        <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:800;
                    color:#00D4AA; letter-spacing:0.05em;'>MICRO-PRÉSTAMOS</div>
        <div style='font-size:0.72rem; color:#6B7A99; letter-spacing:0.12em;
                    text-transform:uppercase; margin-top:2px;'>Motor Inclusivo · v1.0</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.15);
                border-radius:12px; padding:0.9rem 1rem;'>
        <div style='font-size:0.68rem; color:#6B7A99; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.5rem;'>Modelo activo</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.78rem; color:#00D4AA;'>LightGBM Tuned</div>
        <div style='font-size:0.72rem; color:#F0F4FF; margin-top:4px;'>
            ROC-AUC <span style='color:#00D4AA; font-weight:600;'>0.7440</span></div>
        <div style='font-size:0.72rem; color:#F0F4FF; margin-top:2px;'>
            Threshold <span style='color:#F59E0B; font-weight:600;'>0.35</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.3rem;'>
    <span style='font-size:0.72rem; color:#00D4AA; letter-spacing:0.2em;
                 text-transform:uppercase; font-family:"DM Mono",monospace;'>
        Explora el modelo que impulsa cada decisión · Análisis del Modelo
    </span>
</div>
<div style='font-family:"Syne",sans-serif; font-size:2.2rem; font-weight:800;
            line-height:1.1; letter-spacing:-0.02em; margin-bottom:0.6rem;'>
    Rendimiento del Modelo
</div>
<div style='font-size:0.95rem; color:#6B7A99; max-width:580px;
            line-height:1.7; margin-bottom:2rem;'>
    Análisis técnico completo del LightGBM Tuned comparado con el Random Forest baseline.
    Todas las métricas se calculan sobre el conjunto de test (20% del dataset, 61,503 registros).
</div>
""", unsafe_allow_html=True)

# ── Carga de datos ───────────────────────────────────────────────────────────
with st.spinner("Cargando modelo y datos de test..."):
    try:
        lgbm    = load_lgbm()
        summary = load_summary()
        shap_s  = load_shap_summary()
        X_test, y_test = prepare_test_data()

        y_prob_lgbm = lgbm.predict_proba(X_test)[:, 1]
        y_pred_lgbm = (y_prob_lgbm >= THRESHOLD).astype(int)

        # RF baseline — intentar cargar, si no disponible usar métricas del JSON
        rf = load_rf()
        if rf is not None:
            # Reindexar para RF si tiene menos features
            rf_feats  = list(rf.feature_names_in_) if hasattr(rf, 'feature_names_in_') else list(X_test.columns[:227])
            X_test_rf = X_test[rf_feats] if all(f in X_test.columns for f in rf_feats) else X_test.iloc[:, :len(rf_feats)]
            X_test_rf.columns = rf_feats
            y_prob_rf = rf.predict_proba(X_test_rf)[:, 1]
            y_pred_rf = (y_prob_rf >= 0.5).astype(int)
            rf_available = True
        else:
            rf_available = False

    except Exception as e:
        st.error(f"Error cargando datos: {e}")
        st.stop()

# ── KPI CARDS ────────────────────────────────────────────────────────────────
fm = summary["final_metrics"]
bi = summary["business_impact"]

col1, col2, col3, col4, col5 = st.columns(5)
metrics_data = [
    ("ROC-AUC",      f"{fm['roc_auc']:.4f}",   "vs. 0.6484 baseline",  "normal"),
    ("Precision",    f"{fm['precision']:.3f}",  "threshold 0.35",       "off"),
    ("Recall",       f"{fm['recall']:.3f}",     "threshold 0.35",       "off"),
    ("F1-Score",     f"{fm['f1_score']:.3f}",   "balance P/R",          "off"),
    ("Default Rate", f"{bi['default_rate']:.2f}%", "aprobados",         "off"),
]
for col, (label, val, delta, dc) in zip([col1,col2,col3,col4,col5], metrics_data):
    with col:
        st.metric(label=label, value=val, delta=delta, delta_color=dc)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: ROC CURVE INTERACTIVA
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:0.3rem;'>ROC Curve</div>
<div style='font-size:0.82rem; color:#6B7A99; margin-bottom:1.2rem;'>
    Área bajo la curva = capacidad discriminativa del modelo.
    La referencia de la industria fintech oscila entre 0.70 y 0.78.
</div>
""", unsafe_allow_html=True)

fpr_lgbm, tpr_lgbm, thresholds_lgbm = roc_curve(y_test, y_prob_lgbm)
roc_auc_lgbm = auc(fpr_lgbm, tpr_lgbm)

fig_roc = go.Figure()

# Banda de referencia industria
fig_roc.add_shape(
    type="rect", x0=0, x1=1, y0=0.70, y1=0.78,
    fillcolor="rgba(245,158,11,0.06)",
    line=dict(width=0),
)
fig_roc.add_annotation(
    x=0.72, y=0.745, text="Rango industria fintech (0.70–0.78)",
    font=dict(size=10, color="#F59E0B"), showarrow=False,
)

# Random baseline
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1],
    mode="lines",
    line=dict(color="#6B7A99", width=1, dash="dash"),
    name="Clasificador aleatorio (AUC=0.50)",
    hoverinfo="skip",
))

# RF baseline
if rf_available:
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    fig_roc.add_trace(go.Scatter(
        x=fpr_rf, y=tpr_rf,
        mode="lines",
        line=dict(color="#6B7A99", width=2),
        name=f"RF Baseline (AUC={roc_auc_rf:.4f})",
        hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>RF Baseline</extra>",
    ))

# LightGBM
fig_roc.add_trace(go.Scatter(
    x=fpr_lgbm, y=tpr_lgbm,
    mode="lines",
    line=dict(color="#00D4AA", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(0,212,170,0.06)",
    name=f"LightGBM Tuned (AUC={roc_auc_lgbm:.4f})",
    hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra>LightGBM</extra>",
))

# Punto threshold actual
idx_thresh = np.argmin(np.abs(thresholds_lgbm - THRESHOLD))
fig_roc.add_trace(go.Scatter(
    x=[fpr_lgbm[idx_thresh]], y=[tpr_lgbm[idx_thresh]],
    mode="markers",
    marker=dict(color="#F59E0B", size=10, symbol="circle",
                line=dict(color="#0A0D16", width=2)),
    name=f"Threshold actual ({THRESHOLD})",
    hovertemplate=f"Threshold={THRESHOLD}<br>FPR=%{{x:.3f}}<br>TPR=%{{y:.3f}}<extra></extra>",
))

fig_roc.update_layout(
    **PLOTLY_LAYOUT,
    width=600, 
    height=320,
    xaxis_title="Tasa de Falsos Positivos (FPR)",
    yaxis_title="Tasa de Verdaderos Positivos (TPR)",
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor="rgba(0,212,170,0.2)",
        borderwidth=1,
        font=dict(size=11),
    ),
)
fig_roc.update_xaxes(range=[0, 1])
fig_roc.update_yaxes(range=[0, 1])

st.plotly_chart(fig_roc, width='stretch')

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: COMPARATIVA BASELINE vs INCLUSIVO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:0.3rem;'>Baseline vs. Modelo Inclusivo</div>
<div style='font-size:0.82rem; color:#6B7A99; margin-bottom:1.2rem;'>
    Comparación directa de métricas entre el Random Forest sin señales alternativas
    y el LightGBM con las 7 señales de inclusión.
</div>
""", unsafe_allow_html=True)

# Datos comparativos
if rf_available:
    rf_prec  = precision_score(y_test, y_pred_rf,   zero_division=0)
    rf_rec   = recall_score(y_test, y_pred_rf,       zero_division=0)
    rf_f1    = f1_score(y_test, y_pred_rf,           zero_division=0)
    rf_auc   = roc_auc_score(y_test, y_prob_rf)
    rf_aprov = int((y_pred_rf == 0).sum())
else:
    # Usar valores de Fase 3
    rf_prec, rf_rec, rf_f1, rf_auc = 0.1658, 0.1255, 0.1432, 0.6484
    rf_aprov = 57745

lgbm_aprov = int((y_pred_lgbm == 0).sum())

metrics_compare = {
    "ROC-AUC":     (rf_auc,   fm['roc_auc'],   True),
    "Precision":   (rf_prec,  fm['precision'],  True),
    "Recall":      (rf_rec,   fm['recall'],     True),
    "F1-Score":    (rf_f1,    fm['f1_score'],   True),
    "Aprobaciones":(rf_aprov, lgbm_aprov,       True),
}

fig_compare = go.Figure()

labels   = list(metrics_compare.keys())
vals_rf  = [metrics_compare[m][0] for m in labels]
vals_lgbm= [metrics_compare[m][1] for m in labels]

# Normalizar aprobaciones a escala 0-1 para visualización conjunta
max_aprov = max(rf_aprov, lgbm_aprov)
vals_rf_norm   = vals_rf[:4]   + [rf_aprov / max_aprov]
vals_lgbm_norm = vals_lgbm[:4] + [lgbm_aprov / max_aprov]

fig_compare.add_trace(go.Bar(
    name="RF Baseline",
    x=labels,
    y=vals_rf_norm,
    marker_color="#6B7A99",
    marker_line_width=0,
    opacity=0.75,
    customdata=vals_rf,
    hovertemplate="%{x}: %{customdata:.4f}<extra>RF Baseline</extra>",
))
fig_compare.add_trace(go.Bar(
    name="LightGBM Inclusivo",
    x=labels,
    y=vals_lgbm_norm,
    marker_color="#00D4AA",
    marker_line_width=0,
    opacity=0.85,
    customdata=vals_lgbm,
    hovertemplate="%{x}: %{customdata:.4f}<extra>LightGBM Inclusivo</extra>",
))

fig_compare.update_layout(
    **PLOTLY_LAYOUT,
    width=600, 
    height=300,
    barmode="group",
    bargap=0.25,
    bargroupgap=0.08,
    yaxis_title="Valor (normalizado para Aprobaciones)",
    legend=dict(
        bgcolor="rgba(17,24,39,0.8)",
        bordercolor="rgba(0,212,170,0.2)",
        borderwidth=1,
    ),
    annotations=[
        dict(
            x="Aprobaciones", y=vals_lgbm_norm[-1] + 0.03,
            text=f"+{lgbm_aprov - rf_aprov:,}",
            font=dict(color="#00D4AA", size=11, family="DM Mono"),
            showarrow=False,
        )
    ]
)

st.plotly_chart(fig_compare, width='stretch')

# Cards de diferencia
col_d1, col_d2, col_d3 = st.columns(3)
delta_auc  = fm['roc_auc'] - rf_auc
delta_aprov= lgbm_aprov - rf_aprov
delta_pct  = delta_aprov / rf_aprov * 100

with col_d1:
    st.markdown(f"""
    <div style='background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.15);
                border-radius:14px; padding:1rem 1.2rem; text-align:center;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.3rem;'>Mejora ROC-AUC</div>
        <div style='font-family:"Syne",sans-serif; font-size:1.6rem; font-weight:800;
                    color:#00D4AA;'>+{delta_auc:.4f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_d2:
    st.markdown(f"""
    <div style='background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.15);
                border-radius:14px; padding:1rem 1.2rem; text-align:center;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.3rem;'>Aprobaciones extra</div>
        <div style='font-family:"Syne",sans-serif; font-size:1.6rem; font-weight:800;
                    color:#00D4AA;'>+{delta_aprov:,}</div>
    </div>
    """, unsafe_allow_html=True)
with col_d3:
    st.markdown(f"""
    <div style='background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.15);
                border-radius:14px; padding:1rem 1.2rem; text-align:center;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.3rem;'>Incremento</div>
        <div style='font-family:"Syne",sans-serif; font-size:1.6rem; font-weight:800;
                    color:#00D4AA;'>+{delta_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: CONFUSION MATRIX
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:0.3rem;'>Matriz de Confusión</div>
<div style='font-size:0.82rem; color:#6B7A99; margin-bottom:1.2rem;'>
    Desglose de predicciones correctas e incorrectas sobre 61,503 solicitantes del conjunto de test.
    Threshold aplicado: 0.35.
</div>
""", unsafe_allow_html=True)

cm = confusion_matrix(y_test, y_pred_lgbm)
tn, fp, fn, tp = cm.ravel()

# Etiquetas con contexto humano
z_text = [
    [f"<b>{tn:,}</b><br>Verdaderos Negativos<br><span style='font-size:10px'>Aprobados correctamente</span>",
     f"<b>{fp:,}</b><br>Falsos Positivos<br><span style='font-size:10px'>Buenos pagadores rechazados</span>"],
    [f"<b>{fn:,}</b><br>Falsos Negativos<br><span style='font-size:10px'>Defaults no detectados</span>",
     f"<b>{tp:,}</b><br>Verdaderos Positivos<br><span style='font-size:10px'>Defaults detectados</span>"],
]

fig_cm = go.Figure(data=go.Heatmap(
    z=[[tn, fp], [fn, tp]],
    x=["Predicho: No-Default", "Predicho: Default"],
    y=["Real: No-Default", "Real: Default"],
    colorscale=[
        [0.0, "#1A2235"],
        [0.3, "#0D3D2E"],
        [1.0, "#00D4AA"],
    ],
    showscale=False,
    hovertemplate="Real: %{y}<br>Predicho: %{x}<br>Cantidad: %{z:,}<extra></extra>",
))

# Anotaciones legibles como texto plano
anotaciones = [
    (0, 0, tn, "Verdaderos Negativos", "Aprobados correct."),
    (1, 0, fp, "Falsos Positivos",     "Deberían estar aprobados"),
    (0, 1, fn, "Falsos Negativos",     "Defaults no detect."),
    (1, 1, tp, "Verdaderos Positivos", "Defaults detect."),
]

annotations = []
for x, y, val, label, sublabel in anotaciones:
    annotations.append(dict(
        x=x, y=y,
        text=f"<b>{val:,}</b><br>{label}<br><i>{sublabel}</i>",
        showarrow=False,
        font=dict(color="#F0F4FF", size=11, family="DM Sans"),
        align="center",
    ))

fig_cm.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(26,34,53,0.6)',
    width=600, 
    height=300,
    margin=dict(l=20, r=20, t=40, b=20),
    font=dict(family="DM Sans", color="#F0F4FF", size=11),
    annotations=annotations,
)
fig_cm.update_xaxes(side="bottom", gridcolor="rgba(0,0,0,0)")
fig_cm.update_yaxes(gridcolor="rgba(0,0,0,0)")

# Resaltar diagonal
fig_cm.add_shape(type="rect", x0=-0.5, x1=0.5, y0=-0.5, y1=0.5,
    line=dict(color="#00D4AA", width=2))
fig_cm.add_shape(type="rect", x0=0.5, x1=1.5, y0=0.5, y1=1.5,
    line=dict(color="#00D4AA", width=2))

col_cm, col_cm_info = st.columns([3, 2], gap="large")

with col_cm:
    st.plotly_chart(fig_cm, width='stretch')

with col_cm_info:
    st.markdown("<br>", unsafe_allow_html=True)
    casos = [
        ("✅", f"{tn:,}", "Aprobados correctamente", "No eran riesgo y el modelo los aprobó.", "#00D4AA"),
        ("🎯", f"{tp:,}", "Defaults detectados",     "Eran riesgo y el modelo los rechazó.", "#F59E0B"),
        ("⚠️", f"{fn:,}", "Defaults no detectados",  "Eran riesgo pero el modelo los aprobó. Costo financiero.", "#FF6B6B"),
        ("💡", f"{fp:,}", "Excluidos injustamente",  "No eran riesgo pero el modelo los rechazó. Costo social.", "#A78BFA"),
    ]
    for emoji, count, label, desc, color in casos:
        st.markdown(f"""
        <div style='display:flex; gap:0.8rem; align-items:flex-start;
                    margin-bottom:0.9rem;'>
            <div style='font-size:1.2rem; margin-top:2px;'>{emoji}</div>
            <div>
                <div style='font-family:"DM Mono",monospace; font-size:0.95rem;
                            font-weight:600; color:{color};'>{count}</div>
                <div style='font-size:0.78rem; color:#F0F4FF; font-weight:500;'>{label}</div>
                <div style='font-size:0.72rem; color:#6B7A99; line-height:1.4;'>{desc}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: FEATURE IMPORTANCE TOP 20
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:0.3rem;'>Feature Importance — Top 20</div>
<div style='font-size:0.82rem; color:#6B7A99; margin-bottom:1.2rem;'>
    Importancia según SHAP. Las señales alternativas están
    destacadas en naranja — el diferenciador del modelo inclusivo.
</div>
""", unsafe_allow_html=True)

try:
    alt_ranking = shap_s.get("alternative_features_ranking", [])
    alt_shap    = {r["Feature"]: r["Mean_Abs_SHAP"] for r in alt_ranking}

    # Feature importance nativa de LightGBM (gain) para los 20 top
    importance_df = pd.DataFrame({
        "Feature":    lgbm.feature_name_,
        "Importance": lgbm.feature_importances_,
    }).sort_values("Importance", ascending=False).head(20)

    # Enriquecer con SHAP si disponible
    importance_df["SHAP"]        = importance_df["Feature"].map(alt_shap).fillna(0)
    importance_df["Is_Alt"]      = importance_df["Feature"].isin(ALTERNATIVE_FEATURES)
    importance_df["Color"]       = importance_df["Is_Alt"].map(
        {True: "#F59E0B", False: "#00D4AA"}
    )
    importance_df["Label"]       = importance_df.apply(
        lambda r: f"⭐ {r['Feature']}" if r["Is_Alt"] else r["Feature"], axis=1
    )
    importance_df = importance_df.sort_values("Importance", ascending=True)

    fig_fi = go.Figure()

    fig_fi.add_trace(go.Bar(
        x=importance_df["Importance"],
        y=importance_df["Label"],
        orientation="h",
        marker_color=importance_df["Color"].tolist(),
        marker_line_width=0,
        opacity=0.85,
        hovertemplate="%{y}<br>Importance: %{x:,}<extra></extra>",
        showlegend=False,
    ))

    fig_fi.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(26,34,53,0.6)',
        margin=dict(l=20, r=20, t=40, b=20),
        font=dict(family='DM Sans', color='#F0F4FF', size=12),
        width=600, 
        height=460,
        xaxis_title="Feature Importance (gain)",
        showlegend=True,
        barmode="overlay",
        legend=dict(
            bgcolor="rgba(17,24,39,0.8)",
            bordercolor="rgba(0,212,170,0.2)",
            borderwidth=1,
        ),
    )
    fig_fi.update_xaxes(gridcolor='rgba(107,122,153,0.15)')
    fig_fi.update_yaxes(gridcolor='rgba(107,122,153,0.15)', tickfont=dict(size=10))

    # Leyenda manual
    fig_fi.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#F59E0B",
        name="⭐ Feature Alternativo (inclusión)",
        hoverinfo="skip",
    ))
    fig_fi.add_trace(go.Bar(
        x=[None], y=[None],
        marker_color="#00D4AA",
        name="Feature Tradicional / Dataset",
        hoverinfo="skip",
    ))

    # Gráfico de Feature Importance
    st.plotly_chart(fig_fi, width='stretch')

    # Resumen de alternativos en top 20
    alt_in_top20 = importance_df[importance_df["Is_Alt"]].shape[0]
    st.markdown(f"""
    <div style='background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.2);
                border-radius:12px; padding:0.9rem 1.2rem; margin-top:0.5rem;'>
        <span style='font-size:0.82rem; color:#F0F4FF;'>
            <strong style='color:#F59E0B;'>{alt_in_top20} de 7</strong>
            señales alternativas aparecen en el Top 20 de features más importantes —
            validando que los datos de inclusión aportan poder predictivo real,
            no solo contexto narrativo.
        </span>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Feature importance error detallado: {e}")
    import traceback
    st.code(traceback.format_exc())

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="height: 80px;"></div>
    """,
    unsafe_allow_html=True
)
render_footer()
