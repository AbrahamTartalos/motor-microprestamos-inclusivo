import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import os
import re
import time
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_footer, render_sidebar

# ── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Simulador · Motor Inclusivo",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fallback por si __file__ no resuelve correctamente en Streamlit
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

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}
[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: var(--text-muted) !important;
    letter-spacing: 0.03em !important;
}

/* Selectbox */
[data-testid="stSelectbox"] label { font-size:0.82rem !important; color:var(--text-muted) !important; }
[data-testid="stSelectbox"] > div > div {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Number input */
[data-testid="stNumberInput"] label { font-size:0.82rem !important; color:var(--text-muted) !important; }
[data-testid="stNumberInput"] > div > div > input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Botón principal */
.stButton > button {
    background: linear-gradient(135deg, #00D4AA, #00A887) !important;
    color: #0A0D16 !important;
    font-family: var(--font-display) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.7rem 2rem !important;
    letter-spacing: 0.05em !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(0,212,170,0.35) !important;
}

/* Métricas */
[data-testid="stMetric"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.5rem !important;
}
[data-testid="stMetricLabel"] { color:var(--text-muted) !important; font-size:0.75rem !important; letter-spacing:0.08em !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { font-family:var(--font-display) !important; font-size:1.8rem !important; }

hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

/* Footer fijo al fondo */
.footer-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--surface);
    border-top: 1px solid var(--border);
    padding: 0.6rem 2rem;
    z-index: 999;
}

/* Espacio para que el contenido no quede tapado */
[data-testid="block-container"] {
    padding-bottom: 4rem !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
def clean_feature_names(df):
    df = df.copy()
    df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', c) for c in df.columns]
    return df

ALTERNATIVE_FEATURES = [
    'EXT_SOURCE_COMBINED', 'FINANCIAL_INCLUSION_SCORE',
    'INCOME_STABILITY_SCORE_ADJ', 'ADDRESS_TENURE_SCORE',
    'CREDIT_BUREAU_SCORE', 'EMPLOYMENT_STABILITY', 'PAYMENT_BURDEN_SCORE',
]
THRESHOLD = 0.35


# ── Cache: modelo y medianas ─────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    path = os.path.join(MODELS_PATH, "lgbm_tuned_v2.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_medians():
    """Carga medianas del dataset para completar features no visibles."""
    path = os.path.join(DATA_PATH, "train_processed_clean.csv")
    df   = pd.read_csv(path)
    df   = df.drop(columns=["TARGET"], errors="ignore")
    df   = clean_feature_names(df)
    return df.median(numeric_only=True)

@st.cache_resource(show_spinner=False)
def load_explainer(_model):
    return shap.TreeExplainer(_model) # El guión bajo en '_model' le dice a Streamlit que no intente hashear el modelo para el cache


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
                    text-transform:uppercase; margin-bottom:0.6rem;'>Modelo activo</div>
        <div style='font-family:"DM Mono",monospace; font-size:0.78rem; color:#00D4AA;'>LightGBM Tuned</div>
        <div style='font-size:0.72rem; color:#F0F4FF; margin-top:4px;'>
            ROC-AUC <span style='color:#00D4AA; font-weight:600;'>0.7440</span></div>
        <div style='font-size:0.72rem; color:#F0F4FF; margin-top:2px;'>
            Threshold <span style='color:#F59E0B; font-weight:600;'>0.35</span></div>
    </div>
    """, unsafe_allow_html=True)


# ── Header de página ─────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.3rem;'>
    <span style='font-size:0.72rem; color:#00D4AA; letter-spacing:0.2em;
                 text-transform:uppercase; font-family:"DM Mono",monospace;'>
        Simula una solicitud y observa la desición del modelo · Simulador
    </span>
</div>
<div style='font-family:"Syne",sans-serif; font-size:2.2rem; font-weight:800;
            line-height:1.1; letter-spacing:-0.02em; margin-bottom:0.6rem;'>
    Simulador de Solicitud
</div>
<div style='font-size:0.95rem; color:#6B7A99; max-width:560px;
            line-height:1.7; margin-bottom:2rem;'>
    Ingresá el perfil de un solicitante y el modelo evaluará su riesgo crediticio
    en tiempo real, explicando los factores que determinaron la decisión.
</div>
""", unsafe_allow_html=True)


# ── Carga del modelo ─────────────────────────────────────────────────────────
with st.spinner("Cargando modelo..."):
    try:
        model   = load_model()
        medians = load_medians()
        st.success("Modelo listo.", icon="✅")
    except Exception as e:
        st.error(f"Error cargando modelo: {e}")
        st.stop()


# ── FORMULARIO ───────────────────────────────────────────────────────────────
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.05rem; font-weight:700;
            color:#F0F4FF; margin-bottom:1.2rem;'>
    Perfil del solicitante
</div>
""", unsafe_allow_html=True)

col_form, col_result = st.columns([1, 1], gap="large")

with col_form:

    # ── Señales alternativas (protagonistas) ─────────────────────────────────
    st.markdown("""
    <div style='font-size:0.68rem; color:#00D4AA; letter-spacing:0.15em;
                text-transform:uppercase; margin-bottom:0.8rem;
                font-family:"DM Mono",monospace;'>
        ⭐ Señales alternativas de inclusión
    </div>
    """, unsafe_allow_html=True)

    ext_source = st.slider(
        "Score crediticio externo combinado",
        min_value=0.0, max_value=1.0, value=0.51, step=0.01,
        help="Promedio ponderado de 3 fuentes externas. Mayor = menor riesgo."
    )
    financial_inclusion = st.slider(
        "Inclusión financiera (móvil, email, documentación)",
        min_value=0.0, max_value=1.0, value=0.33, step=0.01,
        help="Suma de flags de contacto y documentación / 13. Mayor = más incluido."
    )
    income_stability = st.slider(
        "Estabilidad de ingresos",
        min_value=0.0, max_value=1.0, value=0.88, step=0.01,
        help="Basado en tipo de empleo y años de antigüedad laboral."
    )
    address_tenure = st.slider(
        "Arraigo domiciliario",
        min_value=0.0, max_value=1.0, value=0.68, step=0.01,
        help="Tiempo en la misma dirección normalizado. Mayor = más estable."
    )
    credit_bureau = st.slider(
        "Historial bureau (inverso a consultas)",
        min_value=0.0, max_value=1.0, value=0.79, step=0.01,
        help="Valor alto = pocas consultas al bureau = menor necesidad de crédito urgente."
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Features tradicionales (contexto) ────────────────────────────────────
    st.markdown("""
    <div style='font-size:0.68rem; color:#6B7A99; letter-spacing:0.15em;
                text-transform:uppercase; margin-bottom:0.8rem;
                font-family:"DM Mono",monospace;'>
        Contexto tradicional
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        education = st.selectbox(
            "Nivel educativo",
            options=["Higher education", "Secondary / secondary special",
                     "Incomplete higher", "Lower secondary", "Academic degree"],
            index=1,
        )
        income_type = st.selectbox(
            "Tipo de ingreso",
            options=["Working", "Commercial associate", "Pensioner",
                     "State servant", "Unemployed"],
            index=0,
        )
    with col_t2:
        amt_credit = st.number_input(
            "Monto del crédito (USD)",
            min_value=10_000, max_value=4_000_000,
            value=500_000, step=10_000,
        )
        amt_income = st.number_input(
            "Ingreso anual (USD)",
            min_value=10_000, max_value=1_000_000,
            value=150_000, step=5_000,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("🎯  Evaluar solicitud")


# ── RESULTADO ────────────────────────────────────────────────────────────────
with col_result:

    if not submit:
        # Estado vacío — instrucciones
        st.markdown("""
        <div style='
            height: 100%;
            min-height: 420px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            background: var(--surface2);
            border: 1px dashed rgba(0,212,170,0.2);
            border-radius: 20px;
            padding: 3rem 2rem;
            text-align: center;
        '>
            <div style='font-size:3rem; margin-bottom:1rem;'>🎯</div>
            <div style='font-family:"Syne",sans-serif; font-size:1.1rem;
                        font-weight:700; color:#F0F4FF; margin-bottom:0.5rem;'>
                Listo para evaluar
            </div>
            <div style='font-size:0.85rem; color:#6B7A99; line-height:1.6; max-width:280px;'>
                Completá el perfil del solicitante y presioná
                <strong style='color:#00D4AA;'>Evaluar solicitud</strong>
                para obtener la decisión del modelo con su explicación.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Delay intencional (confianza percibida) ───────────────────────────
        with st.spinner("Analizando perfil..."):
            time.sleep(1.5)

        # ── Construir vector de features ─────────────────────────────────────
        input_data = medians.copy()

        # Features alternativos
        input_data['EXT_SOURCE_COMBINED']        = ext_source
        input_data['FINANCIAL_INCLUSION_SCORE']  = financial_inclusion
        input_data['INCOME_STABILITY_SCORE_ADJ'] = income_stability
        input_data['ADDRESS_TENURE_SCORE']        = address_tenure
        input_data['CREDIT_BUREAU_SCORE']         = credit_bureau

        # Features tradicionales — traducir a columnas del modelo
        edu_cols = {
            "Higher education":               "NAME_EDUCATION_TYPE_Higher_education",
            "Secondary / secondary special":  "NAME_EDUCATION_TYPE_Secondary___secondary_special",
            "Incomplete higher":              "NAME_EDUCATION_TYPE_Incomplete_higher",
            "Lower secondary":                "NAME_EDUCATION_TYPE_Lower_secondary",
            "Academic degree":                "NAME_EDUCATION_TYPE_Academic_degree",
        }
        inc_cols = {
            "Working":              "NAME_INCOME_TYPE_Working",
            "Commercial associate": "NAME_INCOME_TYPE_Commercial_associate",
            "Pensioner":            "NAME_INCOME_TYPE_Pensioner",
            "State servant":        "NAME_INCOME_TYPE_State_servant",
            "Unemployed":           "NAME_INCOME_TYPE_Unemployed",
        }
        # Reset todas las dummies de educación e ingreso
        for col in edu_cols.values():
            if col in input_data.index:
                input_data[col] = 0
        for col in inc_cols.values():
            if col in input_data.index:
                input_data[col] = 0

        # Activar la seleccionada
        edu_col = edu_cols.get(education)
        if edu_col and edu_col in input_data.index:
            input_data[edu_col] = 1

        inc_col = inc_cols.get(income_type)
        if inc_col and inc_col in input_data.index:
            input_data[inc_col] = 1

        # Montos
        if 'AMT_CREDIT' in input_data.index:
            input_data['AMT_CREDIT'] = amt_credit
        if 'AMT_INCOME_TOTAL' in input_data.index:
            input_data['AMT_INCOME_TOTAL'] = amt_income
        if 'CREDIT_INCOME_RATIO' in input_data.index:
            input_data['CREDIT_INCOME_RATIO'] = amt_credit / max(amt_income, 1)

        # Convertir a DataFrame con el orden exacto del modelo
        feature_names = model.feature_name_
        X_input = pd.DataFrame([input_data.reindex(feature_names, fill_value=0)])

        # ── Predicción ───────────────────────────────────────────────────────
        prob     = model.predict_proba(X_input)[0, 1]
        decision = prob < THRESHOLD

        # ── Gauge visual ─────────────────────────────────────────────────────
        color_decision = "#00D4AA" if decision else "#FF6B6B"
        label_decision = "APROBADO" if decision else "RECHAZADO"
        emoji_decision = "✅" if decision else "❌"
        pct            = int(prob * 100)

        # Arco del gauge en SVG
        import math
        angle    = prob * 180
        rad      = math.radians(180 - angle)
        cx, cy, r = 110, 100, 80
        x_end    = cx + r * math.cos(rad)
        y_end    = cy - r * math.sin(rad)
        large    = 1 if angle > 180 else 0

        st.markdown(f"""
        <div style='background:var(--surface2); border:1px solid {color_decision}33;
                    border-radius:20px; padding:1.8rem; text-align:center; margin-bottom:1rem;'>
            <div style='font-size:0.7rem; color:#6B7A99; letter-spacing:0.15em;
                        text-transform:uppercase; margin-bottom:1rem;
                        font-family:"DM Mono",monospace;'>Resultado de evaluación</div>
            <svg width="220" height="120" viewBox="0 0 220 120"
                style="display:block; margin:0 auto 1rem auto;">
                <path d="M 30 100 A 80 80 0 0 1 190 100"
                    fill="none" stroke="#1A2235" stroke-width="14" stroke-linecap="round"/>
                <path d="M 30 100 A 80 80 0 {large} 1 {x_end:.1f} {y_end:.1f}"
                    fill="none" stroke="{color_decision}" stroke-width="14"
                    stroke-linecap="round" opacity="0.9"/>
                <text x="25" y="118" fill="#6B7A99" font-size="10" font-family="DM Mono">0%</text>
                <text x="165" y="118" fill="#6B7A99" font-size="10" font-family="DM Mono">100%</text>
                <text x="110" y="88" fill="{color_decision}" font-size="26" font-weight="800"
                    font-family="Syne" text-anchor="middle">{pct}%</text>
                <text x="110" y="105" fill="#6B7A99" font-size="9"
                    font-family="DM Sans" text-anchor="middle">prob. de default</text>
            </svg>
            <div style='font-family:"Syne",sans-serif; font-size:1.8rem; font-weight:800;
                        color:{color_decision}; letter-spacing:0.05em; margin-bottom:0.3rem;'>
                {emoji_decision} {label_decision}
            </div>
            <div style='font-size:0.82rem; color:#6B7A99;'>
                Probabilidad de default: <strong style='color:{color_decision};'>{prob:.1%}</strong>
                · Threshold: <strong style='color:#F59E0B;'>35%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SHAP local ────────────────────────────────────────────────────────
        try:
            explainer  = load_explainer(model)
            shap_vals  = explainer(X_input)

            if len(shap_vals.values.shape) == 3:
                sv_row = shap_vals[0, :, 1]
            else:
                sv_row = shap_vals[0]

            shap_series = pd.Series(
                sv_row.values,
                index=feature_names
            ).sort_values(key=abs, ascending=False).head(8)

            # Etiquetas en lenguaje humano
            human_labels = {
                'EXT_SOURCE_COMBINED':        '⭐ Score crediticio externo',
                'FINANCIAL_INCLUSION_SCORE':  '⭐ Inclusión financiera',
                'INCOME_STABILITY_SCORE_ADJ': '⭐ Estabilidad de ingresos',
                'ADDRESS_TENURE_SCORE':       '⭐ Arraigo domiciliario',
                'CREDIT_BUREAU_SCORE':        '⭐ Historial bureau',
                'EMPLOYMENT_STABILITY':       '⭐ Estabilidad laboral',
                'PAYMENT_BURDEN_SCORE':       '⭐ Carga de pagos',
                'NAME_EDUCATION_TYPE_Higher_education': 'Educación universitaria',
                'NAME_EDUCATION_TYPE_Secondary___secondary_special': 'Educación secundaria',
                'NAME_INCOME_TYPE_Working':   'Tipo ingreso: empleado',
                'NAME_INCOME_TYPE_Commercial_associate': 'Tipo ingreso: comercial',
                'AMT_CREDIT':                 'Monto del crédito',
                'AMT_INCOME_TOTAL':           'Ingreso anual',
                'CREDIT_INCOME_RATIO':        'Relación crédito/ingreso',
            }

            st.markdown("""
            <div style='font-size:0.7rem; color:#6B7A99; letter-spacing:0.15em;
                        text-transform:uppercase; margin-bottom:0.8rem;
                        font-family:"DM Mono",monospace;'>
                Factores que influyeron en la decisión
            </div>
            """, unsafe_allow_html=True)

            max_abs = shap_series.abs().max()

            for feat, val in shap_series.items():
                label     = human_labels.get(feat, feat)
                bar_pct   = int(abs(val) / max_abs * 100)
                bar_color = "#FF6B6B" if val > 0 else "#00D4AA"
                direction = "↑ aumenta riesgo" if val > 0 else "↓ reduce riesgo"

                st.markdown(f"""
                <div style='margin-bottom:0.6rem;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:3px;'>
                        <span style='font-size:0.78rem; color:#F0F4FF;'>{label}</span>
                        <span style='font-size:0.68rem; color:{bar_color};
                                     font-family:"DM Mono",monospace;'>{direction}</span>
                    </div>
                    <div style='background:#1A2235; border-radius:4px; height:6px;'>
                        <div style='background:{bar_color}; width:{bar_pct}%;
                                    height:6px; border-radius:4px;
                                    opacity:0.85; transition:width 0.3s ease;'>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP no disponible: {e}")

        # ── Nota explicativa ──────────────────────────────────────────────────
        nota_color = "#00D4AA" if decision else "#FF6B6B"
        if decision:
            nota = "El modelo detecta señales de estabilidad que justifican la aprobación. Las señales alternativas compensan la ausencia de historial bancario formal."
        else:
            nota = "El perfil presenta factores de riesgo que superan el umbral de seguridad del modelo. Una mejora en el score externo o la estabilidad domiciliaria podría cambiar la decisión."

        st.markdown(f"""
        <div style='
            background: {nota_color}0D;
            border-left: 3px solid {nota_color};
            border-radius: 0 12px 12px 0;
            padding: 0.9rem 1.1rem;
            margin-top: 0.8rem;
        '>
            <div style='font-size:0.8rem; color:#F0F4FF; line-height:1.6;'>{nota}</div>
        </div>
        """, unsafe_allow_html=True)


# Aviso importante
st.markdown(
    """
    <div style="height: 80px;"></div>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<div style='font-size:0.65rem; color:#6B7A99; text-align:center;'>
    Los valores por defecto de los sliders representan la mediana del dataset.
    Los features no visibles se completan automáticamente con la mediana del conjunto de entrenamiento.
</div>
""", unsafe_allow_html=True)
# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="height: 80px;"></div>
    """,
    unsafe_allow_html=True
)
render_footer()
