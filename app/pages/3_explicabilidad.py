import streamlit as st
import pandas as pd
import json
import os
from PIL import Image
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_footer, render_sidebar

# ── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Explicabilidad · Motor Inclusivo",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Resolución robusta para local y Streamlit Cloud
def find_root():
    """Busca la raíz del proyecto subiendo directorios hasta encontrar models/."""
    path = os.path.abspath(__file__)
    for _ in range(5):  # máximo 5 niveles hacia arriba
        path = os.path.dirname(path)
        if os.path.exists(os.path.join(path, "models")):
            return path
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ROOT        = find_root()
MODELS_PATH = os.path.join(ROOT, "models")
DATA_PATH   = os.path.join(ROOT, "data", "processed")
SHAP_PATH = os.path.join(ROOT, "app", "assets", "shap")

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
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }

/* Tabs */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 8px !important;
    color: var(--text-muted) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
    padding: 0.4rem 1rem !important;
    transition: all 0.2s ease !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: rgba(0,212,170,0.12) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
}
/* Cards de impacto — altura responsiva */
.card-impacto {
    height: 110px;
    overflow: hidden;
}

@media (min-width: 768px) {
    .card-impacto {
        height: 160px;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_shap_summary():
    with open(os.path.join(MODELS_PATH, "shap_summary.json"), "r") as f:
        return json.load(f)


def img_path(filename):
    return os.path.join(SHAP_PATH, filename)

def show_image(filename, caption=None):
    path = img_path(filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.warning(f"Imagen no encontrada: {filename}")

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
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:rgba(245,158,11,0.06); border:1px solid rgba(245,158,11,0.15);
                border-radius:12px; padding:0.9rem 1rem;'>
        <div style='font-size:0.68rem; color:#F59E0B; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.5rem;'>Análisis SHAP</div>
        <div style='font-size:0.72rem; color:#F0F4FF; line-height:1.6;'>
            Muestra: <span style='color:#F59E0B;'>2,000</span> solicitantes<br>
            Background: <span style='color:#F59E0B;'>200</span> filas<br>
            Features: <span style='color:#F59E0B;'>234</span><br>
            Alternativos Top 10: <span style='color:#F59E0B;'>5/7</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-bottom:0.3rem;'>
    <span style='font-size:0.72rem; color:#00D4AA; letter-spacing:0.2em;
                 text-transform:uppercase; font-family:"DM Mono",monospace;'>
        Comprende por qué el modelo toma cada decisión · Explicabilidad
    </span>
</div>
<div style='font-family:"Syne",sans-serif; font-size:2.2rem; font-weight:800;
            line-height:1.1; letter-spacing:-0.02em; margin-bottom:0.6rem;'>
    Explicabilidad SHAP
</div>
<div style='font-size:0.95rem; color:#6B7A99; max-width:620px;
            line-height:1.7; margin-bottom:0.8rem;'>
    SHAP (SHapley Additive exPlanations) mide la contribución de cada variable
    a la predicción. Esta capacidad es crítica en fintech: regulaciones como
    GDPR exigen que los algoritmos puedan explicar sus decisiones de crédito.
</div>
""", unsafe_allow_html=True)

# ── KPI strip desde shap_summary.json ────────────────────────────────────────
try:
    shap_s = load_shap_summary()
    col1, col2, col3, col4 = st.columns(4)
    kpis = [
        ("Top feature global",   shap_s.get("top_feature_overall", "EXT_SOURCE_COMBINED"), "", "#00D4AA"),
        ("Alternativos Top 10",  f"{shap_s.get('alt_features_in_top10', 5)}/7",            "de 7 posibles", "#F59E0B"),
        ("Alternativos Top 20",  f"{shap_s.get('alt_features_in_top20', 5)}/7",            "de 7 posibles", "#A78BFA"),
        ("Muestra analizada",    f"{shap_s.get('shap_sample_size', 2000):,}",               "solicitantes",  "#00D4AA"),
    ]
    for col, (label, val, delta, color) in zip([col1,col2,col3,col4], kpis):
        with col:
            st.markdown(f"""
            <div class='card-impacto' style='background:#1A2235; border:1px solid {color}22;
                        border-left:3px solid {color}; border-radius:14px;
                        padding:1rem 1.2rem; min-height:110px;'>
                <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.1em;
                            text-transform:uppercase; margin-bottom:0.3rem;'>{label}</div>
                <div style='font-family:"Syne",sans-serif; font-size:1.3rem;
                            font-weight:800; color:{color}; line-height:1.2;'>{val}</div>
                <div style='font-size:0.72rem; color:#6B7A99; margin-top:2px;'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)
except Exception:
    pass

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌍  Global",
    "🔍  Features Alternativos",
    "👤  Casos Individuales",
    "⚖️  Baseline vs Inclusivo",
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1: SHAP GLOBAL
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
                color:#F0F4FF; margin:1rem 0 0.4rem 0;'>
        Importancia Global de Features
    </div>
    <div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1.2rem; max-width:600px;'>
        El gráfico de barras muestra el impacto promedio de cada variable sobre
        la probabilidad de default. Las barras naranjas son las señales alternativas
        — el diferenciador del modelo.
    </div>
    """, unsafe_allow_html=True)

    show_image("shap_global_bar.png")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
                color:#F0F4FF; margin-bottom:0.4rem;'>
        Beeswarm Plot — Dirección e Intensidad
    </div>
    <div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1.2rem; max-width:600px;'>
        Cada punto es un solicitante. El color indica el valor de la variable
        (rojo = alto, azul = bajo). La posición horizontal indica si ese valor
        aumenta o reduce la probabilidad de default.
    </div>
    """, unsafe_allow_html=True)

    show_image("shap_beeswarm.png")

    st.markdown("<br>", unsafe_allow_html=True)

    # Interpretación clave
    interpretaciones = [
        ("EXT_SOURCE_COMBINED",       "#00D4AA", "Valor alto → SHAP negativo → menor riesgo. Es el predictor más poderoso del modelo."),
        ("FINANCIAL_INCLUSION_SCORE", "#00D4AA", "Tener móvil, email y documentación completa reduce significativamente el riesgo percibido."),
        ("NAME_EDUCATION_TYPE",       "#6B7A99", "Educación universitaria reduce riesgo. Secundaria lo aumenta levemente — señal tradicional fuerte."),
        ("ADDRESS_TENURE_SCORE",      "#00D4AA", "Vivir muchos años en la misma dirección es una señal robusta de estabilidad."),
        ("INCOME_STABILITY_SCORE_ADJ","#00D4AA", "Empleo estable y de larga data reduce el riesgo incluso sin recibo de sueldo formal."),
    ]

    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1rem; font-weight:700;
                color:#F0F4FF; margin-bottom:0.8rem;'>
        Interpretación de los hallazgos clave
    </div>
    """, unsafe_allow_html=True)

    for feat, color, texto in interpretaciones:
        tag = "⭐ " if color == "#00D4AA" else "   "
        st.markdown(f"""
        <div style='display:flex; gap:1rem; align-items:flex-start;
                    margin-bottom:0.7rem; padding:0.8rem 1rem;
                    background:#1A2235; border-radius:12px;
                    border-left:3px solid {color};'>
            <div style='font-family:"DM Mono",monospace; font-size:0.75rem;
                        color:{color}; min-width:220px; padding-top:2px;'>
                {tag}{feat}
            </div>
            <div style='font-size:0.82rem; color:#6B7A99; line-height:1.5;'>
                {texto}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2: FEATURES ALTERNATIVOS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
                color:#F0F4FF; margin:1rem 0 0.4rem 0;'>
        Contribución de las 7 Señales Alternativas
    </div>
    <div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1.2rem; max-width:600px;'>
        Validación cuantitativa del diferenciador del proyecto: qué tan relevantes
        son las señales de inclusión dentro del modelo completo de 234 features.
    </div>
    """, unsafe_allow_html=True)

    show_image("shap_alternative_features.png")

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabla de ranking desde JSON
    try:
        alt_ranking = shap_s.get("alternative_features_ranking", [])
        if alt_ranking:
            st.markdown("""
            <div style='font-family:"Syne",sans-serif; font-size:1rem; font-weight:700;
                        color:#F0F4FF; margin-bottom:0.8rem;'>
                Ranking SHAP — Features Alternativos (de 234 totales)
            </div>
            """, unsafe_allow_html=True)

            descriptions = {
                "EXT_SOURCE_COMBINED":        "Promedio ponderado de fuentes externas de crédito",
                "ADDRESS_TENURE_SCORE":       "Arraigo domiciliario — años en la misma dirección",
                "INCOME_STABILITY_SCORE_ADJ": "Estabilidad de ingresos por tipo y antigüedad laboral",
                "FINANCIAL_INCLUSION_SCORE":  "Inclusión digital — móvil, email, documentación",
                "CREDIT_BUREAU_SCORE":        "Inversamente proporcional a consultas al bureau",
                "EMPLOYMENT_STABILITY":       "Años de empleo normalizados",
                "PAYMENT_BURDEN_SCORE":       "Carga de pagos respecto al ingreso",
            }

            for r in sorted(alt_ranking, key=lambda x: x["Rank"]):
                feat  = r["Feature"]
                rank  = r["Rank"]
                shap  = r["Mean_Abs_SHAP"]
                desc  = descriptions.get(feat, "")
                bar   = int(shap / 0.04 * 100)  # normalizar sobre max ~0.035
                bar   = min(bar, 100)

                st.markdown(f"""
                <div style='background:#1A2235; border:1px solid rgba(0,212,170,0.1);
                            border-radius:12px; padding:0.9rem 1.1rem; margin-bottom:0.6rem;
                            min-height:110px;'>
                    <div style='display:flex; justify-content:space-between;
                                align-items:center; margin-bottom:0.5rem;'>
                        <div>
                            <span style='font-family:"DM Mono",monospace; font-size:0.7rem;
                                        color:#F59E0B;'>#{rank} global</span>
                            <span style='font-family:"Syne",sans-serif; font-size:0.9rem;
                                        font-weight:700; color:#F0F4FF;
                                        margin-left:0.6rem;'>{feat}</span>
                        </div>
                        <span style='font-family:"DM Mono",monospace; font-size:0.8rem;
                                    color:#00D4AA;'>{shap:.5f}</span>
                    </div>
                    <div style='background:#0A0D16; border-radius:4px; height:5px; margin-bottom:0.4rem;'>
                        <div style='background:#00D4AA; width:{bar}%; height:5px;
                                    border-radius:4px; opacity:0.8;'></div>
                    </div>
                    <div style='font-size:0.75rem; color:#6B7A99;'>{desc}</div>
                </div>
                """, unsafe_allow_html=True)

            # Hallazgos clave del JSON
            findings = shap_s.get("key_findings", [])
            if findings:
                st.markdown("""
                <div style='font-family:"Syne",sans-serif; font-size:1rem; font-weight:700;
                            color:#F0F4FF; margin:1.2rem 0 0.6rem 0;'>
                    Hallazgos clave
                </div>
                """, unsafe_allow_html=True)
                for finding in findings:
                    st.markdown(f"""
                    <div style='display:flex; gap:0.6rem; align-items:flex-start;
                                margin-bottom:0.5rem;'>
                        <span style='color:#00D4AA; font-size:0.9rem;'>→</span>
                        <span style='font-size:0.83rem; color:#6B7A99;
                                     line-height:1.5;'>{finding}</span>
                    </div>
                    """, unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"No se pudo cargar el ranking: {e}")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3: CASOS INDIVIDUALES
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
                color:#F0F4FF; margin:1rem 0 0.4rem 0;'>
        Explicaciones a Nivel Individual
    </div>
    <div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1.2rem; max-width:620px;'>
        SHAP permite explicar la decisión para cada solicitante individual —
        qué factores jugaron a su favor y cuáles en su contra.
        Esta capacidad es exigida por regulaciones de crédito algorítmico.
    </div>
    """, unsafe_allow_html=True)

    # Selector de caso
    casos = {
        "✅ Aprobado correctamente (True Negative)":      "shap_local_tn.png",
        "❌ Rechazado correctamente (True Positive)":     "shap_local_tp.png",
        "⚠️ Rechazado — era buen pagador (False Positive)": "shap_local_fp.png",
    }

    caso_sel = st.selectbox(
        "Seleccioná un tipo de caso para explorar:",
        options=list(casos.keys()),
    )

    descripciones = {
        "✅ Aprobado correctamente (True Negative)": (
            "#00D4AA",
            "El modelo identificó correctamente a un solicitante de bajo riesgo. "
            "Las señales alternativas — score externo combinado y arraigo domiciliario — "
            "compensaron la ausencia de historial bancario formal."
        ),
        "❌ Rechazado correctamente (True Positive)": (
            "#FF6B6B",
            "El modelo detectó un solicitante que efectivamente entró en default. "
            "El EXT_SOURCE_COMBINED bajo y la inestabilidad domiciliaria fueron "
            "las señales más determinantes del rechazo."
        ),
        "⚠️ Rechazado — era buen pagador (False Positive)": (
            "#A78BFA",
            "Caso de exclusión injusta — el solicitante era buen pagador pero fue rechazado. "
            "Ilustra el principal costo social del modelo: personas que el sistema "
            "tradicional excluye por señales insuficientes, no por riesgo real."
        ),
    }

    color_caso, desc_caso = descripciones[caso_sel]

    st.markdown(f"""
    <div style='background:{color_caso}0D; border-left:3px solid {color_caso};
                border-radius:0 12px 12px 0; padding:0.9rem 1.1rem;
                margin-bottom:1rem;'>
        <div style='font-size:0.83rem; color:#F0F4FF; line-height:1.6;'>{desc_caso}</div>
    </div>
    """, unsafe_allow_html=True)

    show_image(casos[caso_sel])

    st.markdown("<br>", unsafe_allow_html=True)

    # Panel de 4 casos comparados
    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
                color:#F0F4FF; margin-bottom:0.4rem;'>
        Comparación de los 4 Tipos de Casos
    </div>
    <div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1rem; max-width:600px;'>
        Vista conjunta de cómo SHAP explica cada tipo de resultado posible
        del modelo — desde aprobaciones correctas hasta casos límite.
    </div>
    """, unsafe_allow_html=True)

    show_image("shap_local_4cases.png")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4: BASELINE VS INCLUSIVO
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:700;
                color:#F0F4FF; margin:1rem 0 0.4rem 0;'>
        Comparación SHAP: RF Baseline vs LightGBM Inclusivo
    </div>
    <div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1.2rem; max-width:620px;'>
        El modelo inclusivo no solo tiene mejor ROC-AUC — toma decisiones
        por razones diferentes. Esta comparación muestra cómo las señales
        alternativas cambian la lógica interna del modelo.
    </div>
    """, unsafe_allow_html=True)

    show_image("shap_baseline_vs_inclusive.png")

    st.markdown("<br>", unsafe_allow_html=True)

    # Insight cards
    insights = [
        ("#00D4AA", "El RF baseline ignora las señales alternativas",
         "Sin EXT_SOURCE_COMBINED ni FINANCIAL_INCLUSION_SCORE como features, "
         "el baseline toma decisiones basadas casi exclusivamente en variables "
         "tradicionales que excluyen a trabajadores informales."),
        ("#F59E0B", "El LightGBM inclusivo reorganiza las prioridades",
         "Al incorporar las 7 señales alternativas, el modelo aprende que "
         "la estabilidad domiciliaria y la inclusión digital son señales "
         "tan predictivas como el historial bancario formal."),
        ("#A78BFA", "La diferencia es estructural, no marginal",
         "No se trata de un ajuste fino de hiperparámetros — los dos modelos "
         "usan información fundamentalmente distinta para tomar decisiones, "
         "lo que se refleja en +4.24% de aprobaciones con default rate controlado."),
    ]

    for color, titulo, texto in insights:
        st.markdown(f"""
        <div style='background:#1A2235; border:1px solid {color}22;
                    border-left:3px solid {color}; border-radius:0 14px 14px 0;
                    padding:1rem 1.2rem; margin-bottom:0.8rem;'>
            <div style='font-family:"Syne",sans-serif; font-size:0.9rem;
                        font-weight:700; color:{color}; margin-bottom:0.4rem;'>
                {titulo}
            </div>
            <div style='font-size:0.82rem; color:#6B7A99; line-height:1.6;'>
                {texto}
            </div>
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
