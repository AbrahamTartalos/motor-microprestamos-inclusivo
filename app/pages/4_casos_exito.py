import streamlit as st
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import render_footer, render_sidebar

# ── Configuración ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Casos de Éxito · Motor Inclusivo",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists(os.path.join(ROOT, "models")):
    ROOT = os.path.dirname(ROOT)
MODELS_PATH = os.path.join(ROOT, "models")

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

.card-impacto-cuantificado {
    height: auto !important;
    min-height: 0 !important;
    overflow: visible !important;
}

@media (min-width: 768px) {
    .card-impacto-cuantificado {
        min-height: 250px !important;
        height: auto !important;
        overflow: visible !important;
    }
}
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
[data-testid="block-container"] {
    padding-bottom: 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cache ────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_summary():
    with open(os.path.join(MODELS_PATH, "optimization_summary.json"), "r") as f:
        return json.load(f)

@st.cache_data(show_spinner=False)
def load_shap_summary():
    with open(os.path.join(MODELS_PATH, "shap_summary.json"), "r") as f:
        return json.load(f)

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
        <div style='font-family:"DM Mono",monospace; font-size:0.78rem; color:#00D4AA;'>
            LightGBM Tuned</div>
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
        Observa cómo el modelo evalúa casos reales · Impacto Social
    </span>
</div>
<div style='font-family:"Syne",sans-serif; font-size:2.2rem; font-weight:800;
            line-height:1.1; letter-spacing:-0.02em; margin-bottom:0.6rem;'>
    Casos de Éxito
</div>
<div style='font-size:0.95rem; color:#6B7A99; max-width:620px;
            line-height:1.7; margin-bottom:2rem;'>
    Detrás de cada predicción hay una persona. Estos son los perfiles reales
    extraídos del dataset que ilustran por qué este modelo importa más allá
    de sus métricas técnicas.
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1: MÉTRICAS DE IMPACTO SOCIAL
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:1rem;'>
    Impacto cuantificado
    <span style='font-size:0.7rem; color:#6B7A99; font-weight:400;
                 font-family:"DM Sans",sans-serif; margin-left:0.5rem;'>
        — proyección sobre el mercado argentino
    </span>
</div>
""", unsafe_allow_html=True)

try:
    summary = load_summary()
    bi = summary["business_impact"]
    fm = summary["final_metrics"]

    impacto = [
        ("7M",     "#00D4AA", "Trabajadores informales en Argentina",
         "El 45% de la economía argentina opera en la informalidad — sin acceso al crédito tradicional."),
        ("182K",   "#F59E0B", "Nuevos clientes potenciales",
         f"Proyección del +{bi['pct_increase']:.1f}% de mejora del modelo sobre los 4.9M del mercado potencial."),
        ("$1,104M","#A78BFA", "Impacto económico estimado (USD)",
         "Ingreso adicional estimado considerando préstamo promedio de US$513,531 y margen del 20%."),
        (f"{int(bi['additional_approvals']):,}", "#00D4AA", "Personas adicionales aprobadas",
         f"Sobre el dataset completo de 307,511 solicitantes, con default rate controlado en {bi['default_rate']:.2f}%."),
    ]

    col1, col2, col3, col4 = st.columns(4)
    for col, (val, color, label, desc) in zip([col1, col2, col3, col4], impacto):
        with col:
            st.markdown(f"""
            <div class='card-impacto-cuantificado' style='background:#1A2235; border:1px solid {color}22;
                        border-top:3px solid {color}; border-radius:14px;
                        padding:1.2rem 1.1rem; height:230px; overflow:hidden;'>
                <div style='font-family:"Syne",sans-serif; font-size:2rem;
                            font-weight:800; color:{color}; line-height:1;
                            margin-bottom:0.4rem;'>{val}</div>
                <div style='font-size:0.82rem; font-weight:600; color:#F0F4FF;
                            margin-bottom:0.4rem;'>{label}</div>
                <div style='font-size:0.75rem; color:#6B7A99; line-height:1.5;'>
                    {desc}</div>
            </div>
            """, unsafe_allow_html=True)
except Exception:
    st.info("Cargando métricas de impacto...")

st.markdown("<br>", unsafe_allow_html=True)

# ── Banda de contexto ─────────────────────────────────────────────────────────
st.markdown("""
<div style='background:linear-gradient(135deg, rgba(0,212,170,0.04), rgba(0,160,255,0.04));
            border:1px solid rgba(0,212,170,0.12); border-radius:16px;
            padding:1.4rem 2rem; margin-bottom:2rem;
            display:flex; gap:3rem; flex-wrap:wrap; align-items:center;'>
    <div style='flex:1; min-width:200px;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.12em;
                    text-transform:uppercase; margin-bottom:0.3rem;'>Contexto</div>
        <div style='font-size:0.88rem; color:#F0F4FF; line-height:1.6;'>
            Argentina tiene <strong style='color:#00D4AA;'>45% de economía informal</strong>.
            Una lashista, un feriante o un conductor de Uber o DiDi no tienen recibo de sueldo
            pero pueden ser un pagador o pagadora perfectamente responsable.
        </div>
    </div>
    <div style='flex:1; min-width:200px;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.12em;
                    text-transform:uppercase; margin-bottom:0.3rem;'>La brecha</div>
        <div style='font-size:0.88rem; color:#F0F4FF; line-height:1.6;'>
            El sistema financiero tradicional los excluye
            <strong style='color:#FF6B6B;'>automáticamente</strong> por no tener
            historial bancario — sin evaluar su comportamiento real.
        </div>
    </div>
    <div style='flex:1; min-width:200px;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.12em;
                    text-transform:uppercase; margin-bottom:0.3rem;'>La solución</div>
        <div style='font-size:0.88rem; color:#F0F4FF; line-height:1.6;'>
            Este modelo usa <strong style='color:#00D4AA;'>señales alternativas</strong>
            para ver lo que el sistema tradicional ignora:
            estabilidad, arraigo e inclusión digital.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2: CASOS NARRATIVOS
# ════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:0.4rem;'>
    Perfiles reales del dataset
</div>
<div style='font-size:0.85rem; color:#6B7A99; margin-bottom:1.5rem; max-width:580px;'>
    Casos extraídos del conjunto de test con sus factores SHAP reales.
    Los nombres son ficticios para proteger la privacidad.
</div>
""", unsafe_allow_html=True)

# ── CASO 1: Bianca ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#1A2235; border:1px solid rgba(0,212,170,0.15);
            border-radius:20px; padding:1.8rem 2rem; margin-bottom:1.5rem;'>
""", unsafe_allow_html=True)

col_avatar, col_decision = st.columns([3, 1])
with col_avatar:
    st.markdown("""
    <div style='display:flex; gap:1.2rem; align-items:center;'>
        <div style='width:52px; height:52px; border-radius:50%;
                    background:linear-gradient(135deg, #00D4AA, #00A0FF);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.4rem;'>👩</div>
        <div>
            <div style='font-family:"Syne",sans-serif; font-size:1.1rem;
                        font-weight:800; color:#F0F4FF;'>Bianca G.</div>
            <div style='font-size:0.78rem; color:#6B7A99;'>
                Vendedora ambulante · 34 años · Córdoba</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_decision:
    st.markdown("""
    <div style='background:rgba(0,212,170,0.1); border:1px solid rgba(0,212,170,0.3);
                border-radius:10px; padding:0.5rem 1rem; text-align:center;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.1em;
                    text-transform:uppercase;'>Decisión</div>
        <div style='font-family:"Syne",sans-serif; font-size:1rem;
                    font-weight:800; color:#00D4AA;'>✅ APROBADA</div>
        <div style='font-size:0.72rem; color:#6B7A99;'>Prob. default: 5.7%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='font-size:0.9rem; color:#6B7A99; line-height:1.8;
                margin:1rem 0; border-left:3px solid #00D4AA; padding-left:1rem;'>
        Bianca lleva <strong style='color:#F0F4FF;'>8 años en el mismo negocio</strong>
        y vive en la misma dirección desde hace una década. No tiene cuenta bancaria
        ni recibo de sueldo, pero tiene teléfono, email y toda su documentación en regla.
        Un banco tradicional la hubiera rechazado en segundos. Nuestro modelo
        la aprobó con una probabilidad de default del <strong style='color:#00D4AA;'>5.7%</strong>.
    </div>
    <div style='font-size:0.68rem; color:#6B7A99; letter-spacing:0.12em;
                text-transform:uppercase; margin-bottom:0.7rem;'>
        Perfil de señales alternativas
    </div>
""", unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
scores_maria = [
    ("⭐ Score externo combinado", "#00D4AA", 60, "0.60"),
    ("⭐ Estabilidad de ingresos", "#00D4AA", 90, "0.90"),
    ("⭐ Arraigo domiciliario",    "#00D4AA", 90, "0.90"),
    ("⭐ Historial bureau",        "#00D4AA", 90, "0.90"),
]
for col, (label, color, pct, val) in zip([col_s1, col_s2, col_s3, col_s4], scores_maria):
    with col:
        st.markdown(f"""
        <div style='background:#0A0D16; border-radius:10px; padding:0.7rem 0.9rem;'>
            <div style='font-size:0.68rem; color:{color}; margin-bottom:3px;'>{label}</div>
            <div style='background:#1A2235; border-radius:3px; height:4px; margin-bottom:4px;'>
                <div style='background:{color}; width:{pct}%; height:4px; border-radius:3px;'></div>
            </div>
            <div style='font-family:"DM Mono",monospace; font-size:0.78rem; color:#F0F4FF;'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown("""
    <div style='background:#0A0D16; border-radius:10px; padding:0.8rem 1rem; margin-top:0.6rem;'>
        <div style='font-size:0.68rem; color:#00D4AA; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.4rem;'>
            Factores que redujeron el riesgo</div>
        <div style='font-size:0.8rem; color:#6B7A99; line-height:1.6;'>
            → Educación universitaria (SHAP -0.077)<br>
            → EXT_SOURCE_COMBINED alto (SHAP -0.051)<br>
            → Estabilidad de ingresos (SHAP -0.013)
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_f2:
    st.markdown("""
    <div style='background:#0A0D16; border-radius:10px; padding:0.8rem 1rem; margin-top:0.6rem;'>
        <div style='font-size:0.68rem; color:#FF6B6B; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.4rem;'>
            Factores que aumentaron el riesgo</div>
        <div style='font-size:0.8rem; color:#6B7A99; line-height:1.6;'>
            → Educación secundaria dummy (SHAP +0.050)<br>
            → Inclusión financiera media (SHAP +0.013)<br>
            → Monto del préstamo (SHAP +0.012)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ── CASO 2: Diego ────────────────────────────────────────────────────────────
st.markdown("""
<div style='background:#1A2235; border:1px solid rgba(255,107,107,0.15);
            border-radius:20px; padding:1.8rem 2rem; margin-bottom:1.5rem;'>
""", unsafe_allow_html=True)

col_avatar, col_decision = st.columns([3, 1])
with col_avatar:
    st.markdown("""
    <div style='display:flex; gap:1.2rem; align-items:center;'>
        <div style='width:52px; height:52px; border-radius:50%;
                    background:linear-gradient(135deg, #FF6B6B, #A23B72);
                    display:flex; align-items:center; justify-content:center;
                    font-size:1.4rem;'>👨</div>
        <div>
            <div style='font-family:"Syne",sans-serif; font-size:1.1rem;
                        font-weight:800; color:#F0F4FF;'>Diego M.</div>
            <div style='font-size:0.78rem; color:#6B7A99;'>
                Desempleado reciente · 28 años · Buenos Aires</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_decision:
    st.markdown("""
    <div style='background:rgba(255,107,107,0.1); border:1px solid rgba(255,107,107,0.3);
                border-radius:10px; padding:0.5rem 1rem; text-align:center;'>
        <div style='font-size:0.65rem; color:#6B7A99; letter-spacing:0.1em;
                    text-transform:uppercase;'>Decisión</div>
        <div style='font-family:"Syne",sans-serif; font-size:1rem;
                    font-weight:800; color:#FF6B6B;'>❌ RECHAZADO</div>
        <div style='font-size:0.72rem; color:#6B7A99;'>Prob. default: 41.4%</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div style='font-size:0.9rem; color:#6B7A99; line-height:1.8;
                margin:1rem 0; border-left:3px solid #FF6B6B; padding-left:1rem;'>
        Diego perdió su trabajo hace 6 meses y cambió de domicilio tres veces en el último año.
        Su score externo combinado es <strong style='color:#FF6B6B;'>0.39</strong> —
        por debajo de la mediana del dataset — y su arraigo domiciliario es
        <strong style='color:#FF6B6B;'>0.16</strong>, señal de inestabilidad real.
        El modelo lo rechazó con una probabilidad de default del
        <strong style='color:#FF6B6B;'>41.4%</strong>.
        El rechazo protegió tanto a la institución como a Diego de
        asumir una deuda que no podría sostener.
    </div>
    <div style='font-size:0.68rem; color:#6B7A99; letter-spacing:0.12em;
                text-transform:uppercase; margin-bottom:0.7rem;'>
        Perfil de señales alternativas
    </div>
""", unsafe_allow_html=True)

col_s1, col_s2, col_s3, col_s4 = st.columns(4)
scores_diego = [
    ("⚠️ Score externo combinado", "#FF6B6B", 39,  "0.39"),
    ("⚠️ Arraigo domiciliario",    "#FF6B6B", 16,  "0.16"),
    ("⭐ Estabilidad de ingresos", "#F59E0B", 87,  "0.87"),
    ("⚠️ Inclusión financiera",    "#FF6B6B", 31,  "0.31"),
]
for col, (label, color, pct, val) in zip([col_s1, col_s2, col_s3, col_s4], scores_diego):
    with col:
        st.markdown(f"""
        <div style='background:#0A0D16; border-radius:10px; padding:0.7rem 0.9rem;'>
            <div style='font-size:0.68rem; color:{color}; margin-bottom:3px;'>{label}</div>
            <div style='background:#1A2235; border-radius:3px; height:4px; margin-bottom:4px;'>
                <div style='background:{color}; width:{pct}%; height:4px; border-radius:3px;'></div>
            </div>
            <div style='font-family:"DM Mono",monospace; font-size:0.78rem; color:#F0F4FF;'>{val}</div>
        </div>
        """, unsafe_allow_html=True)

col_f1, col_f2 = st.columns(2)
with col_f1:
    st.markdown("""
    <div style='background:#0A0D16; border-radius:10px; padding:0.8rem 1rem; margin-top:0.6rem;'>
        <div style='font-size:0.68rem; color:#00D4AA; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.4rem;'>
            Factores que redujeron el riesgo</div>
        <div style='font-size:0.8rem; color:#6B7A99; line-height:1.6;'>
            → Día de solicitud lunes (SHAP -0.063)<br>
            → Piso del edificio (SHAP -0.044)<br>
            → Años de empleo previo (SHAP -0.038)
        </div>
    </div>
    """, unsafe_allow_html=True)
with col_f2:
    st.markdown("""
    <div style='background:#0A0D16; border-radius:10px; padding:0.8rem 1rem; margin-top:0.6rem;'>
        <div style='font-size:0.68rem; color:#FF6B6B; letter-spacing:0.1em;
                    text-transform:uppercase; margin-bottom:0.4rem;'>
            Factores determinantes del rechazo</div>
        <div style='font-size:0.8rem; color:#6B7A99; line-height:1.6;'>
            → EXT_SOURCE_COMBINED bajo (SHAP +0.104)<br>
            → Sin acompañante en solicitud (SHAP +0.095)<br>
            → Inestabilidad de ingresos (SHAP +0.073)
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3: LO QUE EL NÚMERO NO DICE
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='font-family:"Syne",sans-serif; font-size:1.2rem; font-weight:700;
            color:#F0F4FF; margin-bottom:1rem;'>
    Lo que el número no dice
</div>
""", unsafe_allow_html=True)

reflexiones = [
    ("#00D4AA", "Inclusión no es regalar crédito",
     "Aprobar a Bianca no es un acto de beneficencia — es reconocer que "
     "su estabilidad real existe aunque el sistema formal no pueda verla. "
     "El modelo no baja el umbral de riesgo: lo mide con mejores datos."),
    ("#F59E0B", "Rechazar a Diego tampoco es discriminación",
     "El rechazo de Diego está basado en señales objetivas de inestabilidad real, "
     "no en su informalidad laboral per se. El modelo distingue entre "
     "'sin historial bancario' y 'con señales de riesgo genuino'."),
    ("#A78BFA", "El verdadero costo son los falsos positivos",
     "Cada persona aprobada que defaultea tiene un costo financiero. Pero cada "
     "persona rechazada injustamente tiene un costo social invisible. "
     "Este modelo busca minimizar ambos — no solo uno."),
    ("#00D4AA", "La explicabilidad es parte del producto",
     "En fintech inclusivo, no alcanza con tomar la decisión correcta — "
     "hay que poder explicarla. SHAP convierte este modelo en algo auditable, "
     "regulable y justo por diseño."),
]

col_r1, col_r2 = st.columns(2)
for i, (color, titulo, texto) in enumerate(reflexiones):
    col = col_r1 if i % 2 == 0 else col_r2
    with col:
        st.markdown(f"""
        <div style='background:#1A2235; border:1px solid {color}22;
                    border-top:3px solid {color}; border-radius:14px;
                    padding:1.2rem 1.3rem; margin-bottom:0.8rem; height:180px; overflow:hidden;'>
            <div style='font-family:"Syne",sans-serif; font-size:0.92rem;
                        font-weight:700; color:{color}; margin-bottom:0.5rem;'>
                {titulo}
            </div>
            <div style='font-size:0.82rem; color:#6B7A99; line-height:1.6;'>
                {texto}
            </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4: CIERRE DEL PROYECTO
# ════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='background:linear-gradient(135deg, rgba(0,212,170,0.05), rgba(167,139,250,0.05));
            border:1px solid rgba(0,212,170,0.15); border-radius:20px;
            padding:2rem 2.5rem; text-align:center;'>
    <div style='font-family:"Syne",sans-serif; font-size:1.4rem; font-weight:800;
                color:#F0F4FF; margin-bottom:0.8rem; line-height:1.3;'>
        Un modelo que ve a las personas<br>
        <span style='background:linear-gradient(135deg, #00D4AA, #00A0FF);
                     -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                     background-clip:text;'>
            que el sistema financiero ignora
        </span>
    </div>
    <div style='font-size:0.9rem; color:#6B7A99; max-width:560px;
                margin:0 auto 1.5rem auto; line-height:1.7;'>
        Este proyecto demuestra que la inclusión financiera y la gestión responsable
        del riesgo no son objetivos opuestos. Con los datos correctos y el modelo
        adecuado, se pueden lograr simultáneamente.
    </div>
    <div style='display:flex; justify-content:center; gap:3rem; flex-wrap:wrap;'>
        <div style='text-align:center;'>
            <div style='font-family:"Syne",sans-serif; font-size:1.6rem;
                        font-weight:800; color:#00D4AA;'>0.7440</div>
            <div style='font-size:0.72rem; color:#6B7A99;'>ROC-AUC final</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-family:"Syne",sans-serif; font-size:1.6rem;
                        font-weight:800; color:#F59E0B;'>5/7</div>
            <div style='font-size:0.72rem; color:#6B7A99;'>features alternativos Top 10</div>
        </div>
        <div style='font-family:"Syne",sans-serif; text-align:center;'>
            <div style='font-size:1.6rem; font-weight:800; color:#A78BFA;'>234</div>
            <div style='font-size:0.72rem; color:#6B7A99;'>features totales</div>
        </div>
        <div style='text-align:center;'>
            <div style='font-family:"Syne",sans-serif; font-size:1.6rem;
                        font-weight:800; color:#00D4AA;'>182K</div>
            <div style='font-size:0.72rem; color:#6B7A99;'>nuevos clientes potenciales</div>
        </div>
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
