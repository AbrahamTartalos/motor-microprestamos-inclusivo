import streamlit as st


def render_footer():
    """Footer único compartido por todas las páginas del dashboard."""
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="height: 80px;"></div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("""
    <div style='border-top: 1px solid rgba(0,212,170,0.12);
                padding: 1rem 0 0.5rem 0;
                display: flex; justify-content: space-between;
                align-items: center; flex-wrap: wrap; gap: 0.5rem;'>
        <div style='font-size:0.72rem; color:#6B7A99;'>
            <span style='color:#00D4AA; font-family:"DM Mono",monospace;'>
                motor-microprestamos-inclusivo
            </span>
            · LightGBM · SHAP · Streamlit
        </div>
        <div style='font-size:0.72rem; color:#6B7A99;'>
            Dataset: Home Credit Default Risk (Kaggle) ·
            <span style='color:#F0F4FF;'>Abraham Tartalos</span> · 2026
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Sidebar consistente compartido por todas las páginas."""
    with st.sidebar:
        st.markdown("""
        <div style='padding:1rem 0 1.5rem 0;'>
            <div style='font-family:"Syne",sans-serif; font-size:1.1rem; font-weight:800;
                        color:#00D4AA; letter-spacing:0.05em;'>MICRO-PRÉSTAMOS</div>
            <div style='font-size:0.72rem; color:#6B7A99; letter-spacing:0.12em;
                        text-transform:uppercase; margin-top:2px;'>
                Motor Inclusivo · v1.0</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""
        <div style='background:rgba(0,212,170,0.06); border:1px solid rgba(0,212,170,0.15);
                    border-radius:12px; padding:0.9rem 1rem;'>
            <div style='font-size:0.68rem; color:#6B7A99; letter-spacing:0.1em;
                        text-transform:uppercase; margin-bottom:0.6rem;'>Modelo activo</div>
            <div style='font-family:"DM Mono",monospace; font-size:0.78rem;
                        color:#00D4AA;'>LightGBM Tuned</div>
            <div style='font-size:0.72rem; color:#F0F4FF; margin-top:4px;'>
                ROC-AUC <span style='color:#00D4AA; font-weight:600;'>0.7440</span></div>
            <div style='font-size:0.72rem; color:#F0F4FF; margin-top:2px;'>
                Threshold <span style='color:#F59E0B; font-weight:600;'>0.35</span></div>
            <div style='font-size:0.72rem; color:#F0F4FF; margin-top:2px;'>
                Features <span style='color:#F0F4FF; font-weight:600;'>234</span>
                <span style='color:#00D4AA;'>(7 alt.)</span></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:0.65rem; color:#6B7A99; text-align:center; line-height:1.6;'>
            Dataset: Home Credit Default Risk<br>
            307,511 solicitantes · Kaggle<br><br>
            <span style='color:#00D4AA;'>Abraham Tartalos</span> · 2026
        </div>
        """, unsafe_allow_html=True)
