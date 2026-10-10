"""Design system, modern CSS styling, and UI components for TelcoPulse AI."""

# Cohesive Executive Palette - Royal Blue & Deep Slate (Zero Pink/Magenta)
NAVY_SIDEBAR = "#0A1128"
NAVY_SIDEBAR_SURFACE = "#101D42"
PRIMARY_BLUE = "#2563EB"
PRIMARY_BLUE_HOVER = "#1D4ED8"
PRIMARY_BLUE_LIGHT = "#EFF6FF"
TEXT_DARK = "#0F172A"
TEXT_MUTED = "#64748B"
BORDER_LIGHT = "#E2E8F0"
CARD_BG = "#FFFFFF"
BG_MAIN = "#F8FAFC"

# Semantic NPS Colors
COLOR_DETRACTOR = "#DC2626"  # Red-600
COLOR_PASSIVE = "#D97706"    # Amber-600
COLOR_PROMOTER = "#059669"   # Emerald-600

NPS_COLOR_MAP = {
    "Detractor": COLOR_DETRACTOR,
    "Passive": COLOR_PASSIVE,
    "Promoter": COLOR_PROMOTER,
}


def get_custom_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {TEXT_DARK};
    }}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 3rem;
        max-width: 1240px;
    }}

    #MainMenu, header, footer {{ visibility: hidden; }}

    /* -------------------------------------------------------------
       Sidebar Styling
       ------------------------------------------------------------- */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY_SIDEBAR} 0%, {NAVY_SIDEBAR_SURFACE} 100%) !important;
        min-width: 255px !important;
        max-width: 255px !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    
    section[data-testid="stSidebar"] .block-container {{
        padding: 1.2rem 0.8rem !important;
    }}

    .sidebar-brand-box {{
        padding: 0.6rem 0.6rem 1.4rem 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
        margin-bottom: 1.2rem;
    }}

    .brand-title-wrap {{
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }}

    .brand-logo-icon {{
        width: 32px;
        height: 32px;
        background: {PRIMARY_BLUE};
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.4);
    }}

    .brand-name {{
        color: #FFFFFF;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }}

    .brand-tag {{
        color: #94A3B8;
        font-size: 0.70rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.2rem;
    }}

    /* Sidebar Navigation Buttons */
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        border-radius: 8px;
        padding: 0.65rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid transparent;
        transition: all 0.15s ease;
        margin-bottom: 0.35rem;
        letter-spacing: -0.01em;
    }}

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
        background: transparent;
        color: #CBD5E1;
    }}

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
        background: rgba(255, 255, 255, 0.08);
        color: #FFFFFF;
        border-color: rgba(255, 255, 255, 0.15);
    }}

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: {PRIMARY_BLUE};
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }}

    /* -------------------------------------------------------------
       Top Page Header Banner
       ------------------------------------------------------------- */
    .page-hero-banner {{
        background: #FFFFFF;
        border: 1px solid {BORDER_LIGHT};
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }}

    .hero-title-group {{
        display: flex;
        flex-direction: column;
        gap: 0.25rem;
    }}

    .hero-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {PRIMARY_BLUE};
    }}

    .hero-main-title {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {TEXT_DARK};
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.25;
    }}

    .hero-subtitle {{
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        margin-top: 0.15rem;
    }}

    .hero-badge-pill {{
        background: {PRIMARY_BLUE_LIGHT};
        border: 1px solid #BFDBFE;
        color: #1E40AF;
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* -------------------------------------------------------------
       KPI Cards (Fixed Overflow & Anti-Wrap)
       ------------------------------------------------------------- */
    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_LIGHT};
        border-top: 3.5px solid {PRIMARY_BLUE};
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 4px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 130px;
        height: 100%;
        box-sizing: border-box;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(15, 23, 42, 0.07);
    }}

    .kpi-label {{
        font-size: 0.70rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {TEXT_MUTED};
        margin-bottom: 0.35rem;
        line-height: 1.35;
        word-wrap: break-word;
    }}

    .kpi-value {{
        font-size: clamp(1.25rem, 2vw, 1.6rem);
        font-weight: 800;
        color: {TEXT_DARK};
        line-height: 1.15;
        margin: 0.2rem 0;
        word-break: break-word;
    }}

    .kpi-subtext {{
        font-size: 0.73rem;
        color: {TEXT_MUTED};
        line-height: 1.35;
        margin-top: 0.35rem;
        word-wrap: break-word;
    }}

    /* Section Subheaders */
    .section-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {PRIMARY_BLUE};
        margin-bottom: 0.2rem;
    }}

    .section-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {TEXT_DARK};
        letter-spacing: -0.02em;
        margin-bottom: 0.9rem;
    }}

    /* Clean Badges (Zero tacky emojis) */
    .badge-detractor {{
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.45rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        border-left: 4px solid {COLOR_DETRACTOR};
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    .badge-passive {{
        background: #FEF3C7;
        color: #92400E;
        padding: 0.45rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        border-left: 4px solid {COLOR_PASSIVE};
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    .badge-promoter {{
        background: #D1FAE5;
        color: #065F46;
        padding: 0.45rem 0.9rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.88rem;
        border-left: 4px solid {COLOR_PROMOTER};
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* Clean Informational Callout Box */
    .callout-box {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid {PRIMARY_BLUE};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 0.86rem;
        line-height: 1.5;
        color: #1E3A8A;
        margin: 1.2rem 0;
    }}

    /* Native Tabs & Buttons Styling */
    div[data-testid="stTabs"] button[aria-selected="true"] {{
        color: {PRIMARY_BLUE} !important;
        font-weight: 700 !important;
        border-bottom-color: {PRIMARY_BLUE} !important;
    }}

    div.stButton > button[kind="primary"] {{
        background: {PRIMARY_BLUE} !important;
        border-color: {PRIMARY_BLUE} !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background: {PRIMARY_BLUE_HOVER} !important;
        border-color: {PRIMARY_BLUE_HOVER} !important;
    }}
    </style>
    """


def render_page_hero(title: str, subtitle: str, badge_text: str = "Production v2.0") -> None:
    """Render top hero banner for each page."""
    import streamlit as st
    st.markdown(f"""
    <div class="page-hero-banner">
        <div class="hero-title-group">
            <div class="hero-eyebrow">TelcoPulse AI • Intelligence Client & Rétention</div>
            <h1 class="hero-main-title">{title}</h1>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        <div class="hero-badge-pill">
            <span style="display:inline-block; width:7px; height:7px; border-radius:50%; background:#2563EB;"></span>
            {badge_text}
        </div>
    </div>
    """, unsafe_allow_html=True)
