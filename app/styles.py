"""Design system, modern CSS styling, and UI components for NPS Intelligence.
Ultra-Premium Executive Design with Zero Emojis and Cohesive Palette.
"""

# Deep Navy & Electric Sapphire Executive Palette
NAVY_DEEP = "#070E22"
NAVY_SIDEBAR = "#0A1128"
NAVY_SURFACE = "#101D42"
NAVY_CARD_DARK = "#13234F"

PRIMARY_BLUE = "#2563EB"
PRIMARY_BLUE_HOVER = "#1D4ED8"
PRIMARY_BLUE_GLOW = "rgba(37, 99, 235, 0.45)"
CYAN_ACCENT = "#38BDF8"

TEXT_DARK = "#0F172A"
TEXT_MUTED = "#64748B"
BORDER_LIGHT = "#E2E8F0"
CARD_BG = "#FFFFFF"
BG_MAIN = "#F8FAFC"

# Ultra-Premium Semantic NPS Palette
COLOR_DETRACTOR = "#C96F7B"
COLOR_PASSIVE = "#D3A85C"
COLOR_PROMOTER = "#438F7B"
ACCENT_INDIGO = "#6475C8"
ACCENT_TEAL = "#398F91"
ACCENT_VIOLET = "#8171B8"
ACCENT_SUN = "#C99D48"

NPS_COLOR_MAP = {
    "Detractor": COLOR_DETRACTOR,
    "Passive": COLOR_PASSIVE,
    "Promoter": COLOR_PROMOTER,
}

# Ultra-Premium Segment Colors
SEGMENT_COLORS = [ACCENT_INDIGO, ACCENT_TEAL, COLOR_PROMOTER, ACCENT_SUN, ACCENT_VIOLET]


def get_custom_css() -> str:
    return f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        color: {TEXT_DARK};
        background: #F4F6FA !important;
    }}

    .block-container {{
        padding-top: 7.5rem !important;
        padding-bottom: 3rem !important;
        max-width: 1280px !important;
        background: transparent !important;
    }}

    #MainMenu, footer {{ visibility: hidden !important; }}
    header[data-testid="stHeader"] {{
        background: transparent !important;
    }}
    header [data-testid="stToolbar"] {{
        display: none !important;
    }}

    /* -------------------------------------------------------------
       Sidebar: Logo Snapped to Top & Balanced Navigation Menu
       ------------------------------------------------------------- */
    div[data-testid="stSidebarHeader"] {{
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}

    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #09152f 0%, #0c1738 100%) !important;
        min-width: 260px !important;
        max-width: 260px !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 0 0 1px rgba(255,255,255,0.03) inset;
    }}

    section[data-testid="stSidebar"][aria-expanded="false"] {{
        width: 0 !important;
        min-width: 0 !important;
        max-width: 0 !important;
        padding: 0 !important;
        border: 0 !important;
        overflow: hidden !important;
    }}
    
    section[data-testid="stSidebar"] > div:first-child {{
        padding-top: 0 !important;
    }}

    section[data-testid="stSidebarUserContent"] {{
        padding-top: 0.4rem !important;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding: 0.4rem 0.85rem !important;
    }}

    .sidebar-brand-box {{
        padding: 0.6rem 0.4rem 1.1rem 0.4rem;
        margin-top: 0 !important;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.12);
    }}

    .brand-title-wrap {{
        display: flex;
        align-items: center;
        gap: 0.7rem;
    }}

    .brand-logo-icon {{
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, {CYAN_ACCENT} 100%);
        border-radius: 9px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        box-shadow: 0 4px 14px {PRIMARY_BLUE_GLOW};
    }}

    .brand-name {{
        color: #FFFFFF;
        font-size: 1.22rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        line-height: 1.1;
    }}

    .brand-tag {{
        color: #94A3B8;
        font-size: 0.68rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        margin-top: 0.25rem;
    }}

    /* Sidebar Navigation Buttons - Strictly Balanced Dimensions */
    section[data-testid="stSidebar"] .stButton {{
        margin-bottom: 0.45rem !important;
        width: 100% !important;
    }}

    section[data-testid="stSidebar"] .stButton > button {{
        width: 100% !important;
        min-height: 46px !important;
        height: 46px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        border-radius: 9px !important;
        padding: 0 1rem !important;
        font-size: 0.86rem !important;
        font-weight: 600 !important;
        box-sizing: border-box !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
    }}

    /* Inactive items: balanced dark tile */
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {{
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        color: #CBD5E1 !important;
    }}

    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {{
        background: rgba(255, 255, 255, 0.10) !important;
        border-color: rgba(56, 189, 248, 0.35) !important;
        color: #FFFFFF !important;
        transform: translateX(3px) !important;
    }}

    /* Active item: glowing sapphire tile */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, {PRIMARY_BLUE} 0%, #1D4ED8 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 16px {PRIMARY_BLUE_GLOW} !important;
        font-weight: 700 !important;
    }}

    /* -------------------------------------------------------------
       Top Page Header: Sticky, Dark Navy, Prominent (Matches Sidebar)
       ------------------------------------------------------------- */
    .page-hero-banner {{
        position: fixed !important;
        top: 0;
        left: 260px;
        right: 0;
        z-index: 1000;
        background: linear-gradient(135deg, #09142D 0%, #111D3A 100%) !important;
        border: 0 !important;
        border-bottom: 1px solid rgba(108, 132, 192, 0.32) !important;
        border-radius: 0 !important;
        padding: 0.9rem 1.5rem !important;
        margin: 0 !important;
        box-shadow: 0 8px 28px rgba(16, 24, 40, 0.14) !important;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 0.8rem;
        backdrop-filter: blur(12px);
    }}

    body:has(section[data-testid="stSidebar"][aria-expanded="false"]) .page-hero-banner {{
        left: 0;
    }}

    @media (min-width: 981px) {{
        body:has(section[data-testid="stSidebar"][aria-expanded="false"]) .page-hero-banner {{
            padding-left: 4.5rem !important;
        }}
    }}

    body:has(section[data-testid="stSidebar"][aria-expanded="false"]) header [data-testid="stToolbar"] {{
        display: flex !important;
    }}

    body:has(section[data-testid="stSidebar"][aria-expanded="false"]) header [data-testid="stExpandSidebarButton"] {{
        position: fixed !important;
        top: 0.75rem !important;
        left: 0.75rem !important;
        right: auto !important;
        z-index: 1000001 !important;
        display: inline-flex !important;
        width: 2.5rem !important;
        height: 2.5rem !important;
        align-items: center !important;
        justify-content: center !important;
        border-radius: 10px !important;
        background: #FFFFFF !important;
        border: 1px solid #D8DEE9 !important;
        color: #0F172A !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.16) !important;
    }}

    body:has(section[data-testid="stSidebar"][aria-expanded="false"]) header [data-testid="stToolbar"] button[data-testid="stBaseButton-header"] {{
        display: none !important;
    }}

    .hero-title-group {{
        display: flex;
        flex-direction: column;
        gap: 0.2rem;
    }}

    .hero-main-title {{
        font-size: 1.65rem;
        font-weight: 800;
        color: #FFFFFF;
        letter-spacing: -0.025em;
        margin: 0;
        line-height: 1.2;
    }}

    .hero-subtitle {{
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 0.15rem;
    }}

    .hero-badge-pill {{
        background: rgba(37, 99, 235, 0.22);
        border: 1px solid rgba(56, 189, 248, 0.4);
        color: {CYAN_ACCENT};
        padding: 0.45rem 1rem;
        border-radius: 20px;
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.15);
    }}

    /* -------------------------------------------------------------
       KPI Cards: Controlled Heights, Anti-Overflow, Anti-Truncation
       ------------------------------------------------------------- */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}

    .kpi-card {{
        background: rgba(255, 255, 255, 0.72);
        border: 1px solid rgba(99, 102, 241, 0.35);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 126px;
        height: 100%;
        box-sizing: border-box;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }}

    .kpi-card:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
    }}

    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        margin-bottom: 0.25rem;
        line-height: 1.3;
    }}

    .kpi-value {{
        font-size: clamp(1rem, 1.3vw, 1.65rem);
        font-weight: 800;
        color: {TEXT_DARK};
        line-height: 1.2;
        margin: 0.2rem 0;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
        letter-spacing: -0.04em;
    }}

    .kpi-subtext {{
        font-size: 0.74rem;
        color: #64748b;
        line-height: 1.45;
        margin-top: 0.25rem;
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
        font-size: 1.4rem;
        font-weight: 800;
        color: {TEXT_DARK};
        letter-spacing: -0.03em;
        margin-bottom: 0.8rem;
    }}

    .campaign-shell {{
        background: rgba(255, 255, 255, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 18px;
        padding: 1.3rem 1.5rem 1.1rem;
        box-shadow: 0 10px 30px rgba(15, 23, 42, 0.04);
        margin-bottom: 1.5rem;
    }}

    .campaign-shell .section-title {{
        font-size: clamp(1.6rem, 2vw, 2.6rem);
        line-height: 1.1;
        letter-spacing: -0.05em;
        margin: 0 0 0.6rem 0;
    }}

    .campaign-body-copy {{
        font-size: 1.04rem;
        line-height: 1.6;
        color: #475569;
        max-width: 1100px;
        margin: 0;
    }}

    .campaign-kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    }}

    .campaign-kpi-card {{
        background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(148, 163, 184, 0.35);
        border-radius: 14px;
        padding: 0.9rem 1rem;
        min-height: 92px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}

    .campaign-kpi-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_MUTED};
        margin-bottom: 0.25rem;
    }}

    .campaign-kpi-value {{
        font-size: clamp(1.4rem, 2vw, 2rem);
        font-weight: 800;
        color: {TEXT_DARK};
        letter-spacing: -0.04em;
        line-height: 1.1;
    }}

    .campaign-kpi-subtext {{
        font-size: 0.72rem;
        color: {TEXT_MUTED};
        margin-top: 0.3rem;
    }}

    .campaign-filters-header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        margin: 0.6rem 0 0.9rem;
        flex-wrap: wrap;
    }}

    .campaign-filter-label {{
        font-size: 0.92rem;
        font-weight: 700;
        color: {TEXT_DARK};
    }}

    /* Clean Typographic Badges */
    .badge-detractor {{
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.45rem 0.9rem;
        border-radius: 7px;
        font-weight: 700;
        font-size: 0.86rem;
        border-left: 4px solid {COLOR_DETRACTOR};
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    .badge-passive {{
        background: #FEF3C7;
        color: #92400E;
        padding: 0.45rem 0.9rem;
        border-radius: 7px;
        font-weight: 700;
        font-size: 0.86rem;
        border-left: 4px solid {COLOR_PASSIVE};
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    .badge-promoter {{
        background: #D1FAE5;
        color: #065F46;
        padding: 0.45rem 0.9rem;
        border-radius: 7px;
        font-weight: 700;
        font-size: 0.86rem;
        border-left: 4px solid {COLOR_PROMOTER};
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
    }}

    /* Clean Callout Box */
    .callout-box {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid {PRIMARY_BLUE};
        border-radius: 10px;
        padding: 1rem 1.25rem;
        font-size: 0.86rem;
        line-height: 1.55;
        color: #1E3A8A;
        margin: 1.2rem 0;
    }}

    .fairness-donut-title {{
        font-size: 1rem;
        font-weight: 750;
        color: {TEXT_DARK};
        letter-spacing: -0.02em;
        margin: 0.8rem 0 0.15rem;
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
        box-shadow: 0 2px 8px {PRIMARY_BLUE_GLOW} !important;
    }}

    div.stButton > button[kind="primary"]:hover {{
        background: {PRIMARY_BLUE_HOVER} !important;
        border-color: {PRIMARY_BLUE_HOVER} !important;
    }}

    @media (max-width: 980px) {{
        .block-container {{
            padding-top: 8.5rem !important;
            padding-left: 1.25rem !important;
            padding-right: 1.25rem !important;
        }}

        header [data-testid="stToolbar"] {{
            display: flex !important;
            visibility: visible !important;
            background: transparent !important;
        }}

        header [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) {{
            display: none !important;
        }}

        header [data-testid="stExpandSidebarButton"] {{
            display: inline-flex !important;
            border-radius: 10px !important;
            background: #FFFFFF !important;
            border: 1px solid #D8DEE9 !important;
            color: #0F172A !important;
            box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08) !important;
        }}

        .page-hero-banner {{
            left: 0;
            right: 0;
            top: 3.25rem;
            padding: 0.8rem 1rem !important;
            border-radius: 0 !important;
        }}

        .hero-main-title {{
            font-size: 1.35rem !important;
        }}

        .hero-subtitle {{
            display: none;
        }}

        div[data-testid="stSidebarHeader"] {{
            display: flex !important;
            height: auto !important;
            padding: 0.5rem !important;
        }}

        [data-testid="stHorizontalBlock"] {{
            flex-wrap: wrap !important;
            gap: 0.8rem !important;
        }}

        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex: 1 1 calc(50% - 0.8rem) !important;
            min-width: min(100%, 280px) !important;
        }}

        .section-title {{
            font-size: 1.2rem !important;
        }}

        .hero-badge-pill {{
            font-size: 0.66rem;
            padding: 0.35rem 0.65rem;
        }}
    }}

    @media (max-width: 600px) {{
        .block-container {{
            padding-top: 7.5rem !important;
            padding-left: 0.85rem !important;
            padding-right: 0.85rem !important;
        }}

        .page-hero-banner {{
            top: 3rem;
            left: 0;
            right: 0;
            padding: 0.7rem 0.85rem !important;
            gap: 0.4rem;
        }}

        .hero-main-title {{
            font-size: 1.15rem !important;
        }}

        .hero-badge-pill {{
            font-size: 0.58rem;
            padding: 0.28rem 0.55rem;
        }}

        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex-basis: 100% !important;
            min-width: 100% !important;
        }}

        .kpi-card {{
            min-height: 112px;
            padding: 0.85rem 0.95rem;
        }}

        .kpi-value {{
            font-size: clamp(1.15rem, 5vw, 1.5rem);
        }}

        .campaign-shell {{
            padding: 0.9rem;
            border-radius: 14px;
        }}

    }}
    </style>
    """


def render_page_hero(title: str, subtitle: str, badge_text: str = "Production v2.0") -> None:
    """Render the fixed, full-width dark navy page header."""
    import streamlit as st
    st.markdown(f"""
    <div class="page-hero-banner">
        <div class="hero-title-group">
            <div class="hero-main-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        <div class="hero-badge-pill">
            <span style="display:inline-block; width:8px; height:8px; border-radius:50%; background:{CYAN_ACCENT}; box-shadow:0 0 8px {CYAN_ACCENT};"></span>
            {badge_text}
        </div>
    </div>
    """, unsafe_allow_html=True)


def get_plotly_layout_template() -> dict:
    """Return the shared, high-contrast theme for interactive Plotly charts."""
    return dict(
        font=dict(family="Plus Jakarta Sans, sans-serif", color=TEXT_DARK, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(
            bgcolor="#101D42",
            font_size=13,
            font_color="#FFFFFF",
            font_family="Plus Jakarta Sans, sans-serif",
            bordercolor="#6475C8",
        ),
        xaxis=dict(gridcolor="#E5EAF2", zerolinecolor="#CBD5E1", linecolor="#CBD5E1"),
        yaxis=dict(gridcolor="#E5EAF2", zerolinecolor="#CBD5E1", linecolor="#CBD5E1"),
        hovermode="closest",
    )
