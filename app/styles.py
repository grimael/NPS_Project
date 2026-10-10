"""Design system, CSS styling, and UI themes for PulseRetention AI."""

# Palette PulseRetention - Modern Tech Enterprise
NAVY_DEEP = "#0B1528"
NAVY_SURFACE = "#11203C"
ACCENT_BLUE = "#2563EB"
ACCENT_CYAN = "#06B6D4"
TEXT_PRIMARY = "#0F172A"
TEXT_MUTED = "#64748B"
BORDER_COLOR = "#E2E8F0"
BG_LIGHT = "#F8FAFC"
CARD_BG = "#FFFFFF"

# Semantic NPS Colors
COLOR_DETRACTOR = "#EF4444"  # Red-500
COLOR_PASSIVE = "#F59E0B"    # Amber-500
COLOR_PROMOTER = "#10B981"   # Emerald-500

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
    }}

    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        max-width: 1200px;
    }}

    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {NAVY_DEEP} 0%, {NAVY_SURFACE} 100%);
        min-width: 250px !important;
        max-width: 250px !important;
    }}
    
    section[data-testid="stSidebar"] .block-container {{
        padding: 1.2rem 0.8rem !important;
    }}

    .sidebar-brand {{
        padding: 0.5rem 0.6rem 1.2rem 0.6rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }}

    .brand-title {{
        color: #FFFFFF;
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}

    .brand-subtitle {{
        color: #94A3B8;
        font-size: 0.72rem;
        font-weight: 500;
        margin-top: 0.2rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    /* Nav buttons */
    section[data-testid="stSidebar"] .stButton > button {{
        width: 100%;
        text-align: left;
        justify-content: flex-start;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid transparent;
        transition: all 0.15s ease;
        margin-bottom: 0.25rem;
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
        background: {ACCENT_BLUE};
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.35);
    }}

    /* KPI Cards */
    .kpi-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin-bottom: 1.5rem;
    }}

    .kpi-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER_COLOR};
        border-radius: 12px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
    }}

    .kpi-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: {ACCENT_BLUE};
    }}

    .kpi-label {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: {TEXT_MUTED};
        margin-bottom: 0.35rem;
    }}

    .kpi-value {{
        font-size: 1.45rem;
        font-weight: 800;
        color: {TEXT_PRIMARY};
        line-height: 1.2;
    }}

    .kpi-subtext {{
        font-size: 0.75rem;
        color: {TEXT_MUTED};
        margin-top: 0.35rem;
    }}

    /* Section Headers */
    .section-eyebrow {{
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {ACCENT_BLUE};
        margin-bottom: 0.2rem;
    }}

    .section-title {{
        font-size: 1.25rem;
        font-weight: 800;
        color: {TEXT_PRIMARY};
        letter-spacing: -0.02em;
        margin-bottom: 0.8rem;
    }}

    /* Badges */
    .badge-detractor {{
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        border-left: 4px solid {COLOR_DETRACTOR};
        display: inline-block;
    }}

    .badge-passive {{
        background: #FEF3C7;
        color: #92400E;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        border-left: 4px solid {COLOR_PASSIVE};
        display: inline-block;
    }}

    .badge-promoter {{
        background: #D1FAE5;
        color: #065F46;
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        border-left: 4px solid {COLOR_PROMOTER};
        display: inline-block;
    }}

    .callout-box {{
        background: #EFF6FF;
        border: 1px solid #BFDBFE;
        border-left: 4px solid {ACCENT_BLUE};
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        font-size: 0.85rem;
        color: #1E3A8A;
        margin: 1rem 0;
    }}
    </style>
    """
