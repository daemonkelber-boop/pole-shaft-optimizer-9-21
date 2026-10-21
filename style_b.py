"""
style_b.py -- Style B (Light Command Center) layer for the Streamlit app.

Colours, fonts and dataframe colours come from .streamlit/config.toml
([theme.light] / [theme.dark]); this module adds what config cannot express:
the navy header band, pill tabs, metric cards, uppercase blue labels, and
matplotlib colours that follow the active light/dark mode.

Mode switch: top-right menu > Settings > Theme (Light / Dark / System).
Streamlit reports the active mode through st.context.theme; charts drawn
with matplotlib pick up a switch on the next rerun (any interaction).
"""
from __future__ import annotations

import streamlit as st

# Style B tokens (tokens.json). Light = sampled, dark = derived.
TOKENS = {
    "light": {
        "bg_page": "#f5f8f9", "bg_surface": "#ffffff", "bg_thead": "#ecf5fb",
        "bg_nav": "#073247", "nav_text": "#c3d3df", "nav_brand": "#ffffff",
        "border": "#cdd6da", "divider": "#e6ebed",
        "text": "#041e2b", "text_secondary": "#2e4a57", "text_muted": "#8399a3",
        "thead_text": "#2b94d0", "primary": "#3d98d3", "on_primary": "#ffffff",
        "success": "#1ea23c", "warning": "#e08d17", "danger": "#ed352c",
        "info": "#3d98d3", "chart_ring": "#073247",
    },
    "dark": {
        "bg_page": "#0b1720", "bg_surface": "#10232e", "bg_thead": "#133247",
        "bg_nav": "#073247", "nav_text": "#c3d3df", "nav_brand": "#ffffff",
        "border": "#23404f", "divider": "#1d3646",
        "text": "#e8f0f4", "text_secondary": "#b8c9d1", "text_muted": "#8fa7b2",
        "thead_text": "#6cb8e6", "primary": "#3d98d3", "on_primary": "#ffffff",
        "success": "#34c058", "warning": "#f0a23a", "danger": "#f2605a",
        "info": "#6cb8e6", "chart_ring": "#6cb8e6",
    },
}


def mode() -> str:
    """'light' or 'dark' -- the mode the viewer's browser is showing."""
    try:
        t = st.context.theme.type
        return t if t in ("light", "dark") else "light"
    except Exception:
        return "light"


def tokens() -> dict:
    return TOKENS[mode()]


def apply(title: str, subtitle: str) -> None:
    """Inject the CSS layer, draw the header band and set matplotlib colours."""
    T = tokens()
    st.markdown(f"""
<style>
/* header band (bg-nav) */
.sb-header {{ background:{T['bg_nav']}; border-radius:6px; padding:14px 24px;
  margin:0 0 16px 0; display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; }}
.sb-header .sb-mark {{ width:14px; height:14px; border-radius:3px; background:{T['primary']};
  display:inline-block; transform:translateY(1px); }}
.sb-header .sb-title {{ color:{T['nav_brand']}; font-size:22px; font-weight:800; line-height:1.2; }}
.sb-header .sb-sub {{ color:{T['nav_text']}; font-size:13px; }}
/* tabs: current tab = blue pill (selectors cover the react-aria tabs of
   current Streamlit and the older baseweb tabs) */
.stTabs [role="tablist"] {{ gap:4px; flex-wrap:wrap; border-bottom:1px solid {T['divider']};
  padding-bottom:6px; }}
.stTabs [role="tab"] {{ height:34px; padding:0 14px; border-radius:999px; display:flex;
  align-items:center; font-weight:500; background:transparent; }}
.stTabs [role="tab"] p {{ color:{T['text_secondary']}; }}
.stTabs [role="tab"][aria-selected="true"] {{ background:{T['primary']}; }}
.stTabs [role="tab"][aria-selected="true"] p {{ color:{T['on_primary']}; font-weight:700; }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {{ display:none; }}
/* KPI / metric cards */
[data-testid="stMetric"] {{ background:{T['bg_surface']}; border:1px solid {T['border']};
  border-radius:6px; padding:12px 16px; box-shadow:0 1px 2px rgba(4,30,43,0.06); }}
[data-testid="stMetricLabel"] p {{ text-transform:uppercase; font-size:12px; font-weight:700;
  letter-spacing:0.02em; color:{T['thead_text']}; }}
[data-testid="stMetricValue"] {{ font-weight:700; font-variant-numeric:tabular-nums; }}
/* forms and expanders read as cards */
[data-testid="stForm"], [data-testid="stExpander"] details {{ background:{T['bg_surface']};
  border:1px solid {T['border']}; border-radius:6px; }}
h1 {{ font-size:28px !important; font-weight:800 !important; }}
h2, h3 {{ font-weight:700 !important; }}
</style>
<div class="sb-header"><span class="sb-mark"></span><span class="sb-title">{title}</span>
<span class="sb-sub">{subtitle}</span></div>
""", unsafe_allow_html=True)
    _mpl(T)


def _mpl(T: dict) -> None:
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "figure.facecolor": T["bg_surface"], "axes.facecolor": T["bg_surface"],
        "savefig.facecolor": T["bg_surface"],
        "axes.edgecolor": T["border"], "axes.labelcolor": T["text"],
        "axes.titlecolor": T["text"], "text.color": T["text"],
        "xtick.color": T["text_secondary"], "ytick.color": T["text_secondary"],
        "grid.color": T["divider"], "legend.facecolor": T["bg_surface"],
        "legend.edgecolor": T["border"], "legend.labelcolor": T["text"],
        "font.family": ["Roboto", "Segoe UI", "DejaVu Sans", "sans-serif"],
        "axes.prop_cycle": cycler(color=[T["primary"], T["warning"], T["success"],
                                         T["danger"], T["text_muted"], T["chart_ring"]]),
    })
