"""Design tokens, global CSS and Plotly styling for the Engine Health Monitor dashboard.

Colours: dark surfaces; categorical slots 1-3 and status colours from the validated reference palette
(checked on the #1a1a19 chart surface: lightness, chroma, CVD separation and contrast all pass).
Status colours are reserved for health states and always appear with a text label.
"""
import plotly.graph_objects as go

BG = "#121211"          # page
SURFACE = "#1a1a19"     # cards and chart plot area (validated chart surface)
SURFACE_2 = "#222220"   # raised elements, table header
BORDER = "#2f2f2c"
GRID = "#2a2a27"
TEXT = "#f4f4f1"
TEXT_2 = "#c3c2b7"
MUTED = "#8d8c85"

SERIES = ["#3987e5", "#d95926", "#199e70"]   # categorical slots 1-3 (blue, orange, aqua), fixed order
INFO = SERIES[0]                              # prediction / information
NEUTRAL_LINE = "#9a998f"                      # reference lines (ground truth, scenario)
# Data-driven health states (src/health_state.py). Colours from the validated status palette
# (good / warning / serious / critical) plus a neutral grey; every state is shown with an icon and a text label.
STATUS = {
    "Insufficient History": "#8d8c85",
    "Healthy / Baseline-consistent": "#0ca30c",
    "Degrading": "#fab219",
    "Boundary / Uncertain": "#ec835a",
    "Near Failure": "#d03b3b",
}
STATUS_ICON = {"Insufficient History": "◌", "Healthy / Baseline-consistent": "●", "Degrading": "▲",
               "Boundary / Uncertain": "◆", "Near Failure": "■"}
STATE_SLUG = {"Insufficient History": "insufficient", "Healthy / Baseline-consistent": "healthy",
              "Degrading": "degrading", "Boundary / Uncertain": "boundary", "Near Failure": "nearfailure"}
OFFLINE = "#d03b3b"
ONLINE = "#0ca30c"

FONT = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"
MONO = "'JetBrains Mono', 'Cascadia Mono', Consolas, monospace"

CSS = f"""
<style>
:root {{
  --bg:{BG}; --surface:{SURFACE}; --surface-2:{SURFACE_2}; --border:{BORDER};
  --text:{TEXT}; --text-2:{TEXT_2}; --muted:{MUTED}; --info:{INFO};
  --good:{STATUS['Healthy / Baseline-consistent']}; --warn:{STATUS['Degrading']};
  --serious:{STATUS['Boundary / Uncertain']}; --crit:{STATUS['Near Failure']}; --grey:{STATUS['Insufficient History']};
  --font:{FONT}; --mono:{MONO};
}}
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {{ background: var(--bg); }}
/* custom components only: a global font rule would override Streamlit's icon font (Material Symbols) */
.brand, .sys-status, .ctx, .page-head, .sec, .card, .banner, .note, .flow, .tl-wrap, .range-labels {{ font-family: var(--font); }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }}
[data-testid="stSidebar"] {{ background: #0d0d0c; border-right: 1px solid var(--border); }}
[data-testid="stSidebar"] [role="radiogroup"] label {{
  padding: .45rem .6rem; border-radius: 6px; margin: 1px 0; width: 100%; }}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {{ background: var(--surface); }}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {{
  background: var(--surface-2); box-shadow: inset 2px 0 0 var(--info); }}
h1, h2, h3 {{ letter-spacing: .01em; }}

/* brand + status */
.brand {{ padding: .2rem 0 1rem; border-bottom: 1px solid var(--border); margin-bottom: .8rem; }}
.brand-title {{ font-size: .95rem; font-weight: 700; letter-spacing: .14em; color: var(--text); }}
.brand-sub {{ font-size: .78rem; color: var(--muted); letter-spacing: .04em; margin-top: 2px; }}
.sys-status {{ border-top: 1px solid var(--border); padding-top: .8rem; margin-top: 1.2rem;
  font-size: .8rem; color: var(--text-2); line-height: 1.6; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align: 1px; }}
.ctx {{ font-size: .78rem; color: var(--muted); margin-top: .6rem; line-height: 1.5; }}
.ctx b {{ color: var(--text-2); font-weight: 600; }}

/* page header */
.page-head {{ display:flex; justify-content:space-between; align-items:flex-end; flex-wrap:wrap; gap:.8rem;
  border-bottom: 1px solid var(--border); padding-bottom: .9rem; margin-bottom: 1.2rem; }}
.page-title {{ font-size: 1.45rem; font-weight: 700; letter-spacing: .08em; color: var(--text); }}
.page-sub {{ font-size: .85rem; color: var(--muted); margin-top: 2px; }}
.head-meta {{ display:flex; gap:1.4rem; flex-wrap:wrap; font-size:.78rem; color: var(--muted); }}
.head-meta span b {{ color: var(--text-2); font-weight: 600; margin-left: 4px; }}

/* section header */
.sec {{ margin: 1.6rem 0 .7rem; display:flex; align-items:baseline; gap:.7rem; }}
.sec-title {{ font-size: .78rem; font-weight: 700; letter-spacing: .16em; color: var(--text-2); text-transform: uppercase; }}
.sec-sub {{ font-size: .78rem; color: var(--muted); }}
.help {{ border-bottom: 1px dotted var(--muted); cursor: help; }}

/* cards */
.card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.1rem;
  height: 100%; margin-bottom: .8rem; }}
.kpi {{ min-height: 138px; }}
.kpi-label {{ font-size: .7rem; font-weight: 700; letter-spacing: .14em; color: var(--muted); text-transform: uppercase; }}
.kpi-value {{ font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1.15; margin-top: .35rem;
  font-variant-numeric: tabular-nums; }}
.kpi-unit {{ font-size: .9rem; font-weight: 500; color: var(--text-2); margin-left: .35rem; }}
.kpi-sub {{ font-size: .78rem; color: var(--muted); margin-top: .35rem; line-height: 1.45; }}
.kpi.dominant .kpi-value {{ font-size: 3rem; }}
.kpi.tone-info {{ box-shadow: inset 3px 0 0 var(--info); }}
.kpi.tone-healthy {{ box-shadow: inset 3px 0 0 var(--good); }}
.kpi.tone-degrading {{ box-shadow: inset 3px 0 0 var(--warn); }}
.kpi.tone-boundary {{ box-shadow: inset 3px 0 0 var(--serious); }}
.kpi.tone-nearfailure {{ box-shadow: inset 3px 0 0 var(--crit); }}
.kpi.tone-insufficient {{ box-shadow: inset 3px 0 0 var(--grey); }}

/* health-state history strip and evidence panel */
.strip {{ display:flex; height: 26px; border-radius: 4px; overflow: hidden; gap: 1px; margin-top: .4rem; }}
.strip-seg {{ display:flex; align-items:center; justify-content:center; font-size:.68rem; font-weight:700;
  letter-spacing:.06em; color:#0d0d0c; white-space:nowrap; overflow:hidden; }}
.strip-scale {{ display:flex; justify-content:space-between; font-size:.72rem; color: var(--muted); margin-top:.3rem; }}
.legend {{ display:flex; flex-wrap:wrap; gap: .9rem; font-size:.75rem; color: var(--text-2); margin-top:.5rem; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:-1px; }}
.evid {{ display:grid; grid-template-columns: 1fr 1fr; gap: .8rem; }}
.evid .card {{ margin-bottom: 0; }}
.evid-title {{ font-size:.7rem; font-weight:700; letter-spacing:.14em; color: var(--muted); text-transform: uppercase; }}
.evid-value {{ font-size: 1.05rem; font-weight: 700; color: var(--text); margin: .3rem 0 .2rem; }}
.evid-row {{ font-size: .82rem; color: var(--text-2); line-height: 1.55; }}
.flag {{ border: 1px solid var(--serious); color: var(--text); background: rgba(236,131,90,.08); border-radius: 6px;
  padding: .55rem .85rem; font-size: .85rem; margin: .6rem 0; }}
.flag b {{ color: var(--serious); }}
.range-boundary {{ position:absolute; top:2px; width:0; height:42px; border-left: 2px dashed var(--crit); }}
.kpi.sim {{ border-style: dashed; }}

.badge {{ display:inline-flex; align-items:center; gap:.4rem; padding: .18rem .6rem; border-radius: 4px;
  font-size: .78rem; font-weight: 700; letter-spacing: .1em; text-transform: uppercase;
  border: 1px solid currentColor; }}
.badge.lg {{ font-size: 1.15rem; padding: .3rem .8rem; margin-top: .45rem; }}

/* prediction card */
.pred {{ display:flex; gap:2rem; align-items:center; flex-wrap:wrap; }}
.pred-num {{ font-size: 4.6rem; font-weight: 700; line-height: 1; color: var(--text); font-variant-numeric: tabular-nums; }}
.pred-unit {{ font-size: .8rem; letter-spacing: .2em; color: var(--muted); margin-top: .3rem; }}
.pred-side {{ display:flex; flex-direction:column; gap:.55rem; }}
.pred-row {{ font-size: .85rem; color: var(--text-2); }}
.pred-row b {{ color: var(--text); font-size: 1rem; }}

/* uncertainty range + health timeline */
.range {{ position: relative; height: 46px; margin: .6rem 0 .2rem; }}
.range-axis {{ position:absolute; left:0; right:0; top:22px; height:2px; background: var(--border); }}
.range-band {{ position:absolute; top:14px; height:18px; background: rgba(57,135,229,.28);
  border-left: 2px solid var(--info); border-right: 2px solid var(--info); }}
.range-mark {{ position:absolute; top:8px; width:3px; height:30px; background: var(--text); transform: translateX(-1px); }}
.range-labels {{ display:flex; justify-content:space-between; font-size:.78rem; color: var(--muted);
  font-variant-numeric: tabular-nums; }}
.range-labels b {{ color: var(--text); }}
.timeline {{ display:flex; height: 34px; border-radius: 4px; overflow: hidden; gap: 2px; margin-top: .5rem; }}
.tl-seg {{ display:flex; align-items:center; justify-content:center; font-size:.72rem; font-weight:700;
  letter-spacing:.12em; color:#0d0d0c; }}
.tl-wrap {{ position: relative; padding-top: 30px; }}
.tl-marker {{ position:absolute; top:0; transform: translateX(-50%); font-size:.75rem; color: var(--text);
  white-space: nowrap; text-align:center; }}
.tl-marker::after {{ content:""; display:block; width:2px; height:40px; background: var(--text); margin: 2px auto 0; }}
.tl-scale {{ position: relative; height: 1.2rem; font-size:.72rem; color: var(--muted); margin-top:.35rem; }}
.tl-scale span {{ position:absolute; transform: translateX(-50%); white-space: nowrap; }}
.tl-scale span:first-child {{ transform: none; }}
.tl-scale span:last-child {{ transform: translateX(-100%); }}

/* sensor list */
.sensor-row {{ display:flex; justify-content:space-between; align-items:center; padding:.5rem 0;
  border-bottom: 1px solid var(--border); font-size: .88rem; }}
.sensor-row:last-child {{ border-bottom: none; }}
.sensor-name {{ color: var(--text); font-family: var(--mono); font-size: .82rem; }}
.sensor-trend {{ color: var(--text-2); font-variant-numeric: tabular-nums; }}
.sensor-trend .arrow {{ font-weight: 700; margin-right: .35rem; color: var(--text); }}

/* banners and notes */
.banner {{ border: 1px dashed var(--warn); color: var(--text); background: rgba(250,178,25,.07);
  border-radius: 6px; padding: .6rem .9rem; font-size: .85rem; margin-bottom: 1rem; }}
.banner b {{ color: var(--warn); letter-spacing: .12em; }}
.note {{ border-left: 2px solid var(--info); background: var(--surface); padding: .7rem .9rem;
  font-size: .85rem; color: var(--text-2); border-radius: 0 6px 6px 0; margin: .6rem 0; }}
.flow {{ display:flex; flex-direction:column; gap:.35rem; align-items:flex-start; font-size:.88rem; }}
.flow .step {{ background: var(--surface); border: 1px solid var(--border); border-radius: 6px;
  padding: .45rem .8rem; color: var(--text); min-width: 280px; }}
.flow .arrow {{ color: var(--muted); padding-left: 1rem; }}

/* HTML tables (table_view.py) on dark surfaces */
.rul-table table {{ color: var(--text-2); }}
.rul-table th {{ background: var(--surface-2) !important; color: var(--text); }}
.rul-table td, .rul-table th {{ border-bottom: 1px solid var(--border) !important; }}
</style>
"""


def style_fig(fig: go.Figure, height: int = 360, x_title: str | None = None, y_title: str | None = None,
              title: str | None = None, legend: bool = True) -> go.Figure:
    """Recessive axes and grid, dark surface, legend on top, unified hover."""
    fig.update_layout(
        height=height, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family=FONT, color=TEXT_2, size=12),
        title=dict(text=title, font=dict(size=13, color=TEXT), x=0.01, y=0.97) if title else None,
        margin=dict(l=56, r=20, t=48 if title else 28, b=48),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",
                    font=dict(color=TEXT_2, size=11)),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER, font=dict(color=TEXT, family=FONT)),
    )
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=BORDER, tickcolor=BORDER, color=MUTED,
                title_font=dict(color=TEXT_2, size=12))
    fig.update_xaxes(title_text=x_title, **axis)
    fig.update_yaxes(title_text=y_title, **axis)
    return fig
