import streamlit as st

DATA_FILE = "son_analiz.csv"
DB_FILE = "avanza_quant.db"

VARSAYILAN_LISTE = (
    "ELUX-B, ANOD-B, LIME, ELUX-A, MIPS, ARLA, NTEK-B, PRIC-B, BILL, BOOZT, NANO, QLINEA, XBRANE, HOLM-B, "
    "PRFO, RAY-B, BESQ-B, PROFF, CLAS-B, REJL-B, SEZI, SECT-B, BIOA-B, SKIS-B, RROS, CER, YUBICO, HOLM-A, "
    "KABE-B, CARA, ESSITY-A, SOBI, EPEN, ESSITY-B, PREV-B, CEVI, HMS, HEXA-B, SWEC-A, THULE, APOTEA, TROAX, "
    "VIMIAN, NEWA-B, CTT, SVT, BORG, LIFCO-B, SLEEP, SAND, TREL-B, MSAB-B, GARO, MCOV-B, BIOG-B, BONES, "
    "PCELL, SINT, SWEC-B, EPRO-B, HEM, ASSA-B, INWI, VITR, NIL-B, NOLA-B, BEIJ-B, FOI-B, MCAP, SKF-B, "
    "CAMX, G5EN, BEIJ-REF-B, SKF-A, MMGR-B, GETI-B, CCC, BACT-B, CINT, AAK, ALLEI, INVI, DUNI, CTEK, EWRK, "
    "ADDB-B, PION-B, NIBE-B, LAGR-B, NETI-B, HPOL-B, EPI-A, NELLY, FMM-B, CRAD-B, SYST, INDT, NOTE, XANO-B, "
    "ATCO-A, ENGCON-B, EPI-B, ALFA, ATCO-B, BUFAB, AQ, DYNA, MEKE-B, PREC, IRLAB-A, MYCR, DEDI-B, WISE"
)

def set_page_style():
    st.markdown("""
        <style>
        /* Ana konteyner */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; padding-left: 1.2rem; padding-right: 1.2rem; max-width: 1400px; }
        /* Sidebar karanlık tema */
        section[data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f1e 0%, #1a1a2e 100%); }
        section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label { color: #e0e0e0 !important; }
        /* Başlık stilleri */
        h1 { background: linear-gradient(90deg, #00d4ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; letter-spacing: -0.5px; }
        h2, h3 { color: #e8e8e8; font-weight: 600; }
        /* Tab stilleri */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; background: transparent; }
        .stTabs [data-baseweb="tab"] { padding: 10px 20px; border-radius: 10px 10px 0 0; border: 1px solid #333; background: #1a1a2e; color: #aaa; font-weight: 500; font-size: 0.9rem; }
        .stTabs [aria-selected="true"] { background: linear-gradient(135deg, #1e3a5f, #2d5a87); color: #fff !important; border-color: #00d4ff; box-shadow: 0 -2px 8px rgba(0, 212, 255, 0.15); }
        /* Metric kartları */
        [data-testid="stMetric"] { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); padding: 0.4rem 0.6rem; border-radius: 0.5rem; border-left: 2px solid #00d4ff; box-shadow: 0 1px 4px rgba(0,0,0,0.2); }
        [data-testid="stMetric"] label { color: #8ab4f8 !important; font-size: 0.65rem !important; font-weight: 500; text-transform: uppercase; letter-spacing: 0.3px; }
        [data-testid="stMetric"] [data-testid="stMetricValue"] { color: #ffffff !important; font-size: 1rem !important; font-weight: 700; }
        [data-testid="stMetric"] [data-testid="stMetricDelta"] { font-size: 0.7rem !important; }
        div[data-testid="stVerticalBlockBorderWrapper"] { gap: 0.3rem; }
        /* Buton stilleri */
        .stButton > button { border-radius: 10px; font-weight: 600; letter-spacing: 0.3px; transition: all 0.2s; border: none; background: linear-gradient(135deg, #00d4ff, #7b2ff7); }
        .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(0, 212, 255, 0.3); }
        /* Selectbox stilleri */
        [data-baseweb="select"] > div { background: #1a1a2e; border-color: #333; border-radius: 8px; }
        /* Sinyal rozeti */
        .signal-badge { display: inline-block; padding: 2px 10px; border-radius: 6px; font-size: 0.85rem; font-weight: 700; margin-bottom: 4px; }
        .signal-buy { background: #1b4332; color: #52d681; border: 1px solid #52d681; }
        .signal-sell { background: #4a1a1a; color: #ff6b6b; border: 1px solid #ff6b6b; }
        .signal-hold { background: #3d3a0a; color: #ffd93d; border: 1px solid #ffd93d; }
        .stProgress > div > div { background: linear-gradient(90deg, #00d4ff, #7b2ff7); border-radius: 10px; }
        </style>
    """, unsafe_allow_html=True)
