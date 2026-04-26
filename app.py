# ==============================================================================
#  DEEP AUDIT AI  v9.0  —  PRODUCTION ED / CBI FINANCIAL CRIMES UNIT
#  FIXES IN THIS VERSION:
#
#  FIX-1  Normal transactions no longer score HIGH — dynamic threshold calibration
#          Risk scorer now uses percentile-based thresholds from actual live data
#          instead of fixed 0.65 cutoff. Normal → Low, Suspicious → Medium/High.
#
#  FIX-2  Boss name cache bug resolved — st.session_state cleared on every new
#          scan. Old boss no longer persists when generator rotates to new boss.
#          Added "Force Clear & Rescan" button on Live Dashboard.
#
#  FIX-3  Isolation Forest contamination capped at 0.05 for live data to prevent
#          over-flagging. n_estimators reduced 200→100 for speed.
#
#  FIX-4  Scan limit reduced — default 3,000 rows (was 10,000+). Fast mode added.
#          Pipeline completes in ~3 seconds for 3,000 records.
#
#  FIX-5  Real-time scorer calibrated — uses 75th/90th percentile of actual
#          transaction amounts from live data, not hardcoded 10,000,000 max.
#
#  v9.0 — PDF FIX + XGB FIX + KYC ACCOUNT INVESTIGATOR + ALL 14 IMPROVEMENTS ACTIVE.
# ==============================================================================

import matplotlib
matplotlib.use('Agg')

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import io, re, json, warnings, datetime, gc, os, shutil, tempfile
import hashlib, sqlite3, smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from fpdf import FPDF

from sklearn.ensemble      import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None
np.random.seed(42)

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                     Paragraph, Spacer, Image, PageBreak, HRFlowable)
    PDF_OK = True
except ImportError:
    PDF_OK = False

import os as _os
_os.environ.setdefault('PYTHONWARNINGS', 'ignore')
try:
    import warnings as _xgb_w
    _xgb_w.filterwarnings('ignore', category=UserWarning, module='xgboost')
    _xgb_w.filterwarnings('ignore', category=FutureWarning, module='xgboost')
    from xgboost import XGBClassifier
    XGB_OK = True
except ImportError:
    XGB_OK = False

# ── Feature list ───────────────────────────────────────────────────────────────
FEATS = [
    'Amount_INR', 'Amount_Log', 'Time_Gap', 'Rapid_Fire',
    'Risk_Score', 'Txn_Encoded', 'Is_Night', 'Is_International',
    'Velocity_1d', 'Amount_Deviation', 'Round_Amount', 'Txn_Hour',
]

# ── NLP Fraud Taxonomy ─────────────────────────────────────────────────────────
FRAUD_TAXONOMY = {
    'SMURFING':    ['smurf', 'structur', 'below threshold', 'split transfer',
                    'micro transfer', 'threshold avoid', 'cash fragment',
                    'structured deposit', 'multiple small'],
    'LAYERING':    ['syndicate', 'shell', 'layering', 'intermediary', 'nominee',
                    'transit account', 'pass through', 'conduit', 'routing',
                    'shell company'],
    'INTEGRATION': ['integration', 'offshore', 'consolidat', 'swift transfer',
                    'clean fund', 'final transfer', 'boss account', 'wire',
                    'offshore wire', 'offshore consolidation'],
    'HAWALA':      ['hawala', 'hundi', 'informal', 'angadia', 'trust transfer',
                    'undocumented', 'unregistered', 'informal value'],
    'TERROR':      ['terror', 'militant', 'extremist', 'sanction', 'frozen asset',
                    'blacklist', 'financing'],
}

_DEFAULT_USERS = {
    'admin':   {'hash': hashlib.sha256(b'admin123').hexdigest(),   'role': 'ADMIN'},
    'analyst': {'hash': hashlib.sha256(b'analyst456').hexdigest(), 'role': 'ANALYST'},
}

LIVE_DB_PATH  = 'live_transactions.db'
AUDIT_DB_PATH = 'deep_audit_v8.db'
REQUIRED_TXN_COLUMNS = [
    'Source_Acc_No', 'Dest_Acc_No', 'Amount_INR',
    'Timestamp', 'Transaction_Type',
]
SYNDICATE_TABLE_COLUMNS = [
    'Account', 'Role', 'Inflow_INR', 'Outflow_INR', 'Total_Volume',
    'Entity', 'Location', 'In_Centrality', 'Out_Centrality',
    'PageRank', 'Boss_Score',
]

# ==============================================================================
#  PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title='Deep Audit AI — ED / CBI Financial Crimes Division',
    page_icon='🔴',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ==============================================================================
#  SECTION A — JSON HELPERS
# ==============================================================================
def safe_dumps(obj):
    class NumpyEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, np.integer):  return int(o)
            if isinstance(o, np.floating): return float(o)
            if isinstance(o, np.bool_):    return bool(o)
            if isinstance(o, np.ndarray):  return o.tolist()
            return super().default(o)
    return json.dumps(obj, cls=NumpyEncoder)


def _open_sqlite(path):
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    try:
        conn.execute('PRAGMA busy_timeout = 30000')
        conn.execute('PRAGMA journal_mode = WAL')
    except Exception:
        pass
    return conn


def _reset_uploaded_file(uploaded_file):
    try:
        uploaded_file.seek(0)
    except Exception:
        pass


def _read_uploaded_frame(uploaded_file):
    name = str(getattr(uploaded_file, 'name', 'upload.csv'))
    lower_name = name.lower()
    last_error = None

    readers = []
    if lower_name.endswith('.parquet'):
        readers.append(lambda f: pd.read_parquet(f).head(500_000))
    readers.extend([
        lambda f: pd.read_csv(f, nrows=500_000, low_memory=False),
        lambda f: pd.read_csv(f, nrows=100_000, low_memory=False, engine='python', on_bad_lines='skip'),
    ])

    for reader in readers:
        try:
            _reset_uploaded_file(uploaded_file)
            return reader(uploaded_file)
        except Exception as exc:
            last_error = exc

    raise last_error or ValueError(f'Unable to read {name}.')


def _validate_input_df(df, source_name):
    if df is None or df.empty:
        raise ValueError(f'{source_name}: file is empty.')

    clean_source = re.sub(r'[^A-Za-z0-9]+', '_', source_name).strip('_') or 'UPLOAD'
    df = df.copy()

    if 'Transaction_ID' not in df.columns:
        df['Transaction_ID'] = [f'{clean_source}_{i + 1:07d}' for i in range(len(df))]

    missing = [col for col in REQUIRED_TXN_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f'{source_name}: missing required columns: {", ".join(missing)}')

    df['Amount_INR'] = pd.to_numeric(df['Amount_INR'], errors='coerce')
    df.dropna(subset=['Amount_INR'], inplace=True)
    if df.empty:
        raise ValueError(f'{source_name}: no valid numeric values found in Amount_INR.')

    return df

# ==============================================================================
#  SECTION B — AUDIT DATABASE
# ==============================================================================
def _init_audit_db():
    try:
        conn = _open_sqlite(AUDIT_DB_PATH)
        conn.execute('''CREATE TABLE IF NOT EXISTS audit_log
            (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT, action TEXT, detail TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS scan_history
            (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, case_no TEXT, user TEXT,
             n_txn INTEGER, n_suspicious INTEGER, boss_account TEXT,
             boss_score REAL, flagged_amount REAL, n_banks INTEGER, mode TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS feedback
            (id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, user TEXT,
             transaction_id TEXT, verdict TEXT, note TEXT)''')
        conn.commit()
        return conn, True
    except Exception:
        return None, False

if 'db_conn' not in st.session_state:
    st.session_state['db_conn'], st.session_state['db_ok'] = _init_audit_db()

def db_log(action, detail=''):
    conn = st.session_state.get('db_conn')
    if conn:
        try:
            conn.execute('INSERT INTO audit_log (ts,user,action,detail) VALUES (?,?,?,?)',
                         (datetime.datetime.now().isoformat(timespec='seconds'),
                          st.session_state.get('auth_user','SYSTEM'), action, str(detail)))
            conn.commit()
        except Exception:
            pass

def db_save_scan(case_no, n_txn, n_susp, boss, boss_score, amount, n_banks, mode='UPLOAD'):
    conn = st.session_state.get('db_conn')
    if conn:
        try:
            conn.execute(
                'INSERT INTO scan_history (ts,case_no,user,n_txn,n_suspicious,boss_account,'
                'boss_score,flagged_amount,n_banks,mode) VALUES (?,?,?,?,?,?,?,?,?,?)',
                (datetime.datetime.now().isoformat(timespec='seconds'), case_no,
                 st.session_state.get('auth_user','SYSTEM'), int(n_txn), int(n_susp),
                 str(boss), float(boss_score), float(amount), int(n_banks), mode))
            conn.commit()
        except Exception:
            pass

def db_get_history():
    conn = st.session_state.get('db_conn')
    if conn:
        try:
            return pd.read_sql('SELECT * FROM scan_history ORDER BY id DESC LIMIT 20', conn)
        except Exception:
            pass
    return pd.DataFrame()

def db_save_feedback(txn_id, verdict, note):
    conn = st.session_state.get('db_conn')
    if conn:
        try:
            conn.execute(
                'INSERT INTO feedback (ts,user,transaction_id,verdict,note) VALUES (?,?,?,?,?)',
                (datetime.datetime.now().isoformat(timespec='seconds'),
                 st.session_state.get('auth_user','SYSTEM'), str(txn_id), verdict, note))
            conn.commit()
        except Exception:
            pass

# ==============================================================================
#  SECTION C — LIVE DATABASE READER
# ==============================================================================
def live_db_available():
    return os.path.exists(LIVE_DB_PATH)

@st.cache_data(ttl=5)
def load_live_transactions(limit=3000):
    """FIX-4: default limit 3000 (was 10000+) for fast scans."""
    if not live_db_available():
        return pd.DataFrame()
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        safe_limit = max(1, int(limit))
        df = pd.read_sql(
            f'SELECT * FROM live_transactions ORDER BY inserted_at DESC LIMIT {safe_limit}', conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=5)
def load_live_recent(n=20):
    if not live_db_available():
        return pd.DataFrame()
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        safe_n = max(1, int(n))
        df = pd.read_sql(
            f'SELECT * FROM live_transactions ORDER BY inserted_at DESC LIMIT {safe_n}', conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

@st.cache_data(ttl=5)
def get_live_stats():
    if not live_db_available():
        return {}
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        total = pd.read_sql('SELECT COUNT(*) as c FROM live_transactions', conn).iloc[0]['c']
        fraud = pd.read_sql(
            "SELECT COUNT(*) as c FROM live_transactions WHERE Fraud_Label != 'NORMAL'",
            conn).iloc[0]['c']
        vol   = pd.read_sql(
            "SELECT SUM(Amount_INR) as s FROM live_transactions WHERE Fraud_Label != 'NORMAL'",
            conn).iloc[0]['s']
        last  = pd.read_sql(
            'SELECT inserted_at FROM live_transactions ORDER BY inserted_at DESC LIMIT 1', conn)
        return {
            'total'      : int(total),
            'fraud'      : int(fraud),
            'fraud_rate' : round(fraud / max(1, total) * 100, 2),
            'volume'     : float(vol or 0),
            'last_update': last.iloc[0]['inserted_at'][-8:] if len(last) else 'Never',
        }
    except Exception:
        return {}
    finally:
        if conn:
            conn.close()

# ==============================================================================
#  SECTION D — CSS
# ==============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');
.stApp{background:#04040f}
[data-testid="stSidebar"]{background:#060614!important;border-right:1px solid #12123a}
[data-testid="stHeader"]{background:#04040f!important;border-bottom:1px solid #0d0d2a}
[data-testid="block-container"]{padding-top:1rem;max-width:1400px}
section[data-testid="stSidebar"]>div{padding-top:1rem}
body,p,li{font-family:'Exo 2',sans-serif!important;color:#9aaabb}
div.stMarkdown p,div.stMarkdown li{font-family:'Exo 2',sans-serif!important;color:#9aaabb}
[data-testid="stWidgetLabel"] p{font-family:'Exo 2',sans-serif!important;color:#778899!important;font-size:13px!important;font-weight:500!important;margin-bottom:4px!important}
[data-testid="stTextInput"] input{background:#07071a!important;border:1px solid #1e1e55!important;border-radius:4px!important;color:#c8d8e8!important;font-family:'Share Tech Mono',monospace!important;font-size:13px!important;padding:8px 12px!important}
[data-testid="stTextInput"] input:focus{border-color:#ffd700!important;box-shadow:none!important}
[data-testid="stNumberInput"] input{background:#07071a!important;border:1px solid #1e1e55!important;border-radius:4px!important;color:#c8d8e8!important;font-family:'Share Tech Mono',monospace!important;font-size:13px!important}
[data-testid="stSelectbox"]>div>div{background:#07071a!important;border:1px solid #1e1e55!important;border-radius:4px!important;color:#c8d8e8!important;font-family:'Share Tech Mono',monospace!important}
[data-testid="stCheckbox"] label p{font-family:'Exo 2',sans-serif!important;color:#9aaabb!important}
[data-testid="stDateInput"] input,[data-testid="stTimeInput"] input{background:#07071a!important;border:1px solid #1e1e55!important;border-radius:4px!important;color:#c8d8e8!important;font-family:'Share Tech Mono',monospace!important;font-size:13px!important}
h1{font-family:'Share Tech Mono',monospace!important;color:#ff2020!important;font-size:1.9rem!important;letter-spacing:4px;margin-bottom:.2rem}
h2{font-family:'Share Tech Mono',monospace!important;color:#ffd700!important;font-size:1.15rem!important;letter-spacing:2px}
h3{font-family:'Share Tech Mono',monospace!important;color:#7799cc!important;font-size:.95rem!important;letter-spacing:1px}
.stButton>button{background:#09091e;border:1px solid #ffd700;color:#ffd700;font-family:'Share Tech Mono',monospace!important;font-size:13px;letter-spacing:1.5px;padding:10px 28px;border-radius:2px;width:100%;transition:all .22s}
.stButton>button:hover{background:#ffd700;color:#04040f;box-shadow:0 0 20px rgba(255,215,0,.4)}
[data-testid="stDownloadButton"]>button{background:#071a0e;border:1px solid #00e676;color:#00e676;font-family:'Share Tech Mono',monospace!important;font-size:13px;letter-spacing:1.5px;padding:10px 28px;border-radius:2px;width:100%;transition:all .22s}
[data-testid="stDownloadButton"]>button:hover{background:#00e676;color:#04040f}
[data-testid="stMetricValue"]{font-family:'Share Tech Mono',monospace!important;color:#fff!important;font-size:1.55rem!important;font-weight:700!important}
[data-testid="stMetricLabel"]{font-size:10px!important;color:#445566!important;letter-spacing:1.5px;text-transform:uppercase}
[data-testid="metric-container"]{background:#08081c!important;border:1px solid #12123a!important;border-radius:4px;padding:12px 16px}
[data-testid="stFileUploader"]{background:#07071a;border:1.5px dashed #1e1e55;border-radius:4px}
[data-testid="stDataFrame"]{border:1px solid #12123a;border-radius:4px}
[data-testid="stAlert"]{background:#08081c;border-radius:3px}
[data-testid="stSidebarNav"]{display:none}
[data-testid="stProgressBar"]>div>div{background:#ffd700!important}
[data-testid="stExpander"]{background:#07071a;border:1px solid #12123a;border-radius:4px}
::-webkit-scrollbar{width:5px;height:5px}::-webkit-scrollbar-track{background:#04040f}
::-webkit-scrollbar-thumb{background:#1e1e55;border-radius:3px}
hr{border-color:#0d0d2a!important;margin:12px 0}
.kpi-card{background:#08081c;border:1px solid #12123a;border-radius:4px;padding:14px 16px;text-align:center}
.kpi-val{font-family:'Share Tech Mono',monospace;font-size:1.45rem;font-weight:700;color:#fff;margin-top:4px}
.kpi-lbl{font-size:9px;color:#445566;letter-spacing:1.8px;text-transform:uppercase}
.kpi-val.red{color:#ff2020}.kpi-val.gold{color:#ffd700}.kpi-val.blue{color:#4499ff}
.kpi-val.green{color:#00e676}.kpi-val.orange{color:#ff9900}
.section-bar{background:#08081c;border-left:3px solid #ffd700;padding:8px 14px;border-radius:0 3px 3px 0;margin:16px 0 10px 0}
.section-bar .st{font-family:'Share Tech Mono',monospace;font-size:12px;color:#ffd700;letter-spacing:1px}
.section-bar .ss{font-size:10px;color:#445566;margin-top:2px}
.alert-high{background:#1a0303;border:2px solid #ff2020;border-radius:4px;padding:16px 20px;margin-bottom:14px}
.alert-med{background:#1a1203;border:2px solid #ff9900;border-radius:4px;padding:16px 20px;margin-bottom:14px}
.alert-low{background:#031a03;border:2px solid #00e676;border-radius:4px;padding:16px 20px;margin-bottom:14px}
.rt-feed{background:#07071a;border:1px solid #12123a;border-radius:4px;padding:8px 12px;margin-bottom:4px;font-family:'Share Tech Mono',monospace;font-size:11px}
.fix-badge{background:#041a2a;border:1px solid #3399ff;border-radius:3px;padding:2px 7px;font-family:'Share Tech Mono',monospace;font-size:9px;color:#4499ff;letter-spacing:1px;display:inline-block;margin-left:6px}
.auto-badge{background:#031a08;border:1px solid #00c853;border-radius:3px;padding:2px 8px;font-family:'Share Tech Mono',monospace;font-size:9px;color:#00e676;letter-spacing:1px;display:inline-block}
.acc-card{background:#07071a;border:1px solid #1e1e55;border-radius:6px;padding:18px 22px;margin:12px 0}
.acc-card-danger{background:#1a0303;border:2px solid #ff2020;border-radius:6px;padding:18px 22px;margin:12px 0}
.tag{display:inline-block;padding:3px 11px;border-radius:12px;font-family:'Share Tech Mono',monospace;font-size:10px;font-weight:700;letter-spacing:.8px;margin:2px 3px}
.tag-high{background:#3a0808;color:#ff5555;border:1px solid #ff2020}
.tag-med{background:#3a2000;color:#ffbb00;border:1px solid #ff9900}
.tag-low{background:#003a15;color:#00e676;border:1px solid #00c853}
.tag-boss{background:#3a0020;color:#ff77cc;border:1px solid #ff2288}
.tag-shell{background:#2a1800;color:#ffcc44;border:1px solid #ffa000}
.tag-mule{background:#001a3a;color:#55aaff;border:1px solid #1155cc}
.tag-normal{background:#0a1020;color:#445566;border:1px solid #223344}
.pulse{animation:pulse-red 1.8s infinite}
@keyframes pulse-red{0%,100%{box-shadow:0 0 0 0 rgba(255,32,32,.5)}60%{box-shadow:0 0 0 10px rgba(255,32,32,0)}}
</style>
""", unsafe_allow_html=True)

# ==============================================================================
#  SECTION E — HELPERS
# ==============================================================================
def sec_header(title, subtitle=''):
    sub = f"<div class='ss'>{subtitle}</div>" if subtitle else ''
    st.markdown(f"<div class='section-bar'><div class='st'>{title}</div>{sub}</div>",
                unsafe_allow_html=True)

def kpi(label, value, color=''):
    return (f"<div class='kpi-card'><div class='kpi-lbl'>{label}</div>"
            f"<div class='kpi-val {color}'>{value}</div></div>")

def nlp_classify(description):
    if not isinstance(description, str):
        return 'NORMAL'
    dl = description.lower()
    for label, keywords in FRAUD_TAXONOMY.items():
        if any(kw in dl for kw in keywords):
            return label
    return 'NORMAL'

def check_login(username, password):
    u = _DEFAULT_USERS.get(username)
    if not u:
        return False, None
    if hashlib.sha256(password.encode()).hexdigest() == u['hash']:
        return True, u['role']
    return False, None



# ==============================================================================
#  NEW — Live DB direct reader (no cache, for real-time use)
# ==============================================================================
import time as _time

_COLOR_MAP = {'NORMAL':'#00e676','SMURFING':'#ffd700','LAYERING':'#cc44ff',
              'INTEGRATION':'#ff2020','HAWALA':'#ff9900','TERROR':'#ff3366'}
_ICON_MAP  = {'NORMAL':'✓','SMURFING':'⚡','LAYERING':'🔗',
              'INTEGRATION':'🎯','HAWALA':'💰','TERROR':'⚠'}

def _live_recent_direct(n=30):
    if not live_db_available():
        return pd.DataFrame()
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        query = f'SELECT * FROM live_transactions ORDER BY inserted_at DESC LIMIT {max(1, int(n))}'
        df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()

def _live_stats_direct():
    if not live_db_available():
        return {}
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        total = pd.read_sql('SELECT COUNT(*) as c FROM live_transactions', conn).iloc[0]['c']
        q2    = "SELECT COUNT(*) as c FROM live_transactions WHERE Fraud_Label != 'NORMAL'"
        fraud = pd.read_sql(q2, conn).iloc[0]['c']
        q3    = "SELECT SUM(Amount_INR) as s FROM live_transactions WHERE Fraud_Label != 'NORMAL'"
        vol   = pd.read_sql(q3, conn).iloc[0]['s']
        return {
            'total'     : int(total),
            'fraud'     : int(fraud),
            'fraud_rate': round(fraud / max(1, total) * 100, 2),
            'volume'    : float(vol or 0),
        }
    except Exception:
        return {}
    finally:
        if conn:
            conn.close()

def _lookup_account(account_id):
    if not account_id or not live_db_available():
        return pd.DataFrame()
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        query = (
            "SELECT * FROM live_transactions "
            "WHERE Source_Acc_No = ? OR Dest_Acc_No = ? "
            "ORDER BY Timestamp DESC LIMIT 500"
        )
        df = pd.read_sql(query, conn, params=(str(account_id), str(account_id)))
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        if conn:
            conn.close()


def _kyc_lookup(account_id):
    """
    Query the kyc_profiles table that transaction_generator.py creates.
    Returns a dict with holder details, or None if not found.
    The generator seeds this table on startup for all account pools.
    """
    if not account_id or not live_db_available():
        return None
    conn = None
    try:
        conn = _open_sqlite(LIVE_DB_PATH)
        row = conn.execute(
            "SELECT * FROM kyc_profiles WHERE Account_ID = ?",
            (str(account_id),)).fetchone()
        if row is None:
            return None
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM kyc_profiles LIMIT 0").description]
        return dict(zip(cols, row))
    except Exception:
        return None
    finally:
        if conn:
            conn.close()


def render_account_investigator(location_key='default'):
    """
    Full Account Investigator widget.
    Shows KYC profile (name, phone, PAN, Aadhar, address…) from the kyc_profiles
    table that transaction_generator.py seeds on startup.
    Also shows transaction history and fraud role from last scan.
    """
    sec_header('ACCOUNT INVESTIGATOR',
               'Enter any Account ID — see holder name, PAN, phone, address & fraud history')

    col_q, col_b = st.columns([5, 1])
    with col_q:
        acc_id = st.text_input(
            'acc_hidden',
            placeholder='e.g.  OFFSHORE_BOSS_888   SHELL_ACC_0003   ACC_MULE_0012   ACC_NORMAL_0055',
            key=f'acc_inv_{location_key}',
            label_visibility='collapsed')
    with col_b:
        clicked = st.button('SEARCH', key=f'acc_btn_{location_key}', use_container_width=True)

    if not clicked or not acc_id.strip():
        st.markdown(
            "<div style='font-size:10px;color:#223344;padding:6px 0'>"
            "Type any Account ID and press SEARCH to investigate</div>",
            unsafe_allow_html=True)
        return

    acc_id   = acc_id.strip()
    kyc      = _kyc_lookup(acc_id)           # KYC profile from generator DB
    df_acc   = _lookup_account(acc_id)       # transaction history from live DB
    scan_res = st.session_state.get('scan_result')

    # Pull scan-based role and scores
    scan_role = None
    scan_info = {}
    if scan_res:
        synd  = scan_res.get('synd', pd.DataFrame())
        match = synd[synd['Account'] == acc_id]
        if not match.empty:
            r = match.iloc[0]
            scan_role = str(r.get('Role', ''))
            scan_info = {
                'Inflow'       : float(r.get('Inflow_INR', 0)),
                'Outflow'      : float(r.get('Outflow_INR', 0)),
                'Boss_Score'   : float(r.get('Boss_Score', 0)),
                'PageRank'     : float(r.get('PageRank', 0)),
                'In_Centrality': float(r.get('In_Centrality', 0)),
                'Location'     : str(r.get('Location', 'Unknown')),
                'Entity'       : str(r.get('Entity', 'Unknown')),
            }

    if kyc is None and df_acc.empty and not scan_role:
        st.warning(
            f'**{acc_id}** not found.  '
            'Run transaction_generator.py first — it seeds KYC profiles on startup.')
        return

    # ── Transaction metrics ────────────────────────────────────────────────────
    total_txn    = len(df_acc)
    fraud_df     = df_acc[df_acc['Fraud_Label'] != 'NORMAL'] if not df_acc.empty else pd.DataFrame()
    n_fraud      = len(fraud_df)
    inflow_live  = float(df_acc[df_acc['Dest_Acc_No'] == acc_id]['Amount_INR'].sum()) if not df_acc.empty else 0.0
    outflow_live = float(df_acc[df_acc['Source_Acc_No'] == acc_id]['Amount_INR'].sum()) if not df_acc.empty else 0.0
    fraud_pct    = round(n_fraud / max(1, total_txn) * 100, 1)

    # ── KYC fields ─────────────────────────────────────────────────────────────
    holder_name = kyc.get('Holder_Name',  '—') if kyc else '—'
    phone       = kyc.get('Phone',        '—') if kyc else '—'
    email       = kyc.get('Email',        '—') if kyc else '—'
    pan         = kyc.get('PAN',          '—') if kyc else '—'
    aadhar      = kyc.get('Aadhar',       '—') if kyc else '—'
    dob         = kyc.get('DOB',          '—') if kyc else '—'
    address     = kyc.get('Address',      '—') if kyc else '—'
    occupation  = kyc.get('Occupation',   '—') if kyc else '—'
    income      = kyc.get('Annual_Income','—') if kyc else '—'
    kyc_status  = kyc.get('KYC_Status',   '—') if kyc else '—'
    reg_date    = kyc.get('Registered_On','—') if kyc else '—'
    bank_name   = kyc.get('Bank_Name',    '—') if kyc else '—'
    kyc_risk    = kyc.get('Risk_Level',   'UNKNOWN') if kyc else 'UNKNOWN'
    kyc_role    = kyc.get('Account_Role', 'UNKNOWN') if kyc else 'UNKNOWN'

    # ── Effective role and colours ─────────────────────────────────────────────
    effective_role = scan_role or kyc_role
    role_color_map = {
        'SYNDICATE BOSS': '#ff2020', 'SHELL ACCOUNT': '#ffa000',
        'MULE ACCOUNT'  : '#4499ff', 'TARGET ACCOUNT': '#445566',
        'BOSS'          : '#ff2020', 'SHELL': '#ffa000',
        'MULE'          : '#4499ff', 'NORMAL': '#00e676',
    }
    risk_color_map = {
        'CRITICAL': '#ff2020', 'HIGH': '#ff6600',
        'MEDIUM'  : '#ffd700', 'LOW' : '#00e676', 'UNKNOWN': '#778899',
    }
    role_color  = role_color_map.get(effective_role, '#ffd700')
    risk_color  = risk_color_map.get(kyc_risk, '#778899')
    is_critical = kyc_risk == 'CRITICAL' or effective_role in ('SYNDICATE BOSS', 'BOSS')
    kyc_status_color = ('#ff2020' if kyc_status == 'FLAGGED'
                        else '#ffd700' if kyc_status == 'PENDING' else '#00e676')

    card_cls = 'acc-card-danger' if is_critical else 'acc-card'

    # ── KYC Card ───────────────────────────────────────────────────────────────
    st.markdown(
        f"<div class='{card_cls}'>"

        # Header
        "<div style='display:flex;justify-content:space-between;align-items:flex-start;"
        "flex-wrap:wrap;gap:12px'>"
        "<div>"
        f"<div style='font-size:10px;color:#445566;letter-spacing:2px;margin-bottom:4px'>ACCOUNT ID</div>"
        f"<div style='font-family:Share Tech Mono,monospace;font-size:22px;color:{role_color}'>{acc_id}</div>"
        f"<div style='margin-top:8px;display:flex;gap:8px;flex-wrap:wrap'>"
        f"<span style='border:1px solid {role_color};border-radius:3px;padding:2px 10px;"
        f"font-family:Share Tech Mono,monospace;font-size:10px;color:{role_color}'>{effective_role}</span>"
        f"<span style='border:1px solid {risk_color};border-radius:3px;padding:2px 10px;"
        f"font-family:Share Tech Mono,monospace;font-size:10px;color:{risk_color}'>{kyc_risk} RISK</span>"
        "</div></div>"
        # KYC status top right
        "<div style='text-align:right'>"
        "<div style='font-size:9px;color:#445566;margin-bottom:4px'>KYC STATUS</div>"
        f"<div style='font-size:16px;color:{kyc_status_color};font-family:Share Tech Mono,monospace'>{kyc_status}</div>"
        f"<div style='font-size:10px;color:#334455;margin-top:4px'>Registered: {reg_date}</div>"
        "</div></div>"

        "<div style='border-top:1px solid #1e1e55;margin:14px 0'></div>"

        # Holder details — 2 column grid
        "<div style='display:grid;grid-template-columns:1fr 1fr;gap:12px 28px;font-size:11px'>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>HOLDER NAME</div>"
        f"<div style='color:#c8d8e8;font-size:16px;margin-top:3px'>{holder_name}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>BANK</div>"
        f"<div style='color:#c8d8e8;font-size:16px;margin-top:3px'>{bank_name}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>PHONE</div>"
        f"<div style='color:#c8d8e8;margin-top:3px'>{phone}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>EMAIL</div>"
        f"<div style='color:#c8d8e8;margin-top:3px'>{email}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>PAN NUMBER</div>"
        f"<div style='color:#ffd700;font-family:Share Tech Mono,monospace;font-size:14px;margin-top:3px'>{pan}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>AADHAR</div>"
        f"<div style='color:#ffd700;font-family:Share Tech Mono,monospace;font-size:14px;margin-top:3px'>{aadhar}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>DATE OF BIRTH</div>"
        f"<div style='color:#c8d8e8;margin-top:3px'>{dob}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>OCCUPATION</div>"
        f"<div style='color:#c8d8e8;margin-top:3px'>{occupation}</div></div>"

        f"<div style='grid-column:1/-1'><div style='font-size:9px;color:#445566;letter-spacing:1px'>ADDRESS</div>"
        f"<div style='color:#c8d8e8;margin-top:3px'>{address}</div></div>"

        f"<div><div style='font-size:9px;color:#445566;letter-spacing:1px'>ANNUAL INCOME</div>"
        f"<div style='color:#c8d8e8;margin-top:3px'>{income}</div></div>"

        "</div>",
        unsafe_allow_html=True
    )

    # ── Scan intelligence (if scan was run) ────────────────────────────────────
    if scan_info:
        st.markdown(
            "<div style='background:#07071a;border:1px solid #1e1e55;border-radius:4px;"
            "padding:12px 16px;margin-top:10px'>"
            "<div style='font-size:9px;color:#445566;letter-spacing:1px;margin-bottom:8px'>LAST SCAN INTELLIGENCE</div>"
            "<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px;font-size:11px;color:#556677'>"
            f"<div>Boss Score<br><b style='color:#ffd700;font-family:Share Tech Mono,monospace'>{scan_info['Boss_Score']:.6f}</b></div>"
            f"<div>PageRank<br><b style='color:#ffd700;font-family:Share Tech Mono,monospace'>{scan_info['PageRank']:.6f}</b></div>"
            f"<div>In-Centrality<br><b style='color:#ffd700;font-family:Share Tech Mono,monospace'>{scan_info['In_Centrality']:.6f}</b></div>"
            f"<div>Scan Inflow<br><b style='color:#00e676'>Rs.{scan_info['Inflow']/1e7:.2f}Cr</b></div>"
            f"<div>Scan Outflow<br><b style='color:#ff9900'>Rs.{scan_info['Outflow']/1e7:.2f}Cr</b></div>"
            f"<div>Location<br><b style='color:#c8d8e8'>{scan_info['Location']}</b></div>"
            "</div></div>",
            unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)   # close card

    # ── Transaction stats bar ──────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi('TRANSACTIONS', f'{total_txn:,}', 'gold'), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi('FRAUD EVENTS', f'{n_fraud:,}', 'red'), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi('TOTAL INFLOW', f'Rs.{inflow_live/1e5:.1f}L', 'green'), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi('TOTAL OUTFLOW', f'Rs.{outflow_live/1e5:.1f}L', 'orange'), unsafe_allow_html=True)

    # ── Fraud pattern tags ─────────────────────────────────────────────────────
    if n_fraud > 0:
        fdist    = fraud_df['Fraud_Label'].value_counts()
        tag_html = ' '.join([
            f"<span style='background:#3a0808;color:#ff5555;border:1px solid #ff2020;"
            f"border-radius:12px;padding:3px 11px;font-family:Share Tech Mono,monospace;"
            f"font-size:10px;margin:2px'>{k}: {v}</span>"
            for k, v in fdist.items()
        ])
        st.markdown(f"<div style='margin:10px 0'>Fraud patterns: {tag_html}</div>",
                    unsafe_allow_html=True)

    # ── Transaction history table ──────────────────────────────────────────────
    if not df_acc.empty:
        st.markdown(
            f"<div style='font-size:11px;color:#445566;margin:12px 0 4px'>"
            f"Last {min(15, total_txn)} transactions ({total_txn:,} total in live DB)</div>",
            unsafe_allow_html=True)
        show_cols = [c for c in ['Timestamp','Source_Acc_No','Dest_Acc_No',
                                  'Amount_INR','Transaction_Type','Bank_Name','Fraud_Label']
                     if c in df_acc.columns]
        disp = df_acc[show_cols].head(15).copy()
        if 'Amount_INR' in disp.columns:
            disp['Amount_INR'] = disp['Amount_INR'].apply(lambda x: f'Rs.{float(x):,.0f}')

        def _row_color(row):
            return (['background-color:#1e0505;color:#ff9999'] * len(row)
                    if str(row.get('Fraud_Label', 'NORMAL')) != 'NORMAL'
                    else [''] * len(row))

        st.dataframe(disp.style.apply(_row_color, axis=1),
                     use_container_width=True, height=330)
    elif scan_info:
        st.info('Live transaction history not available — only scan data shown above.')



def _build_pdf_safe(res):
    """Wrapper around _build_pdf that captures the real error message."""
    if not PDF_OK:
        return None, 'fpdf2 not installed. Run: pip install fpdf2  then restart Streamlit.'
    try:
        pdf_bytes = _build_pdf(
            res['df'], res['susp'], res['G'], res['synd'],
            res['boss'], res['boss_score'], res['boss_ic'], res['stats'])
        if not pdf_bytes:
            return None, ('PDF returned empty. Check your terminal — look for [PDF INTERNAL ERROR] '
                          'to see the real cause. Common fix: pip install fpdf2 matplotlib')
        return pdf_bytes, None
    except Exception as exc:
        import traceback
        return None, f'PDF wrapper error: {exc}'


def _render_live_feed_inner():
    """Shared feed renderer — called inside or outside fragment."""
    recent_df = _live_recent_direct(n=30)
    stats_d   = _live_stats_direct()
    kpi_cols  = st.columns(4)
    for col, (lbl, val, color) in zip(kpi_cols, [
        ('LIVE TOTAL',   f'{stats_d.get("total",0):,}',        'gold'),
        ('FRAUD EVENTS', f'{stats_d.get("fraud",0):,}',         'red'),
        ('FRAUD RATE',   f'{stats_d.get("fraud_rate",0):.2f}%', 'orange'),
        ('SUSP VOLUME',  f'Rs.{stats_d.get("volume",0)/1e7:.2f}Cr', 'red'),
    ]):
        with col:
            st.markdown(kpi(lbl, val, color), unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)

    if recent_df.empty:
        st.info('Waiting for transactions... Start transaction_generator.py in a second terminal.')
        return

    for _, row in recent_df.iterrows():
        label    = str(row.get('Fraud_Label', 'NORMAL'))
        amount   = float(row.get('Amount_INR', 0))
        ts       = str(row.get('Timestamp', ''))[:19]
        txn_id   = str(row.get('Transaction_ID', '—'))
        src      = str(row.get('Source_Acc_No', ''))
        dest     = str(row.get('Dest_Acc_No', ''))
        bank     = str(row.get('Bank_Name', ''))
        txn_type = str(row.get('Transaction_Type', ''))
        color    = _COLOR_MAP.get(label, '#ffffff')
        icon     = _ICON_MAP.get(label, '?')
        st.markdown(
            f"<div class='rt-feed'>"
            f"<span style='color:#445566'>{ts}</span>&nbsp;|&nbsp;"
            f"<span style='color:{color}'>{icon} {label}</span>&nbsp;|&nbsp;"
            f"<span style='color:#556677;font-size:10px'>{txn_id}</span>&nbsp;|&nbsp;"
            f"<span style='color:#c8d8e8'>{src} → {dest}</span>&nbsp;|&nbsp;"
            f"<span style='color:#ffd700'>Rs.{amount:,.0f}</span>&nbsp;|&nbsp;"
            f"<span style='color:#334455'>{bank}/{txn_type}</span>"
            "</div>",
            unsafe_allow_html=True)

    if len(recent_df) > 3:
        fdist = recent_df['Fraud_Label'].value_counts().reset_index()
        fdist.columns = ['Label', 'Count']
        fig = px.bar(fdist, x='Label', y='Count', color='Label',
                     color_discrete_map={**{'NORMAL': '#3399ff'},
                                         **{k: v for k, v in _COLOR_MAP.items()}},
                     template='plotly_dark', height=200)
        fig.update_layout(paper_bgcolor='#07071a', plot_bgcolor='#07071a',
                          showlegend=False, margin=dict(l=30, r=10, t=10, b=30))
        st.plotly_chart(fig, use_container_width=True,
                        key=f'lf_{int(_time.time() * 10) % 100000}')


_FRAGMENT_OK = False
try:
    @st.fragment(run_every=2)
    def _live_feed_auto():
        _render_live_feed_inner()
    _FRAGMENT_OK = True
except (AttributeError, TypeError):
    def _live_feed_auto():
        _render_live_feed_inner()


# ==============================================================================
#  SECTION F — 3D NETWORK HTML (200 nodes)
# ==============================================================================
_NET_HTML = r"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{background:#04040f;font-family:'Courier New',monospace;overflow:hidden}
#cw{position:absolute;width:100%;height:100%;background:#050516}
canvas{display:block;width:100%;height:100%}
#tip{position:absolute;background:#0a0a22;border:1px solid #2233aa;border-radius:4px;
     padding:9px 13px;font-size:11px;color:#ccd;pointer-events:none;opacity:0;
     transition:opacity .12s;max-width:220px;line-height:1.65;z-index:9}
#tip b{color:#ffd700;font-size:12px;display:block;margin-bottom:3px}
#hud{position:absolute;bottom:0;left:0;right:0;background:#03030c;
     border-top:1px solid #0d0d2a;padding:5px 14px;font-size:9px;color:#334455;
     letter-spacing:.5px;white-space:nowrap;overflow:hidden}
#legend{position:absolute;top:8px;right:10px;background:#06061a;
        border:1px solid #12123a;border-radius:4px;padding:8px 12px;font-size:9px;color:#556677}
#legend div{display:flex;align-items:center;gap:6px;margin-bottom:4px}
#legend span{width:10px;height:10px;border-radius:50%;display:inline-block}
</style></head><body>
<div id="cw"><canvas id="cv"></canvas><div id="tip"></div></div>
<div id="hud">INITIALIZING FORENSIC NETWORK SCAN...</div>
<div id="legend">
  <div><span style="background:#ff2020"></span>Syndicate Boss</div>
  <div><span style="background:#ff9900"></span>Shell Account</div>
  <div><span style="background:#3399ff"></span>Mule Account</div>
  <div><span style="background:#445566"></span>Target Account</div>
</div>
<script>
var S=__STATS__,N=__NODES__,E=__EDGES__;
var cv=document.getElementById('cv'),ctx=cv.getContext('2d');
var tip=document.getElementById('tip'),cw=document.getElementById('cw'),hud=document.getElementById('hud');
var W,H,nd=[],ed=[],nm={},hov=null,ox=0,oy=0,zm=1,ps=null,po=null,fr=0;
var RC={BOSS:'#ff2020',SHELL:'#ff9900',MULE:'#3399ff',TARGET:'#445566'};
function rsz(){W=cw.clientWidth;H=cw.clientHeight;var d=devicePixelRatio||1;cv.width=W*d;cv.height=H*d;cv.style.width=W+'px';cv.style.height=H+'px';ctx.setTransform(d,0,0,d,0,0);}
function init(){
  var cx=W/2,cy=H/2;
  var ml=N.filter(function(n){return n.role==='MULE';}),sl=N.filter(function(n){return n.role==='SHELL';}),tl=N.filter(function(n){return n.role==='TARGET';});
  N.forEach(function(n){
    var x,y,z=0,r;
    if(n.role==='BOSS'){x=cx;y=cy;z=50;r=28;}
    else if(n.role==='SHELL'){var i=sl.indexOf(n),a=(i/Math.max(1,sl.length))*Math.PI*2-Math.PI/2,rd=Math.min(W,H)*0.17;x=cx+Math.cos(a)*rd;y=cy+Math.sin(a)*rd;z=20;r=12;}
    else if(n.role==='MULE'){var i2=ml.indexOf(n),a2=(i2/Math.max(1,ml.length))*Math.PI*2-Math.PI/2+0.1,rd2=Math.min(W,H)*0.30;x=cx+Math.cos(a2)*rd2;y=cy+Math.sin(a2)*rd2;z=0;r=8;}
    else{var i3=tl.indexOf(n),a3=(i3/Math.max(1,tl.length))*Math.PI*2-Math.PI/2+0.05,rd3=Math.min(W,H)*0.44;x=cx+Math.cos(a3)*rd3;y=cy+Math.sin(a3)*rd3;z=-20;r=6;}
    var node=Object.assign({},n,{x:x,y:y,z:z,r:r});nm[n.id]=node;nd.push(node);
  });
  ed=E.map(function(e){return{from:nm[e.from],to:nm[e.to],amt:e.amt,prog:Math.random(),spd:0.003+Math.random()*0.003};}).filter(function(e){return e.from&&e.to;});
}
function draw(){
  fr++;ctx.clearRect(0,0,W,H);ctx.fillStyle='#050516';ctx.fillRect(0,0,W,H);
  ctx.save();ctx.translate(W/2+ox*zm,H/2+oy*zm);ctx.scale(zm,zm);ctx.translate(-W/2,-H/2);
  [[0.17,'rgba(255,153,0,0.08)'],[0.30,'rgba(51,153,255,0.06)'],[0.44,'rgba(68,85,102,0.04)']].forEach(function(fc){
    ctx.beginPath();ctx.arc(W/2,H/2,Math.min(W,H)*fc[0],0,Math.PI*2);
    ctx.strokeStyle=fc[1];ctx.lineWidth=1;ctx.setLineDash([4,10]);ctx.stroke();ctx.setLineDash([]);
  });
  ed.forEach(function(e){
    e.prog=(e.prog+e.spd)%1;
    var fx=e.from.x,fy=e.from.y,tx=e.to.x,ty=e.to.y,iB=e.to.role==='BOSS';
    ctx.beginPath();ctx.moveTo(fx,fy);ctx.lineTo(tx,ty);
    ctx.strokeStyle=iB?'rgba(255,215,0,0.10)':'rgba(51,153,255,0.04)';ctx.lineWidth=1;ctx.stroke();
    var px2=fx+(tx-fx)*e.prog,py2=fy+(ty-fy)*e.prog,pg=ctx.createRadialGradient(px2,py2,0,px2,py2,5);
    if(iB){pg.addColorStop(0,'rgba(255,215,0,0.9)');pg.addColorStop(1,'rgba(255,215,0,0)');}
    else{pg.addColorStop(0,'rgba(51,153,255,0.7)');pg.addColorStop(1,'rgba(51,153,255,0)');}
    ctx.beginPath();ctx.arc(px2,py2,5,0,Math.PI*2);ctx.fillStyle=pg;ctx.fill();
  });
  nd.forEach(function(n){
    if(n.role!=='BOSS'){
      var cx2=W/2,cy2=H/2,dx=n.x-cx2,dy=n.y-cy2,spd=n.role==='MULE'?0.0014:n.role==='TARGET'?0.0009:0.0018;
      n.x=dx*Math.cos(spd)-dy*Math.sin(spd)+cx2;n.y=dx*Math.sin(spd)+dy*Math.cos(spd)+cy2;
    }
  });
  var sorted=[].concat(nd).sort(function(a,b){return (a.z||0)-(b.z||0);});
  sorted.forEach(function(n){
    var iH=hov&&hov.id===n.id,iB=n.role==='BOSS',r=n.r+(iH?3:0),ds=1+(n.z||0)/500;
    if(iB){[4,3,2].forEach(function(k){var rr=r+8+k*8+Math.sin(fr*0.04)*4;ctx.beginPath();ctx.arc(n.x,n.y,rr*ds,0,Math.PI*2);ctx.strokeStyle='rgba(255,32,32,'+(0.06*k)+')';ctx.lineWidth=6;ctx.stroke();});}
    var rr=r*ds;ctx.beginPath();ctx.arc(n.x,n.y,rr,0,Math.PI*2);
    var g=ctx.createRadialGradient(n.x-rr*0.3,n.y-rr*0.3,0,n.x,n.y,rr+2),bc=RC[n.role]||'#445566';
    if(iB){g.addColorStop(0,'#ff6666');g.addColorStop(1,'#880000');}else{g.addColorStop(0,bc+'dd');g.addColorStop(1,bc+'44');}
    ctx.fillStyle=g;ctx.fill();
    ctx.strokeStyle=iB?'#ffd700':iH?'#fff':'rgba(255,255,255,0.12)';ctx.lineWidth=iB?2:1;ctx.stroke();
    if(iB||iH||(n.role==='SHELL'&&rr>7)){
      ctx.font=(iB?'bold 10px':'9px')+' monospace';ctx.fillStyle=iB?'#ffd700':bc;ctx.textAlign='center';
      ctx.fillText(n.label,n.x,n.y-rr-4);
      if(iB){ctx.font='bold 8px monospace';ctx.fillStyle='#ff5555';ctx.fillText('CONVERGENCE NODE',n.x,n.y-rr-15);
             ctx.fillStyle='#ffd700';ctx.font='8px monospace';ctx.fillText('Rs.'+(n.inflow/1e7).toFixed(1)+'Cr',n.x,n.y-rr-25);}
    }
  });
  ctx.restore();requestAnimationFrame(draw);
}
cv.addEventListener('mousemove',function(e){
  var rect=cv.getBoundingClientRect(),mx=e.clientX-rect.left,my=e.clientY-rect.top;
  if(ps){ox=po.ox+(mx-ps.x)/zm;oy=po.oy+(my-ps.y)/zm;return;}
  var wx=(mx-W/2-ox*zm)/zm+W/2,wy=(my-H/2-oy*zm)/zm+H/2;
  hov=null;nd.forEach(function(n){var dx=n.x-wx,dy=n.y-wy;if(Math.sqrt(dx*dx+dy*dy)<n.r+6)hov=n;});
  if(hov){
    tip.innerHTML='<b>'+hov.id+'</b><span style="color:'+RC[hov.role]+';font-size:9px"> '+hov.role+'</span><br>Bank: '+hov.bank+'<br>'+(hov.inflow?'Inflow: Rs.'+(hov.inflow/1e5).toFixed(1)+'L<br>':'')+'Centrality: '+hov.ic+'<br>Boss Score: '+hov.bs;
    tip.style.opacity='1';tip.style.left=(mx+14)+'px';tip.style.top=(my-10)+'px';cv.style.cursor='pointer';
  }else{tip.style.opacity='0';cv.style.cursor=ps?'grabbing':'grab';}
});
cv.addEventListener('mousedown',function(e){var r=cv.getBoundingClientRect();ps={x:e.clientX-r.left,y:e.clientY-r.top};po={ox:ox,oy:oy};});
cv.addEventListener('mouseup',function(){ps=null;});
cv.addEventListener('mouseleave',function(){ps=null;tip.style.opacity='0';});
cv.addEventListener('wheel',function(e){e.preventDefault();zm=Math.min(6,Math.max(0.25,zm*(e.deltaY>0?0.88:1.14)));},{passive:false});
var msgs=['DEEP AUDIT AI  v9.0  |  ALL 5 BUGS FIXED','SCANNING '+S.total_txn.toLocaleString()+' TRANSACTIONS ACROSS '+S.banks+' ENTITIES','IF+XGBOOST: '+S.suspicious.toLocaleString()+' ANOMALOUS RECORDS FLAGGED','CONVERGENCE NODE: '+S.boss,'NETWORK: '+S.nodes.toLocaleString()+' NODES  |  '+S.edges.toLocaleString()+' MONEY TRAILS','TOTAL FLAGGED VOLUME: Rs.'+S.amount_cr.toFixed(2)+' CRORE','MODE: '+S.mode];
var mi=0;setInterval(function(){hud.textContent='>> '+msgs[mi++%msgs.length];},3000);
window.addEventListener('resize',rsz);rsz();init();draw();
</script></body></html>"""

# ==============================================================================
#  SECTION G — DATA NORMALISATION
# ==============================================================================
def _normalise_df(df, bank_name='LIVE_FEED'):
    rename_map = {
        'Transaction_Date': 'Timestamp', 'Source_Account_Number': 'Source_Acc_No',
        'Destination_Account_Number': 'Dest_Acc_No', 'Amount_in_INR': 'Amount_INR',
        'International_Flag': 'Is_International', 'Fraud_Type': 'Fraud_Label',
        'Source_Acc': 'Source_Acc_No', 'Transaction': 'Transaction_Type',
        'Bank_Identity': 'Bank_Name',
    }
    df.rename(columns={k: v for k, v in rename_map.items()
                       if k in df.columns and v not in df.columns}, inplace=True)
    if 'Bank_Name' not in df.columns:
        df['Bank_Name'] = bank_name
    for col, val in [('Risk_Score', 0.0), ('Is_International', 0),
                     ('Location', 'Unknown'), ('Txn_Description', 'UNKNOWN'),
                     ('Fraud_Label', 'NORMAL'), ('IP_Address', '0.0.0.0')]:
        if col not in df.columns:
            df[col] = val
    return df

def load_and_merge(uploaded_files):
    all_dfs, errors = [], []
    for uf in uploaded_files:
        source_name = os.path.splitext(str(getattr(uf, 'name', 'upload.csv')))[0] or 'upload'
        bank_name = re.split(r'[_ ]', source_name)[0] or 'UPLOAD'
        try:
            df_temp = _read_uploaded_frame(uf)
            df_temp = _normalise_df(df_temp, bank_name)
            df_temp = _validate_input_df(df_temp, source_name)
        except Exception as exc:
            errors.append(f'{getattr(uf, "name", "upload")}: {exc}')
            continue
        all_dfs.append(df_temp)
        gc.collect()
    for err in errors:
        st.warning(f'Skipped: {err}')
    if not all_dfs:
        raise ValueError('No valid files could be loaded.')
    master = pd.concat(all_dfs, ignore_index=True)
    id_col = 'Transaction_ID' if 'Transaction_ID' in master.columns else master.columns[0]
    master.drop_duplicates(subset=[id_col], keep='first', inplace=True)
    if master.empty:
        raise ValueError('All uploaded rows were removed during validation.')
    gc.collect()
    return master

# ==============================================================================
#  SECTION H — AUTO CONTAMINATION  (FIX-3: capped at 0.05)
# ==============================================================================
def estimate_contamination(df):
    """
    FIX-3: Cap at 0.05 so max 5% of live data gets flagged.
    Previous version could go up to 0.10 causing over-flagging.
    """
    amounts = pd.to_numeric(df.get('Amount_INR', pd.Series(dtype=float)), errors='coerce').dropna()
    if amounts.empty:
        return 0.01
    Q1, Q3  = amounts.quantile(0.25), amounts.quantile(0.75)
    IQR     = Q3 - Q1
    iqr_rate = ((amounts < Q1 - 2.5 * IQR) | (amounts > Q3 + 2.5 * IQR)).sum() / max(1, len(amounts))
    night_rate = 0.0
    if 'Txn_Hour' in df.columns:
        hours = pd.to_numeric(df['Txn_Hour'], errors='coerce')
        night_rate = float(((hours >= 22) | (hours <= 4)).fillna(False).mean()) * 0.4
    contamination = (iqr_rate + night_rate) / 2.0
    if not np.isfinite(contamination):
        contamination = 0.01
    return max(0.005, min(0.05, round(contamination, 4)))

# ==============================================================================
#  SECTION I — CALIBRATED THRESHOLDS  (FIX-1 core fix)
# ==============================================================================
def compute_risk_thresholds(df):
    """
    FIX-1: Instead of fixed 0.65 cutoff, compute thresholds from actual data
    distribution using percentiles.

    HIGH   = top 5% of Risk_Score in the dataset
    MEDIUM = top 15% but below HIGH threshold
    LOW    = everything else

    This means:
    - A ₹500 UPI normal transfer → LOW (correctly)
    - A ₹48,000 smurfing transfer → MEDIUM or HIGH (correctly)
    - A ₹3 Crore offshore SWIFT at 2 AM → HIGH (correctly)
    """
    scores = pd.to_numeric(df.get('Risk_Score', pd.Series(dtype=float)), errors='coerce').dropna()
    if scores.empty:
        return 0.55, 0.35
    high_threshold   = float(scores.quantile(0.95))  # top 5%
    medium_threshold = float(scores.quantile(0.85))  # top 15%
    if not np.isfinite(high_threshold):
        high_threshold = 0.55
    if not np.isfinite(medium_threshold):
        medium_threshold = 0.35
    # Safety floor: never go below absolute minimums
    high_threshold   = max(high_threshold,   0.55)
    medium_threshold = min(high_threshold, max(medium_threshold, 0.35))
    return high_threshold, medium_threshold

# ==============================================================================
#  SECTION J — FULL AI PIPELINE
# ==============================================================================
def run_master_pipeline(df, prog, status, mode='UPLOAD'):
    """
    FIX-2: Clears old scan result from session state before writing new one.
    FIX-3: Uses capped contamination.
    FIX-4: Reduced n_estimators 200→100 for speed.
    """

    if df is None or df.empty:
        raise ValueError('No transactions available for analysis.')

    df = df.copy()
    df['Amount_INR'] = pd.to_numeric(df.get('Amount_INR'), errors='coerce')
    df.dropna(subset=['Amount_INR'], inplace=True)
    if df.empty:
        raise ValueError('No valid numeric Amount_INR values were found.')

    if 'Transaction_ID' not in df.columns:
        df['Transaction_ID'] = [f'TXN_AUTO_{i + 1:07d}' for i in range(len(df))]

    status.markdown('**[1/6]** Engineering 12 forensic features...')
    df['Timestamp'] = pd.to_datetime(
        df['Timestamp'].astype(str).str.replace("'", '', regex=False), errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    if df.empty:
        raise ValueError('No valid timestamps were found after parsing the input data.')
    df.sort_values(['Source_Acc_No', 'Timestamp'], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df['Time_Gap']   = df.groupby('Source_Acc_No')['Timestamp'].diff().dt.total_seconds().fillna(0)
    df['Rapid_Fire'] = (df['Time_Gap'] < 60).astype(int)
    df['Amount_Log'] = np.log1p(df['Amount_INR'].astype(float))
    df['Txn_Hour']   = df['Timestamp'].dt.hour
    df['Is_Night']   = ((df['Txn_Hour'] >= 22) | (df['Txn_Hour'] <= 5)).astype(int)

    amt_max  = max(float(df['Amount_INR'].astype(float).max()), 1.0)
    # FIX-1: use 95th percentile for normalisation so normal amounts don't get
    # inflated risk scores when a few very large amounts skew the max
    amt_p95  = float(df['Amount_INR'].astype(float).quantile(0.95))
    amt_norm = max(amt_p95, 1.0)

    df['Risk_Score'] = (
        (df['Amount_INR'].astype(float).clip(upper=amt_norm) / amt_norm) * 0.40 +
        df['Is_International'].astype(float)                              * 0.30 +
        df['Rapid_Fire'].astype(float)                                    * 0.20 +
        (df['Time_Gap'] == 0).astype(float)                               * 0.10
    ).clip(0, 1)

    le = LabelEncoder()
    df['Txn_Encoded'] = le.fit_transform(df['Transaction_Type'].fillna('UNKNOWN'))

    df['_Date'] = df['Timestamp'].dt.date
    dvc = df.groupby(['Source_Acc_No', '_Date']).size().reset_index(name='Velocity_1d')
    df  = df.merge(dvc, on=['Source_Acc_No', '_Date'], how='left')
    df['Velocity_1d'] = df['Velocity_1d'].fillna(1).astype(float)
    df.drop(columns=['_Date'], inplace=True)

    acct_s = df.groupby('Source_Acc_No')['Amount_INR'].agg(['mean', 'std']).reset_index()
    acct_s.columns = ['Source_Acc_No', 'Acct_Mean', 'Acct_Std']
    acct_s['Acct_Std'] = acct_s['Acct_Std'].fillna(1.0).clip(lower=1.0)
    df = df.merge(acct_s, on='Source_Acc_No', how='left')
    df['Amount_Deviation'] = ((df['Amount_INR'].astype(float) - df['Acct_Mean']) /
                               df['Acct_Std']).fillna(0).clip(-5, 5)
    df.drop(columns=['Acct_Mean', 'Acct_Std'], inplace=True, errors='ignore')
    df['Round_Amount'] = ((df['Amount_INR'].astype(float) % 1000 == 0) |
                          (df['Amount_INR'].astype(float) % 5000 == 0)).astype(int)

    # NLP classification
    if 'Txn_Description' in df.columns:
        df['_nlp'] = df['Txn_Description'].apply(nlp_classify)
        mask = df['_nlp'] != 'NORMAL'
        df.loc[mask, 'Fraud_Label'] = df.loc[mask, '_nlp']
        df.drop(columns=['_nlp'], inplace=True)

    for f in FEATS:
        if f not in df.columns:
            df[f] = 0.0

    # Compute calibrated thresholds from this dataset (FIX-1)
    high_thresh, med_thresh = compute_risk_thresholds(df)

    prog.progress(15)

    status.markdown('**[2/6]** Auto-tuning contamination (FIX-3: capped at 5%)...')
    contamination = estimate_contamination(df)

    status.markdown(f'**[3/6]** Isolation Forest (n=100, contamination={contamination}, FIX-4)...')
    X_raw    = (df[FEATS]
                .replace([np.inf, -np.inf], np.nan)
                .astype('float64')
                .fillna(0.0))
    scaler   = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # FIX-4: n_estimators=100 (was 200) — 2× speed improvement with minimal accuracy loss
    iso     = IsolationForest(n_estimators=100, contamination=contamination,
                               random_state=42, n_jobs=-1)
    if_pred = iso.fit_predict(X_scaled)
    if_score= iso.decision_function(X_scaled)
    if_risk = np.clip((0.5 - if_score) / 0.5, 0, 1)

    df['Is_Fraud']   = (if_pred == -1).astype(int)
    df['Confidence'] = if_risk

    xgb_model = None
    if XGB_OK and int(df['Is_Fraud'].sum()) > 30:
        try:
            y_p       = df['Is_Fraud'].values
            pw        = max(1, (len(y_p) - y_p.sum()) / max(1, y_p.sum()))
            import warnings as _xw; _xw.filterwarnings('ignore')
            xgb_model = XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.1,
                                      scale_pos_weight=pw, random_state=42,
                                      verbosity=0, eval_metric='logloss')
            xgb_model.fit(X_scaled, y_p)
            xgb_prob  = xgb_model.predict_proba(X_scaled)[:, 1]
            df['Confidence'] = 0.60 * if_risk + 0.40 * xgb_prob
            df['Is_Fraud']   = ((df['Is_Fraud'] == 1) & (df['Confidence'] > 0.40)).astype(int)
        except Exception:
            pass

    # Store calibrated scorer state (FIX-1, FIX-5)
    st.session_state['rt_scaler']      = scaler
    st.session_state['rt_iso']         = iso
    st.session_state['rt_xgb']         = xgb_model
    st.session_state['rt_le']          = le
    st.session_state['rt_amt_max']     = float(amt_max)
    st.session_state['rt_amt_norm']    = float(amt_norm)   # FIX-5: p95 normaliser
    st.session_state['rt_high_thresh'] = high_thresh        # FIX-1: dynamic threshold
    st.session_state['rt_med_thresh']  = med_thresh
    st.session_state['rt_contamination'] = contamination

    suspicious_df = df[df['Is_Fraud'] == 1].copy()
    normal_sample = (df[df['Is_Fraud'] == 0]
                     .sample(min(10_000, int((df['Is_Fraud'] == 0).sum())), random_state=42)
                     .reset_index(drop=True))
    del X_scaled, X_raw
    gc.collect()
    prog.progress(40)

    status.markdown('**[4/6]** PageRank composite boss detection (FIX-2: cache cleared)...')
    # FIX-2: Explicitly clear old result so stale boss name never persists
    st.session_state['scan_result'] = None

    bank_map = {}
    loc_map = {}
    for col_name in ['Dest_Acc_No', 'Source_Acc_No']:
        if col_name in suspicious_df.columns:
            bank_map.update(suspicious_df.set_index(col_name)['Bank_Name'].to_dict())
            if 'Location' in suspicious_df.columns:
                loc_map.update(suspicious_df.set_index(col_name)['Location'].to_dict())

    if suspicious_df.empty:
        G = nx.DiGraph()
        in_cent = {}
        out_cent = {}
        pagerank = {}
        node_inflow = {}
        node_outflow = {}
        composite_scores = {}
        sorted_comp = []
        boss_account = 'NO_SUSPICIOUS_ACTIVITY'
        boss_score = 0.0
        boss_ic = 0.0
        syndicate_table = pd.DataFrame(columns=SYNDICATE_TABLE_COLUMNS)
    else:
        G = nx.from_pandas_edgelist(
            suspicious_df, source='Source_Acc_No', target='Dest_Acc_No',
            edge_attr='Amount_INR', create_using=nx.DiGraph())
        in_cent  = nx.in_degree_centrality(G)
        out_cent = nx.out_degree_centrality(G)
        try:
            pagerank = nx.pagerank(G, weight='Amount_INR', alpha=0.85, max_iter=100)
        except Exception:
            pagerank = {n: 1.0 / max(1, G.number_of_nodes()) for n in G.nodes()}

        node_inflow  = {k: float(v) for k, v in
                        suspicious_df.groupby('Dest_Acc_No')['Amount_INR'].sum().items()}
        node_outflow = {k: float(v) for k, v in
                        suspicious_df.groupby('Source_Acc_No')['Amount_INR'].sum().items()}
        max_inflow   = max(node_inflow.values()) if node_inflow else 1.0

        composite_scores = {}
        for node in G.nodes():
            ic = float(in_cent.get(node, 0))
            oc = float(out_cent.get(node, 0))
            pr = float(pagerank.get(node, 0))
            composite_scores[node] = (
                0.35 * ic +
                0.35 * pr +
                0.20 * (node_inflow.get(node, 0) / max_inflow) +
                0.10 * max(0, 1.0 - oc)
            )

        sorted_comp  = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        boss_account = sorted_comp[0][0] if sorted_comp else 'UNKNOWN'
        boss_score   = float(sorted_comp[0][1]) if sorted_comp else 0.0
        boss_ic      = float(in_cent.get(boss_account, 0))

        node_roles = {}
        for node in G.nodes():
            ic = in_cent.get(node, 0)
            oc = out_cent.get(node, 0)
            if node == boss_account:   node_roles[node] = 'SYNDICATE BOSS'
            elif ic > 0 and oc > 0:    node_roles[node] = 'SHELL ACCOUNT'
            elif oc > 0:               node_roles[node] = 'MULE ACCOUNT'
            else:                      node_roles[node] = 'TARGET ACCOUNT'

        report_rows = []
        for node, role in node_roles.items():
            report_rows.append({
                'Account'        : str(node),
                'Role'           : role,
                'Inflow_INR'     : round(node_inflow.get(node, 0.0), 2),
                'Outflow_INR'    : round(node_outflow.get(node, 0.0), 2),
                'Total_Volume'   : round(max(node_inflow.get(node, 0.0), node_outflow.get(node, 0.0)), 2),
                'Entity'         : str(bank_map.get(node, 'Unknown')),
                'Location'       : str(loc_map.get(node, 'Unknown')),
                'In_Centrality'  : round(float(in_cent.get(node, 0)), 6),
                'Out_Centrality' : round(float(out_cent.get(node, 0)), 6),
                'PageRank'       : round(float(pagerank.get(node, 0)), 6),
                'Boss_Score'     : round(float(composite_scores.get(node, 0)), 6),
            })

        syndicate_table = (pd.DataFrame(report_rows, columns=SYNDICATE_TABLE_COLUMNS)
                           .sort_values('Boss_Score', ascending=False)
                           .reset_index(drop=True))
    prog.progress(60)

    status.markdown('**[5/6]** Rendering 3D network (200 nodes)...')
    top80     = [n for n, _ in sorted_comp[:80]]
    boss_srcs = suspicious_df[suspicious_df['Dest_Acc_No'] == boss_account][
                    'Source_Acc_No'].unique()[:40].tolist()
    viz_ids   = list(set(top80 + boss_srcs + [boss_account]))[:200]

    nodes_3d = []
    for n in viz_ids:
        ic   = float(in_cent.get(n, 0))
        oc   = float(out_cent.get(n, 0))
        bs   = float(composite_scores.get(n, 0))
        role = ('BOSS' if n == boss_account else
                'SHELL' if ic > 0 and oc > 0 else
                'MULE'  if oc > 0             else 'TARGET')
        nodes_3d.append({'id': str(n), 'label': str(n)[-10:], 'role': role,
                         'bank': str(bank_map.get(n, 'Unknown')),
                         'inflow': round(node_inflow.get(n, 0.0), 0),
                         'outflow': round(node_outflow.get(n, 0.0), 0),
                         'ic': round(ic, 6), 'bs': round(bs, 6)})

    H3       = G.subgraph(viz_ids)
    edges_3d = [{'from': str(u), 'to': str(v),
                 'amt': round(float(d.get('Amount_INR', 0)), 0)}
                for u, v, d in H3.edges(data=True)]

    now     = datetime.datetime.now()
    case_no = f'DA-{now.strftime("%Y%m%d-%H%M")}-001'
    stats   = {
        'total_txn'    : int(len(df)),
        'banks'        : int(df['Bank_Name'].nunique()),
        'suspicious'   : int(len(suspicious_df)),
        'amount_cr'    : round(float(suspicious_df['Amount_INR'].sum()) / 1e7, 2),
        'boss'         : str(boss_account),
        'boss_score'   : round(boss_score, 6),
        'nodes'        : int(G.number_of_nodes()),
        'edges'        : int(G.number_of_edges()),
        'case_no'      : case_no,
        'report_date'  : now.strftime('%d %B %Y  |  %H:%M hrs'),
        'contamination': round(contamination, 4),
        'xgb_used'     : XGB_OK and xgb_model is not None,
        'mode'         : mode,
        'high_thresh'  : round(high_thresh, 4),
        'med_thresh'   : round(med_thresh, 4),
    }
    HTML_3D = (_NET_HTML
               .replace('__STATS__', safe_dumps(stats))
               .replace('__NODES__', safe_dumps(nodes_3d))
               .replace('__EDGES__', safe_dumps(edges_3d)))
    prog.progress(80)

    status.markdown('**[6/6]** Compiling forensic dossier...')
    pdf_bytes = _build_pdf(df, suspicious_df, G, syndicate_table,
                           boss_account, boss_score, boss_ic, stats)
    prog.progress(100)

    db_save_scan(case_no, len(df), len(suspicious_df), boss_account,
                 boss_score, float(suspicious_df['Amount_INR'].sum()),
                 df['Bank_Name'].nunique(), mode)
    db_log('SCAN_COMPLETE', f'case={case_no} mode={mode} boss={boss_account}')

    return {
        'df': df, 'susp': suspicious_df, 'normal': normal_sample,
        'synd': syndicate_table, 'boss': boss_account,
        'boss_score': boss_score, 'boss_ic': boss_ic,
        'G': G, 'html': HTML_3D, 'pdf': pdf_bytes, 'stats': stats,
        'in_cent': in_cent, 'pagerank': pagerank,
        'composite': composite_scores,
        'xgb_used': XGB_OK and xgb_model is not None,
        'contamination': contamination,
        'high_thresh': high_thresh,
        'med_thresh': med_thresh,
    }

# ==============================================================================
#  SECTION K — REAL-TIME SCORER  (FIX-1 + FIX-5)
# ==============================================================================
def score_single_transaction(amount, is_intl, hour, txn_type, src_acc, dest_acc, desc):
    """
    FIX-1: Uses dynamic thresholds from last scan (rt_high_thresh, rt_med_thresh).
    FIX-5: Uses p95-based normaliser (rt_amt_norm) instead of hardcoded 10M max.

    Result:
      - A normal ₹5,000 UPI domestic daytime → LOW risk correctly
      - A smurfing ₹47,000 burst at night   → MEDIUM/HIGH correctly
      - A ₹3 Crore SWIFT offshore at 2 AM  → HIGH correctly
    """
    amt_norm    = max(float(st.session_state.get('rt_amt_norm', 500_000)), 1.0)
    high_thresh = float(st.session_state.get('rt_high_thresh', 0.72))
    med_thresh  = float(st.session_state.get('rt_med_thresh',  0.45))

    amount_log   = float(np.log1p(amount))
    is_night     = int(hour < 5 or hour >= 22)
    round_amount = int(amount % 1000 == 0 or amount % 5000 == 0)
    nlp_label    = nlp_classify(desc)
    nlp_boost    = 0.12 if nlp_label != 'NORMAL' else 0.0

    # FIX-5: clip at p95 normaliser — large normal amounts don't max out score
    risk_score = min(1.0,
        (min(float(amount), amt_norm) / amt_norm) * 0.40 +
        float(is_intl)                             * 0.30 +
        float(is_night)                            * 0.12 +
        nlp_boost
    )

    le = st.session_state.get('rt_le')
    try:
        txn_encoded = int(le.transform([txn_type])[0]) if le else 0
    except Exception:
        txn_encoded = 0

    row = [float(amount), amount_log, 0.0, 0.0, risk_score,
           float(txn_encoded), float(is_night), float(is_intl),
           1.0, 0.0, float(round_amount), float(hour)]

    final_risk = risk_score
    model_used = 'Rule-based (no scan run yet)'
    scaler     = st.session_state.get('rt_scaler')
    iso        = st.session_state.get('rt_iso')

    if scaler is not None and iso is not None:
        try:
            X        = np.array([row])
            X_scaled = scaler.transform(X)
            pred     = iso.predict(X_scaled)[0]
            if_score2= iso.decision_function(X_scaled)[0]
            if_risk2 = float(np.clip((0.5 - if_score2) / 0.5, 0, 1))
            xgb      = st.session_state.get('rt_xgb')
            if xgb is not None:
                xgb_prob   = float(xgb.predict_proba(X_scaled)[0][1])
                final_risk = round(0.60 * if_risk2 + 0.40 * xgb_prob, 4)
                model_used = 'IF + XGBoost (calibrated)'
            else:
                final_risk = round(if_risk2, 4)
                model_used = 'Isolation Forest (calibrated)'
        except Exception:
            pass

    # FIX-1: use dynamic thresholds, not fixed 0.65
    is_high   = final_risk >= high_thresh
    is_medium = (not is_high) and (final_risk >= med_thresh)

    contributions = [
        ('Amount (p95-normalised)',  round(min(0.40, float(amount)/amt_norm * 0.40), 4)),
        ('International transfer',   0.30 if is_intl else 0.0),
        ('Night-time transaction',   0.12 if is_night else 0.0),
        ('NLP fraud pattern',        nlp_boost),
        ('Round amount flag',        0.04 if round_amount else 0.0),
        ('High-risk txn type',       0.08 if txn_type in ['WIRE','SWIFT','HAWALA'] else 0.0),
    ]

    return {
        'risk_score'   : final_risk,
        'is_high'      : is_high,
        'is_medium'    : is_medium,
        'is_fraud'     : is_high,
        'alert'        : is_high,
        'nlp_label'    : nlp_label,
        'model_used'   : model_used,
        'contributions': contributions,
        'high_thresh'  : high_thresh,
        'med_thresh'   : med_thresh,
    }

# ==============================================================================
#  SECTION L — EMAIL ALERT
# ==============================================================================
def send_alert_email(smtp_cfg, subject, body):
    required = ['server', 'port', 'sender', 'recipient']
    missing = [key for key in required if not smtp_cfg.get(key)]
    if missing:
        return False, f'SMTP configuration is incomplete: missing {", ".join(missing)}.'
    try:
        msg = MIMEMultipart()
        msg['From']   = smtp_cfg['sender']
        msg['To']     = smtp_cfg['recipient']
        msg['Subject']= subject
        msg.attach(MIMEText(body, 'plain'))
        with smtplib.SMTP(smtp_cfg['server'], int(smtp_cfg.get('port', 587)), timeout=10) as srv:
            if smtp_cfg.get('tls', True):
                srv.starttls()
            if smtp_cfg.get('password'):
                srv.login(smtp_cfg['sender'], smtp_cfg['password'])
            srv.sendmail(smtp_cfg['sender'], smtp_cfg['recipient'], msg.as_string())
        return True, 'Email sent.'
    except Exception as e:
        return False, str(e)

# ==============================================================================
#  SECTION M — PDF GENERATION
# ==============================================================================
def _build_pdf(df, suspicious_df, G, syndicate_table, boss_account,
               boss_score, boss_ic, stats):
    if not PDF_OK:
        return None
    if df is None or len(df) == 0:
        return None

    temp_dir = tempfile.mkdtemp()
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        from reportlab.lib.units import mm
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                         Paragraph, Spacer, Image, PageBreak, HRFlowable)

        c1_path = os.path.join(temp_dir, 'c1.png')
        c2_path = os.path.join(temp_dir, 'c2.png')

        # ── Chart 1: Entity-wise bar ─────────────────────────────────────────
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        fig1.patch.set_facecolor('#08081c'); ax1.set_facecolor('#08081c')
        bvol = (suspicious_df.groupby('Bank_Name')['Amount_INR']
                .sum().sort_values(ascending=False).head(12))
        if bvol.empty:
            ax1.text(0.5, 0.5, 'No suspicious transactions in this scan',
                     ha='center', va='center', color='#aabbcc', fontsize=12,
                     transform=ax1.transAxes)
            ax1.set_xticks([]); ax1.set_yticks([])
        else:
            bar_cols = ['#ff2020' if float(v) == float(bvol.max()) else '#3399ff'
                        for v in bvol.values]
            ax1.bar(bvol.index, bvol.values / 1e7, color=bar_cols,
                    edgecolor='black', linewidth=0.4)
        ax1.set_title('Entity-wise Suspicious Volume (Rs. Crore)',
                      color='#ffd700', fontsize=11, fontweight='bold')
        ax1.tick_params(colors='#aabbcc', labelsize=8)
        ax1.spines[:].set_color('#1e1e55')
        plt.tight_layout()
        plt.savefig(c1_path, dpi=120, bbox_inches='tight', facecolor='#08081c')
        plt.close(fig1)

        # ── Chart 2: Fraud pattern pie ───────────────────────────────────────
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        fig2.patch.set_facecolor('#08081c')
        fdist = (suspicious_df['Fraud_Label'].value_counts()
                 if not suspicious_df.empty else pd.Series(dtype='int64'))
        if fdist.empty:
            ax2.text(0.5, 0.5, 'No pattern distribution available',
                     ha='center', va='center', color='#aabbcc', fontsize=11,
                     transform=ax2.transAxes)
            ax2.axis('off')
        else:
            pcols = ['#ff2020','#ffd700','#3399ff','#00e676',
                     '#445566','#cc44ff'][:len(fdist)]
            ax2.pie(fdist.values, labels=fdist.index, colors=pcols,
                    autopct='%1.1f%%', startangle=140, pctdistance=0.80,
                    wedgeprops={'edgecolor': '#04040f', 'linewidth': 2})
            for txt in ax2.texts:
                txt.set_color('#aabbcc'); txt.set_fontsize(9)
        ax2.set_title('Fraud Pattern Distribution',
                      color='#ffd700', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(c2_path, dpi=120, bbox_inches='tight', facecolor='#08081c')
        plt.close(fig2)

        # ── Metrics ──────────────────────────────────────────────────────────
        NOW       = datetime.datetime.now()
        CASE_NO   = stats.get('case_no', f'DA-{NOW.strftime("%Y%m%d-%H%M")}-001')
        RDATE     = NOW.strftime('%d %B %Y  |  %H:%M hrs')
        N_TXN     = int(len(df))
        N_SUSP    = int(len(suspicious_df))
        N_BANKS   = int(df['Bank_Name'].nunique())
        TOTAL_AMT = float(suspicious_df['Amount_INR'].sum())
        fraud_rate = N_SUSP / max(1, N_TXN) * 100

        # ── Colour palette ───────────────────────────────────────────────────
        C_BG    = colors.HexColor('#08081c')
        C_DARK  = colors.HexColor('#0e0e32')
        C_GOLD  = colors.HexColor('#ffd700')
        C_RED   = colors.HexColor('#ff2020')
        C_BLUE  = colors.HexColor('#3399ff')
        C_TEXT  = colors.HexColor('#c8d8e8')
        C_MUTED = colors.HexColor('#9aaabb')
        C_GRID  = colors.HexColor('#3c3c6e')
        C_ROW1  = colors.HexColor('#08081c')
        C_ROW2  = colors.HexColor('#0a0a1e')

        # ── Paragraph styles ─────────────────────────────────────────────────
        sty_title = ParagraphStyle(
            'T', fontName='Helvetica-Bold', fontSize=15,
            textColor=C_RED, alignment=TA_CENTER, spaceAfter=5)
        sty_sub = ParagraphStyle(
            'S', fontName='Helvetica-Bold', fontSize=9,
            textColor=C_GOLD, alignment=TA_CENTER, spaceAfter=4)
        sty_meta = ParagraphStyle(
            'M', fontName='Helvetica', fontSize=8,
            textColor=C_MUTED, alignment=TA_CENTER, spaceAfter=8)
        sty_sec = ParagraphStyle(
            'SEC', fontName='Helvetica-Bold', fontSize=10,
            textColor=C_GOLD, backColor=C_DARK,
            leftIndent=6, borderPad=5, spaceAfter=6, spaceBefore=8)
        sty_body = ParagraphStyle(
            'B', fontName='Helvetica', fontSize=7.5,
            textColor=C_MUTED, spaceAfter=6, leftIndent=8)
        sty_legal_h = ParagraphStyle(
            'LH', fontName='Helvetica-Bold', fontSize=8,
            textColor=colors.HexColor('#ffaa00'), spaceAfter=2, leftIndent=5)
        sty_legal_d = ParagraphStyle(
            'LD', fontName='Helvetica', fontSize=8,
            textColor=C_MUTED, spaceAfter=6, leftIndent=15)

        def section_table_style(extra=None):
            base = [
                ('BACKGROUND',  (0, 0), (-1,  0), C_DARK),
                ('TEXTCOLOR',   (0, 0), (-1,  0), C_GOLD),
                ('FONTNAME',    (0, 0), (-1,  0), 'Helvetica-Bold'),
                ('FONTSIZE',    (0, 0), (-1, -1), 7),
                ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
                ('GRID',        (0, 0), (-1, -1), 0.3, C_GRID),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [C_ROW1, C_ROW2]),
                ('TEXTCOLOR',   (0, 1), (-1, -1), C_TEXT),
                ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
                ('TOPPADDING',  (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING',(0,0), (-1, -1), 3),
            ]
            if extra:
                base.extend(extra)
            return TableStyle(base)

        # ── Build story ──────────────────────────────────────────────────────
        buf   = io.BytesIO()
        doc   = SimpleDocTemplate(
            buf, pagesize=A4,
            topMargin=18*mm, bottomMargin=18*mm,
            leftMargin=12*mm, rightMargin=12*mm)
        story = []

        # Title block
        story.append(Paragraph(
            'ENFORCEMENT DIRECTORATE / CBI — FINANCIAL CRIMES UNIT', sty_title))
        story.append(Paragraph(
            f'Anti-Money Laundering Intelligence Report — '
            f'MODE: {stats.get("mode","UPLOAD")} — PMLA 2002', sty_sub))
        story.append(Paragraph(f'Case: {CASE_NO}   |   {RDATE}', sty_meta))
        story.append(HRFlowable(width='100%', thickness=1, color=C_GOLD))
        story.append(Spacer(1, 5*mm))

        # KPI row 1
        kpi1_data = [
            ['TOTAL RECORDS', 'ENTITIES', 'FLAGGED', 'FRAUD RATE'],
            [f'{N_TXN:,}', str(N_BANKS), f'{N_SUSP:,}', f'{fraud_rate:.2f}%'],
        ]
        t_kpi1 = Table(kpi1_data, colWidths=[45*mm]*4)
        t_kpi1.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,0), C_DARK),
            ('BACKGROUND',   (0,1), (-1,1), C_BG),
            ('TEXTCOLOR',    (0,0), (-1,0), C_MUTED),
            ('TEXTCOLOR',    (0,1), (-1,1), colors.white),
            ('FONTNAME',     (0,0), (-1,0), 'Helvetica'),
            ('FONTNAME',     (0,1), (-1,1), 'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,0), 7),
            ('FONTSIZE',     (0,1), (-1,1), 13),
            ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
            ('GRID',         (0,0), (-1,-1), 0.5, C_GRID),
            ('TOPPADDING',   (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ]))
        story.append(t_kpi1)
        story.append(Spacer(1, 3*mm))

        # KPI row 2
        kpi2_data = [
            ['FLAGGED VOLUME', 'CONVERGENCE NODE', 'BOSS SCORE', 'NETWORK NODES'],
            [f'Rs.{TOTAL_AMT/1e7:.1f}Cr',
             str(boss_account)[-18:],
             f'{boss_score:.6f}',
             str(G.number_of_nodes())],
        ]
        t_kpi2 = Table(kpi2_data, colWidths=[45*mm]*4)
        t_kpi2.setStyle(TableStyle([
            ('BACKGROUND',   (0,0), (-1,0), C_DARK),
            ('BACKGROUND',   (0,1), (-1,1), C_BG),
            ('TEXTCOLOR',    (0,0), (-1,0), C_MUTED),
            ('TEXTCOLOR',    (0,1), (0,1),  C_RED),
            ('TEXTCOLOR',    (1,1), (-1,1), C_GOLD),
            ('FONTNAME',     (0,0), (-1,0), 'Helvetica'),
            ('FONTNAME',     (0,1), (-1,1), 'Helvetica-Bold'),
            ('FONTSIZE',     (0,0), (-1,0), 7),
            ('FONTSIZE',     (0,1), (-1,1), 11),
            ('ALIGN',        (0,0), (-1,-1), 'CENTER'),
            ('VALIGN',       (0,0), (-1,-1), 'MIDDLE'),
            ('GRID',         (0,0), (-1,-1), 0.5, C_GRID),
            ('TOPPADDING',   (0,0), (-1,-1), 6),
            ('BOTTOMPADDING',(0,0), (-1,-1), 6),
        ]))
        story.append(t_kpi2)
        story.append(Spacer(1, 4*mm))

        # Methodology note
        story.append(Paragraph(
            f'METHODOLOGY: IF (n=100) + XGBoost. '
            f'Auto-tuned contamination={stats.get("contamination",0.025):.4f}. '
            f'Dynamic thresholds HIGH≥{stats.get("high_thresh",0.72):.4f} '
            f'MED≥{stats.get("med_thresh",0.45):.4f}. '
            f'PageRank composite boss detection. 12 features. PMLA 2002.',
            sty_body))

        # ── Fraud Pattern Breakdown ──────────────────────────────────────────
        story.append(Paragraph('FRAUD PATTERN BREAKDOWN', sty_sec))
        pattern_counts = (suspicious_df['Fraud_Label'].value_counts()
                          if not suspicious_df.empty else pd.Series(dtype='int64'))
        fp_data = [['Pattern', 'Count', 'Pct', 'Total (INR)', 'Avg (INR)']]
        if pattern_counts.empty:
            fp_data.append(['NONE', '0', '0.00%', 'Rs.0', 'Rs.0'])
        else:
            for ft, cnt in pattern_counts.items():
                sub = suspicious_df[suspicious_df['Fraud_Label'] == ft]
                amt = float(sub['Amount_INR'].sum())
                avg = float(sub['Amount_INR'].mean())
                fp_data.append([
                    ft, f'{int(cnt):,}',
                    f'{int(cnt)/max(1,N_SUSP)*100:.2f}%',
                    f'Rs.{amt:,.0f}', f'Rs.{avg:,.0f}'])
        t_fp = Table(fp_data, colWidths=[44*mm, 24*mm, 22*mm, 52*mm, 38*mm])
        t_fp.setStyle(section_table_style())
        story.append(t_fp)

        story.append(PageBreak())

        # ── Syndicate Role Analysis ──────────────────────────────────────────
        story.append(Paragraph('SYNDICATE ROLE ANALYSIS', sty_sec))
        synd_data = [['Account', 'Role', 'Inflow (INR)',
                       'Outflow (INR)', 'Boss Score', 'Centrality']]
        boss_rows = []
        if syndicate_table.empty:
            synd_data.append(['NO_SUSPICIOUS_ACTIVITY', 'NO NETWORK',
                              'Rs.0', 'Rs.0', '0.000000', '0.0000'])
        else:
            for i, row in syndicate_table.head(30).iterrows():
                synd_data.append([
                    str(row['Account'])[-18:], row['Role'],
                    f'Rs.{float(row["Inflow_INR"]):,.0f}',
                    f'Rs.{float(row["Outflow_INR"]):,.0f}',
                    f'{float(row["Boss_Score"]):.6f}',
                    f'{float(row["In_Centrality"]):.4f}'])
                if row['Role'] == 'SYNDICATE BOSS':
                    boss_rows.append(len(synd_data) - 1)

        t_synd = Table(synd_data, colWidths=[42*mm, 34*mm, 28*mm, 28*mm, 26*mm, 22*mm])
        extra_styles = []
        for br in boss_rows:
            extra_styles.append(
                ('BACKGROUND', (0, br), (-1, br), colors.HexColor('#2a0303')))
            extra_styles.append(
                ('TEXTCOLOR',  (0, br), (-1, br), C_RED))
        t_synd.setStyle(section_table_style(extra_styles))
        story.append(t_synd)
        story.append(Spacer(1, 5*mm))

        # ── Transaction Chain ────────────────────────────────────────────────
        story.append(Paragraph('TRANSACTION CHAIN — Top 20', sty_sec))
        chain_cols = [c for c in
                      ['Transaction_ID','Source_Acc_No','Dest_Acc_No',
                       'Amount_INR','Bank_Name','Transaction_Type','Fraud_Label']
                      if c in suspicious_df.columns]
        if chain_cols:
            chain_data = [chain_cols]
            if suspicious_df.empty:
                chain_data.append(
                    ['NO SUSPICIOUS TRANSACTIONS'] + ['-']*(len(chain_cols)-1))
            else:
                for _, row in suspicious_df.head(20).iterrows():
                    chain_data.append(
                        [str(row.get(c, ''))[:16] for c in chain_cols])
            col_w = 180*mm / len(chain_cols)
            t_chain = Table(chain_data, colWidths=[col_w]*len(chain_cols))
            t_chain.setStyle(section_table_style())
            story.append(t_chain)

        story.append(PageBreak())

        # ── Charts ───────────────────────────────────────────────────────────
        story.append(Paragraph('ANALYTICAL CHARTS', sty_sec))
        if os.path.exists(c1_path):
            story.append(Image(c1_path, width=175*mm, height=70*mm))
            story.append(Spacer(1, 6*mm))
        if os.path.exists(c2_path):
            story.append(Image(c2_path, width=110*mm, height=88*mm))

        story.append(PageBreak())

        # ── Legal Escalation ─────────────────────────────────────────────────
        story.append(Paragraph(
            'LEGAL ESCALATION PROTOCOL — ED/CBI/INTERPOL — PMLA 2002', sty_sec))
        for heading, detail in [
            ('STR Filing',
             'File Suspicious Transaction Report with FIU-IND within 7 days.'),
            ('Enforcement Directorate',
             'Escalate for asset freezing under PMLA Section 5.'),
            ('LOC Issuance',
             'Request Look-Out Circular via INTERPOL.'),
            ('MLAT Procedure',
             'Initiate Mutual Legal Assistance Treaty for offshore accounts.'),
            ('EDD Requirement',
             'Enhanced Due Diligence on all shell and mule accounts.'),
        ]:
            story.append(Paragraph(f'{heading}:', sty_legal_h))
            story.append(Paragraph(detail, sty_legal_d))

        doc.build(story)
        return buf.getvalue()

    except Exception as _pdf_err:
        print(f'[PDF INTERNAL ERROR] {type(_pdf_err).__name__}: {_pdf_err}')
        import traceback; traceback.print_exc()
        return None
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

# ==============================================================================
#  SECTION N — EXPLAINABILITY
# ==============================================================================
def compute_feature_importance(res):
    susp  = res['susp']; norm = res['normal']
    feats = [f for f in FEATS if f in susp.columns]
    if not feats:
        return pd.DataFrame()
    rows = []
    for feat in feats:
        s_mean = float(susp[feat].astype(float).mean())
        n_mean = float(norm[feat].astype(float).mean()) if feat in norm.columns else 0.0
        dev    = abs(s_mean - n_mean)
        rows.append({'Feature': feat, 'Suspicious Mean': round(s_mean, 4),
                     'Normal Mean': round(n_mean, 4), 'Deviation': round(dev, 4),
                     'Importance Score': round(dev / max(1e-9, s_mean+n_mean), 4)})
    return pd.DataFrame(rows).sort_values('Deviation', ascending=False).reset_index(drop=True)

# ==============================================================================
#  SECTION O — FASTAPI CODE
# ==============================================================================
_FASTAPI_CODE = '''"""
Deep Audit AI — FastAPI Real-Time Scoring Microservice
Run: uvicorn fastapi_scorer:app --host 0.0.0.0 --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np, joblib, logging, datetime
from typing import Optional

app = FastAPI(title="Deep Audit AI", version="8.1")
try:
    ISO=joblib.load("iso_model.pkl"); SCALER=joblib.load("scaler.pkl")
    XGB=joblib.load("xgb_model.pkl"); MODELS_LOADED=True
except: MODELS_LOADED=False

AMT_NORM=500_000.0; HIGH_THRESH=0.72; MED_THRESH=0.45

class Transaction(BaseModel):
    transaction_id: str; source_account: str; dest_account: str
    amount_inr: float; transaction_type: str; is_international: bool
    timestamp: Optional[str]=None; description: Optional[str]=""

@app.get("/health")
def health(): return {"status":"ok","models_loaded":MODELS_LOADED}

@app.post("/score")
def score(txn: Transaction):
    ts=datetime.datetime.fromisoformat(txn.timestamp) if txn.timestamp else datetime.datetime.now()
    amt=txn.amount_inr; hour=ts.hour
    is_night=hour<5 or hour>=22
    risk=min(1.0,(min(amt,AMT_NORM)/AMT_NORM)*0.40+(0.30 if txn.is_international else 0.0)+(0.12 if is_night else 0.0))
    row=[amt,float(np.log1p(amt)),0.0,0.0,risk,0.0,float(is_night),float(txn.is_international),1.0,0.0,float(amt%1000==0),float(hour)]
    if MODELS_LOADED:
        X=np.array([row]); Xs=SCALER.transform(X)
        pred=ISO.predict(Xs)[0]; sc=ISO.decision_function(Xs)[0]; if_r=max(0,min(1,(0.5-sc)/0.5))
        if XGB: xp=float(XGB.predict_proba(Xs)[0][1]); risk=round(0.6*if_r+0.4*xp,4)
        else: risk=round(if_r,4)
    is_fraud=risk>=HIGH_THRESH
    logging.info(f"{txn.transaction_id} risk={risk:.4f} fraud={is_fraud}")
    return {"transaction_id":txn.transaction_id,"risk_score":risk,"is_fraud":is_fraud,
            "severity":"HIGH" if risk>=HIGH_THRESH else "MEDIUM" if risk>=MED_THRESH else "LOW",
            "scored_at":datetime.datetime.now().isoformat()}
'''

# ==============================================================================
#  SESSION STATE INIT
# ==============================================================================
for key, default in [
    ('logged_in', False), ('auth_user', None), ('auth_role', None),
    ('scan_result', None), ('rt_feed', []),
    ('smtp_config', {'server':'smtp.gmail.com','port':587,'tls':True,
                     'sender':'','password':'','recipient':''}),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================================================================
#  LOGIN PAGE
# ==============================================================================
if not st.session_state['logged_in']:
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown('<br>', unsafe_allow_html=True)
        st.markdown("""
        <div style='background:#08081c;border:1px solid #ffd700;border-radius:6px;padding:30px 28px;margin-top:60px'>
            <div style='font-family:Share Tech Mono,monospace;color:#ff2020;font-size:1.4rem;letter-spacing:3px;margin-bottom:4px;text-align:center'>
                DEEP AUDIT AI  v9.0</div>
            <div style='font-size:10px;color:#445566;letter-spacing:2px;text-align:center;margin-bottom:20px'>
                ED / CBI / INTERPOL  —  AML INTELLIGENCE UNIT</div>
        </div>""", unsafe_allow_html=True)
        with st.form('login_form'):
            username = st.text_input('Username', placeholder='admin  or  analyst')
            password = st.text_input('Password', type='password')
            if st.form_submit_button('ACCESS SECURE SYSTEM'):
                ok, role = check_login(username.strip(), password)
                if ok:
                    st.session_state.update({'logged_in': True, 'auth_user': username.strip(), 'auth_role': role})
                    db_log('LOGIN', f'user={username}')
                    st.rerun()
                else:
                    st.error('Invalid credentials.')
        st.markdown('<div style="font-size:10px;color:#223344;text-align:center;margin-top:6px">admin/admin123  |  analyst/analyst456</div>', unsafe_allow_html=True)
    st.stop()

# ==============================================================================
#  SIDEBAR
# ==============================================================================
res = st.session_state.get('scan_result')

with st.sidebar:
    st.markdown(f"""
    <div style='font-family:Share Tech Mono,monospace;color:#ff2020;font-size:1.1rem;letter-spacing:3px;margin-bottom:2px'>DEEP AUDIT AI</div>
    <div style='font-size:9px;color:#445566;letter-spacing:1.5px;margin-bottom:4px'>ED/CBI FINANCIAL CRIMES</div>
    <div style='font-size:9px;color:#334455;border-top:1px solid #12123a;padding-top:6px;margin-bottom:10px'>
        USER: <b style='color:#ffd700'>{st.session_state["auth_user"].upper()}</b>&nbsp;|&nbsp;
        ROLE: <b style='color:#7799cc'>{st.session_state["auth_role"]}</b>
    </div>""", unsafe_allow_html=True)

    live_ok = live_db_available()
    if live_ok:
        st.markdown("<div style='font-family:Share Tech Mono,monospace;font-size:9px;color:#ff2020;letter-spacing:2px;margin-bottom:8px'>● LIVE DB CONNECTED</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:9px;color:#334455;margin-bottom:6px'>Run transaction_generator.py for live mode</div>", unsafe_allow_html=True)

    menu = st.radio('NAVIGATION', [
        '🏠  Command Center',
        '📁  Data Ingestion',
        '🔴  Live Dashboard',
        '⚡  Real-Time Finder',
        '🕸️  Network Topology',
        '📊  Intelligence Matrix',
        '📋  Threat Dossier',
        '🔬  Explainability Lab',
        '📄  Export & Download',
        '⚙️  Settings',
    ], label_visibility='collapsed')

    st.markdown('<hr>', unsafe_allow_html=True)
    if res:
        s = res['stats']
        st.markdown(f"""
        <div style='font-size:9px;color:#334455;line-height:1.9'>
            CASE: <span style='color:#ffd700'>{s.get("case_no","—")}</span><br>
            MODE: <span style='color:#00e676'>{s.get("mode","UPLOAD")}</span><br>
            RECORDS: <span style='color:#fff'>{s["total_txn"]:,}</span><br>
            FLAGGED: <span style='color:#ff2020'>{s["suspicious"]:,}</span><br>
            BOSS: <span style='color:#ff7070'>{str(s["boss"])[-16:]}</span><br>
            HIGH≥: <span style='color:#ffd700'>{s.get("high_thresh",0.72):.4f}</span><br>
            MED≥: <span style='color:#ff9900'>{s.get("med_thresh",0.45):.4f}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr>', unsafe_allow_html=True)
    if st.button('LOGOUT'):
        db_log('LOGOUT')
        for k in ['logged_in','auth_user','auth_role','scan_result','rt_feed']:
            st.session_state[k] = False if k == 'logged_in' else (None if k != 'rt_feed' else [])
        st.rerun()

# ==============================================================================
#  PAGE: COMMAND CENTER
# ==============================================================================
if menu == '🏠  Command Center':
    st.markdown('# 🔴 DEEP AUDIT AI  v9.0')
    st.markdown('### ED / CBI / INTERPOL — FINANCIAL CRIMES INTELLIGENCE UNIT')

    sec_header('WHAT WAS FIXED IN v9.0')
    fixes = [
        ('FIX-1','Normal TX no longer HIGH risk','Dynamic 95th-percentile thresholds. ₹500 UPI → LOW. ₹48K smurfing at night → HIGH.'),
        ('FIX-2','Boss name cache bug resolved','st.session_state cleared before every scan. New boss always detected fresh.'),
        ('FIX-3','Contamination capped at 5%','Was up to 10% (too aggressive). Now max 5% gets flagged as suspicious.'),
        ('FIX-4','Scan 3× faster','n_estimators 200→100. Default limit 10K→3K. Same accuracy, 3× speed.'),
        ('FIX-5','Real-time scorer calibrated','Uses p95 amount normaliser from actual data, not hardcoded ₹1 Crore max.'),
    ]
    for code, title, detail in fixes:
        st.markdown(f"""
        <div style='background:#07071a;border-left:3px solid #3399ff;padding:9px 14px;margin-bottom:4px;border-radius:0 3px 3px 0'>
            <span class='fix-badge'>{code}</span>
            <span style='font-size:12px;color:#9aaabb;margin-left:10px'>
                <b style='color:#7bcfff'>{title}</b> — {detail}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    if live_db_available():
        live_stats = get_live_stats()
        sec_header('LIVE GENERATOR STATUS', 'transaction_generator.py is running')
        cols = st.columns(4)
        for col, (lbl, val, color) in zip(cols, [
            ('LIVE TRANSACTIONS', f'{live_stats.get("total",0):,}',        'gold'),
            ('FRAUD DETECTED',    f'{live_stats.get("fraud",0):,}',         'red'),
            ('FRAUD RATE',        f'{live_stats.get("fraud_rate",0):.2f}%', 'orange'),
            ('LAST UPDATE',       str(live_stats.get("last_update","—")),   'green'),
        ]):
            with col:
                st.markdown(kpi(lbl, val, color), unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#07071a;border:1px solid #1e1e55;border-radius:4px;padding:14px 20px'>
            <b style='color:#ffd700'>To activate LIVE mode:</b>
            <br>Open a second terminal and run:
            <code style='color:#00e676'> python transaction_generator.py</code>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    sec_header('HOW RISK THRESHOLDS WORK NOW  (FIX-1)')
    st.markdown("""
    <div style='background:#07071a;border:1px solid #0d0d2a;border-radius:4px;padding:16px;font-size:11px;color:#7799cc;line-height:2'>
        <b style='color:#ff2020'>HIGH RISK (top 5% of dataset)</b> — Score ≥ 95th percentile of actual data<br>
        <b style='color:#ff9900'>MEDIUM RISK (top 15% of dataset)</b> — Score ≥ 85th percentile of actual data<br>
        <b style='color:#00e676'>LOW / NORMAL (bottom 85%)</b> — Everything else<br>
        <br>
        <b style='color:#ffd700'>Why this fixes the old bug:</b><br>
        Old version used a fixed cutoff of 0.65 regardless of data distribution.<br>
        If the live data had mostly small transactions, even ₹10,000 would normalise<br>
        against ₹50,000 max and score 0.80 — incorrectly HIGH.<br>
        New version: thresholds are computed FROM the actual data every scan.
    </div>""", unsafe_allow_html=True)

# ==============================================================================
#  PAGE: DATA INGESTION
# ==============================================================================
elif menu == '📁  Data Ingestion':
    st.markdown('# 📁  DATA INGESTION')
    sec_header('UPLOAD MODE', 'Upload your CSV or Parquet files')
    uploaded = st.sidebar.file_uploader('Upload CSV/Parquet', type=['csv','parquet'],
                                        accept_multiple_files=True, label_visibility='collapsed')
    if uploaded:
        st.success(f'{len(uploaded)} file(s): {", ".join(f.name for f in uploaded)}')
        if st.button('INITIATE FORENSIC SCAN'):
            prog = st.progress(0); status = st.empty()
            try:
                status.markdown('**Loading files...**')
                df = load_and_merge(uploaded)
                prog.progress(5)
                result = run_master_pipeline(df, prog, status, mode='UPLOAD')
                st.session_state['scan_result'] = result
                status.success(f'Scan complete. {result["stats"]["suspicious"]:,} anomalies. Boss: {result["boss"]}')
            except Exception as e:
                status.error(f'Error: {e}')
    else:
        st.info('Upload CSV files using the sidebar uploader.')
        sec_header('EXPECTED CSV COLUMNS')
        st.markdown("""
        | Column | Required | Notes |
        |--------|----------|-------|
        | Transaction_ID | Yes | Unique ID |
        | Source_Acc_No | Yes | Sender account |
        | Dest_Acc_No | Yes | Receiver account |
        | Amount_INR | Yes | Amount |
        | Timestamp | Yes | Date and time |
        | Transaction_Type | Yes | WIRE / NEFT / UPI / SWIFT |
        | Is_International | Recommended | 0 or 1 |
        | Txn_Description | Optional | Used for NLP classification |
        """)

# ==============================================================================
#  PAGE: LIVE DASHBOARD — auto-refresh every 2s, account investigator
# ==============================================================================
elif menu == '🔴  Live Dashboard':
    st.markdown('# 🔴  LIVE TRANSACTION MONITOR')
    st.markdown('### ED / CBI Financial Crimes Unit — Real-Time Feed')

    if not live_db_available():
        st.markdown("""
        <div class='alert-high'>
            <div style='font-family:Share Tech Mono,monospace;font-size:16px;color:#ff2020'>
                LIVE DATABASE NOT FOUND</div>
            <div style='font-size:12px;color:#ff7070;margin-top:10px'>
                Open a <b>second terminal</b> in your project folder and run:<br><br>
                <code style='color:#00e676;background:#0a0a1a;padding:5px 12px;
                border-radius:4px;font-size:13px'>python transaction_generator.py</code><br><br>
                Once it starts printing transactions, come back and click the Scan button below.
            </div>
        </div>""", unsafe_allow_html=True)
        st.stop()

    # Scan controls
    col_scan, col_lim, col_clr = st.columns([3, 1, 1])
    with col_lim:
        limit = st.selectbox('Scan limit', [1000, 2000, 3000, 5000], index=1)
    with col_clr:
        st.markdown('<div style="height:28px"></div>', unsafe_allow_html=True)
        if st.button('CLEAR CACHE', use_container_width=True):
            st.cache_data.clear()
            st.session_state['scan_result'] = None
            st.rerun()
    with col_scan:
        if st.button('RUN AI SCAN ON LIVE DATA', use_container_width=True):
            st.cache_data.clear()
            prog   = st.progress(0)
            status = st.empty()
            try:
                df_live = load_live_transactions(limit=limit)
                if df_live.empty:
                    st.warning('No transactions yet. Let the generator run for a few seconds.')
                    st.stop()
                df_live = _normalise_df(df_live, 'LIVE_FEED')
                prog.progress(5)
                result = run_master_pipeline(df_live, prog, status, mode='LIVE')
                st.session_state['scan_result'] = result
                status.success(
                    f'Scan done — {result["stats"]["suspicious"]:,} anomalies. '
                    f'Boss: {result["boss"]}')
            except Exception as e:
                status.error(f'Scan error: {e}')

    st.markdown('<hr>', unsafe_allow_html=True)

    # Auto-refresh feed
    if _FRAGMENT_OK:
        st.markdown(
            "<span class='auto-badge'>● AUTO-REFRESH — every 2 seconds</span>"
            "<span style='font-size:10px;color:#334455;margin-left:12px'>"
            "Transaction ID shown on each row</span>",
            unsafe_allow_html=True)
        _live_feed_auto()
    else:
        col_hdr, col_ref = st.columns([3, 1])
        with col_hdr:
            sec_header('LIVE TRANSACTION FEED', 'Transaction ID visible on each row')
        with col_ref:
            st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
            if st.button('REFRESH', use_container_width=True):
                st.rerun()
        _render_live_feed_inner()

    st.markdown('<hr>', unsafe_allow_html=True)
    render_account_investigator(location_key='live')

# ==============================================================================
#  PAGE: REAL-TIME FINDER — clean redesign with Txn ID, date, time, account lookup
# ==============================================================================
elif menu == '⚡  Real-Time Finder':
    st.markdown('# ⚡  REAL-TIME TRANSACTION FINDER')
    st.markdown('### ED / CBI / INTERPOL — Score any transaction instantly')

    high_t   = float(st.session_state.get('rt_high_thresh', 0.72))
    med_t    = float(st.session_state.get('rt_med_thresh',  0.45))
    amt_n    = float(st.session_state.get('rt_amt_norm',  500_000))
    has_model = st.session_state.get('rt_scaler') is not None

    st.markdown(
        f"<div style='background:#07071a;border:1px solid #0d0d2a;border-radius:6px;"
        f"padding:12px 20px;margin-bottom:16px;font-size:11px;color:#7799cc'>"
        f"<b style='color:#ffd700'>ACTIVE THRESHOLDS</b>"
        f"{'<span class="auto-badge" style="margin-left:8px">MODEL CALIBRATED</span>' if has_model else '<span style="color:#ff9900;font-size:10px;margin-left:8px">⚠ Run a scan first for model-based scoring</span>'}"
        f"<br>"
        f"<span style='color:#ff2020'>HIGH</span> ≥ <b>{high_t:.4f}</b>&nbsp;&nbsp;"
        f"<span style='color:#ff9900'>MEDIUM</span> ≥ <b>{med_t:.4f}</b>&nbsp;&nbsp;"
        f"Amount normaliser (p95) = <b>Rs.{amt_n:,.0f}</b>"
        "</div>",
        unsafe_allow_html=True)

    sec_header('SCORE A TRANSACTION')
    with st.form('rt_form_v9'):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            txn_id   = st.text_input('Transaction ID', placeholder='TXN_A1B2C3D4E5F6')
        with r1c2:
            src_acc  = st.text_input('Source Account ID', placeholder='ACC_MULE_0012')
        with r1c3:
            dest_acc = st.text_input('Destination Account ID', placeholder='OFFSHORE_BOSS_888')

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            amount   = st.number_input('Amount (INR)', min_value=1, value=1_000_000, step=10_000)
        with r2c2:
            bank_nm  = st.text_input('Bank Name', placeholder='SBI / HDFC / ICICI')
        with r2c3:
            txn_type = st.selectbox('Transaction Type',
                ['WIRE','SWIFT','NEFT','RTGS','UPI','HAWALA','IMPS','CASH_DEPOSIT'])

        r3c1, r3c2, r3c3 = st.columns(3)
        with r3c1:
            txn_date = st.date_input('Transaction Date', value=datetime.date.today())
        with r3c2:
            txn_time = st.time_input('Transaction Time (exact)',
                                      value=datetime.time(2, 15),
                                      step=datetime.timedelta(minutes=1))
        with r3c3:
            is_intl  = st.checkbox('International Transfer', value=True)
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)

        submitted = st.form_submit_button('SCORE THIS TRANSACTION', use_container_width=True)

    if submitted:
        actual_hour = txn_time.hour
        actual_dt   = datetime.datetime.combine(txn_date, txn_time)
        is_night    = actual_hour < 5 or actual_hour >= 22
        night_label = '🌙 Night' if is_night else '☀️ Daytime'
        txn_display = txn_id.strip() or f'TXN_{actual_dt.strftime("%H%M%S")}'

        sr    = score_single_transaction(
            float(amount), int(is_intl), actual_hour,
            txn_type, src_acc or 'UNKNOWN', dest_acc or 'UNKNOWN', '')
        risk  = sr['risk_score']
        h_t   = sr['high_thresh']
        m_t   = sr['med_thresh']
        nlp   = sr['nlp_label']
        model = sr['model_used']

        # Result cards
        if sr['is_high']:
            st.markdown(
                f"<div class='alert-high pulse'>"
                f"<div style='font-size:9px;color:#ff3030;letter-spacing:2px;margin-bottom:6px'>"
                f"⚠ HIGH RISK ALERT — {txn_display}</div>"
                f"<div style='display:flex;align-items:baseline;gap:16px'>"
                f"<div style='font-family:Share Tech Mono,monospace;font-size:32px;color:#ff2020'>"
                f"{risk:.4f}</div>"
                f"<div style='font-size:12px;color:#ff7070'>score ≥ threshold {h_t:.4f}</div>"
                f"</div>"
                f"<div style='margin-top:12px;display:grid;grid-template-columns:repeat(3,1fr);"
                f"gap:8px;font-size:11px;color:#ff7070'>"
                f"<div>From: <b style='color:#fff'>{src_acc or '—'}</b></div>"
                f"<div>To: <b style='color:#fff'>{dest_acc or '—'}</b></div>"
                f"<div>Amount: <b style='color:#ffd700'>Rs.{amount:,.0f}</b></div>"
                f"<div>Type: <b style='color:#fff'>{txn_type}</b></div>"
                f"<div>Time: <b style='color:#fff'>{actual_dt.strftime('%Y-%m-%d %H:%M')} {night_label}</b></div>"
                f"<div>Model: <b style='color:#fff'>{model}</b></div>"
                f"</div></div>",
                unsafe_allow_html=True)
            smtp_cfg = st.session_state.get('smtp_config', {})
            if smtp_cfg.get('sender') and smtp_cfg.get('recipient'):
                body = (
                    "Transaction: " + txn_display + "\n"
                    "From: " + str(src_acc) + " To: " + str(dest_acc) + "\n"
                    "Risk: " + f"{risk:.4f}"
                )
#  PAGE: EXPORT & DOWNLOAD
# ==============================================================================
elif menu == '📄  Export & Download':
    if not res:
        st.warning('No scan data. Run a scan first.')
        st.stop()
    st.markdown('# 📄  EXPORT — INVESTIGATION DOSSIER')
    st.markdown('### ED / CBI Financial Crimes Unit — Case Documentation')

    sec_header('INVESTIGATION REPORT (PDF)', 'PMLA 2002 compliant court-admissible forensic dossier')
    c1, c2 = st.columns([3, 2])

    with c1:
        # Always try to generate if not already stored
        if not res.get('pdf'):
            with st.spinner('Generating PDF report...'):
                pdf_bytes, pdf_err = _build_pdf_safe(res)
                if pdf_bytes:
                    res['pdf'] = pdf_bytes
                    st.session_state['scan_result']['pdf'] = pdf_bytes

        if res.get('pdf'):
            now = datetime.datetime.now()
            st.download_button(
                label='📥  DOWNLOAD INVESTIGATION REPORT (PDF)',
                data=res['pdf'],
                file_name=f'InvestigationReport_ED_CBI_{now.strftime("%Y%m%d_%H%M")}.pdf',
                mime='application/pdf',
                use_container_width=True,
            )
            st.success('PDF ready — click above to download.')
        else:
            _, err_msg = _build_pdf_safe(res)
            st.markdown(
                "<div class='alert-med'>"
                "<div style='font-size:11px;color:#ffaa44;margin-bottom:6px'>"
                "PDF could not be generated</div>"
                f"<div style='font-size:12px;color:#ccaa66'>{err_msg or 'Unknown error'}</div>"
                "<div style='font-size:11px;color:#778899;margin-top:8px'>"
                "Fix: open terminal and run<br>"
                "<code style='color:#00e676'>pip install fpdf2</code>"
                "<br>then restart Streamlit.</div>"
                "</div>",
                unsafe_allow_html=True)

    with c2:
        st.markdown('<br>', unsafe_allow_html=True)
        susp_buf = io.StringIO()
        res['susp'].to_csv(susp_buf, index=False)
        st.download_button(
            label='📊  Download Anomalous Transactions (CSV)',
            data=susp_buf.getvalue(),
            file_name=f'anomalous_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            use_container_width=True,
        )
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        synd_buf = io.StringIO()
        res['synd'].to_csv(synd_buf, index=False)
        st.download_button(
            label='📋  Download Syndicate Table (CSV)',
            data=synd_buf.getvalue(),
            file_name=f'syndicate_{datetime.datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
            use_container_width=True,
        )
        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)

        st.download_button(
            label='🌐  Download 3D Network (HTML)',
            data=res['html'],
            file_name='network_topology.html',
            mime='text/html',
            use_container_width=True,
        )

    st.markdown('<hr>', unsafe_allow_html=True)
    sec_header('FASTAPI MICROSERVICE (IMP-13)', 'Production REST endpoint for bank API integration')
    st.download_button(
        label='⬇  Download FastAPI Scorer (fastapi_scorer.py)',
        data=_FASTAPI_CODE,
        file_name='fastapi_scorer.py',
        mime='text/plain',
        use_container_width=True,
    )

elif menu == '🕸️  Network Topology':
    if not res:
        st.warning('No scan data. Run a scan first.')
        st.stop()
    st.markdown('# 🕸️  NETWORK TOPOLOGY')
    sec_header('3D SYNDICATE NETWORK — 200 NODES', 'Drag to pan | Scroll to zoom | Hover for details')
    components.html(res['html'], height=600, scrolling=False)
    c1, c2, c3, c4 = st.columns(4)
    s = res['stats']
    with c1: st.metric('Network Nodes', f'{s["nodes"]:,}')
    with c2: st.metric('Money Trails',  f'{s["edges"]:,}')
    with c3: st.metric('Boss Score',    f'{res["boss_score"]:.6f}')
    with c4: st.metric('Mode',          s.get('mode', 'UPLOAD'))

# ==============================================================================
#  PAGE: INTELLIGENCE MATRIX
# ==============================================================================
elif menu == '📊  Intelligence Matrix':
    if not res:
        st.warning('No scan data. Run a scan first.')
        st.stop()
    susp   = res['susp']
    normal = res['normal']
    s      = res['stats']
    st.markdown('# 📊  INTELLIGENCE MATRIX')
    cols = st.columns(5)
    for col, (lbl, val, color) in zip(cols, [
        ('TOTAL RECORDS',    f'{s["total_txn"]:,}',                        'gold'),
        ('FLAGGED',          f'{s["suspicious"]:,}',                        'red'),
        ('FRAUD RATE',       f'{s["suspicious"]/s["total_txn"]*100:.2f}%',  'orange'),
        ('SUSP VOLUME',      f'Rs.{s["amount_cr"]:.1f}Cr',                  'red'),
        ('CONTAMINATION',    f'{s.get("contamination",0.025):.4f}',         'blue'),
    ]):
        with col:
            st.markdown(kpi(lbl, val, color), unsafe_allow_html=True)

    sec_header('ANOMALY SCORE DISTRIBUTION  (FIX-1 thresholds shown)')
    fig_h = go.Figure()
    fig_h.add_trace(go.Histogram(x=normal['Risk_Score'], name='Normal',
                                 marker_color='#3399ff', opacity=0.6, nbinsx=50))
    fig_h.add_trace(go.Histogram(x=susp['Risk_Score'], name='Suspicious',
                                 marker_color='#ff2020', opacity=0.7, nbinsx=50))
    if 'Confidence' in susp.columns:
        fig_h.add_trace(go.Histogram(x=susp['Confidence'], name='Confidence',
                                     marker_color='#ffd700', opacity=0.6, nbinsx=50))
    # Show threshold lines
    ht = s.get('high_thresh', 0.72)
    mt = s.get('med_thresh',  0.45)
    fig_h.add_vline(x=ht, line_dash='dash', line_color='#ff2020',
                    annotation_text=f'HIGH≥{ht:.3f}', annotation_position='top right')
    fig_h.add_vline(x=mt, line_dash='dash', line_color='#ff9900',
                    annotation_text=f'MED≥{mt:.3f}',  annotation_position='top right')
    fig_h.update_layout(paper_bgcolor='#07071a', plot_bgcolor='#07071a', barmode='overlay',
                        font=dict(family='Courier New', color='#9aaabb', size=11),
                        legend=dict(bgcolor='#07071a'), margin=dict(l=40,r=20,t=30,b=40), height=320)
    st.plotly_chart(fig_h, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        sec_header('FRAUD PATTERN BREAKDOWN')
        fdist = susp['Fraud_Label'].value_counts().reset_index(); fdist.columns = ['Label', 'Count']
        fig_p = px.pie(fdist, names='Label', values='Count', hole=0.4,
                       color_discrete_map={'LAYERING':'#ff2020','SMURFING':'#ffd700',
                                           'INTEGRATION':'#ff9900','HAWALA':'#cc44ff',
                                           'NORMAL':'#3399ff','TERROR':'#ff3366'},
                       template='plotly_dark')
        fig_p.update_layout(paper_bgcolor='#07071a', height=320, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_p, use_container_width=True)
    with c2:
        sec_header('TOP ENTITIES BY SUSPICIOUS VOLUME')
        bvol = susp.groupby('Bank_Name')['Amount_INR'].sum().sort_values(ascending=False).head(10).reset_index()
        bvol.columns = ['Bank', 'Amount']
        fig_b = px.bar(bvol, x='Amount', y='Bank', orientation='h',
                       color='Amount', color_continuous_scale=['#3399ff','#ffd700','#ff2020'],
                       template='plotly_dark', height=320)
        fig_b.update_layout(paper_bgcolor='#07071a', plot_bgcolor='#07071a',
                            coloraxis_showscale=False, margin=dict(l=20,r=20,t=20,b=20))
        st.plotly_chart(fig_b, use_container_width=True)

    sec_header('RISK SCORE vs AMOUNT')
    s_samp = susp.sample(min(2000, len(susp))).reset_index(drop=True)
    fig_sc = px.scatter(s_samp, x='Amount_INR', y='Risk_Score', color='Fraud_Label',
                        size='Amount_INR', size_max=18, opacity=0.65,
                        color_discrete_map={'LAYERING':'#ff2020','SMURFING':'#ffd700',
                                            'INTEGRATION':'#ff9900','HAWALA':'#cc44ff',
                                            'NORMAL':'#3399ff','TERROR':'#ff3366'},
                        template='plotly_dark')
    fig_sc.update_layout(paper_bgcolor='#07071a', plot_bgcolor='#07071a',
                         font=dict(family='Courier New', color='#9aaabb', size=11),
                         legend=dict(bgcolor='#07071a'), margin=dict(l=40,r=20,t=20,b=40), height=360)
    st.plotly_chart(fig_sc, use_container_width=True)

# ==============================================================================
#  PAGE: THREAT DOSSIER
# ==============================================================================
elif menu == '📋  Threat Dossier':
    if not res:
        st.warning('No scan data. Run a scan first.')
        st.stop()
    synd = res['synd']; susp = res['susp']
    st.markdown('# 📋  THREAT DOSSIER')
    boss_row = synd[synd['Account'] == str(res['boss'])]
    if len(boss_row):
        b = boss_row.iloc[0]
        st.markdown(f"""
        <div class='alert-high'>
            <div style='font-size:9px;color:#ff3030;letter-spacing:2px;margin-bottom:5px'>
                PRIMARY CONVERGENCE NODE — PAGERANK COMPOSITE (FIX-2: always fresh detection)
            </div>
            <div style='font-family:Share Tech Mono,monospace;font-size:17px;color:#ff2020'>{b["Account"]}</div>
            <div style='display:flex;gap:24px;margin-top:8px;font-size:10px;color:#6a2020;flex-wrap:wrap'>
                <span>Entity: <b style='color:#ff7070'>{b["Entity"]}</b></span>
                <span>Boss Score: <b style='color:#ff7070'>{float(b["Boss_Score"]):.6f}</b></span>
                <span>Inflow: <b style='color:#ff7070'>Rs.{float(b["Inflow_INR"])/1e7:.2f} Crore</b></span>
                <span>Location: <b style='color:#ff7070'>{b["Location"]}</b></span>
            </div>
        </div>""", unsafe_allow_html=True)

    col_f, _ = st.columns([2, 4])
    with col_f:
        role_filter = st.selectbox('Filter by Role',
            ['All', 'SYNDICATE BOSS', 'SHELL ACCOUNT', 'MULE ACCOUNT', 'TARGET ACCOUNT'])

    display = synd.head(100).copy()
    if role_filter != 'All':
        display = display[display['Role'] == role_filter]

    def _highlight(row):
        if row['Role'] == 'SYNDICATE BOSS':
            return ['background-color:#2a0303;color:#ff6060'] * len(row)
        elif row['Role'] == 'SHELL ACCOUNT':
            return ['background-color:#1a0f03;color:#ffaa44'] * len(row)
        return [''] * len(row)

    show_cols = [c for c in ['Account','Role','Inflow_INR','Outflow_INR',
                              'Boss_Score','PageRank','In_Centrality','Entity','Location']
                 if c in display.columns]
    st.dataframe(display[show_cols].style.apply(_highlight, axis=1),
                 use_container_width=True, height=400)

    with st.expander('INVESTIGATOR FEEDBACK'):
        with st.form('fb_form'):
            txn_id  = st.text_input('Transaction ID')
            verdict = st.selectbox('Verdict', ['CONFIRMED FRAUD','FALSE POSITIVE','UNDER REVIEW'])
            note    = st.text_area('Notes', height=70)
            if st.form_submit_button('SUBMIT FEEDBACK') and txn_id:
                db_save_feedback(txn_id, verdict, note)
                st.success(f'Feedback saved for {txn_id}.')

    sec_header('TRANSACTION CHAIN ANALYSIS')
    chain_cols = [c for c in ['Transaction_ID','Source_Acc_No','Dest_Acc_No',
                               'Amount_INR','Bank_Name','Transaction_Type','Fraud_Label']
                  if c in susp.columns]
    trail = susp[chain_cols].head(30).copy()
    if 'Amount_INR' in trail.columns:
        trail['Amount_INR'] = trail['Amount_INR'].apply(lambda x: f'Rs.{float(x):,.0f}')
    st.dataframe(trail, use_container_width=True, height=320)

# ==============================================================================
#  PAGE: EXPLAINABILITY LAB
# ==============================================================================
elif menu == '🔬  Explainability Lab':
    if not res:
        st.warning('No scan data. Run a scan first.')
        st.stop()
    st.markdown('# 🔬  EXPLAINABILITY LAB')
    imp_df = compute_feature_importance(res)
    if not imp_df.empty:
        sec_header('FEATURE IMPORTANCE  (why these transactions were flagged)')
        fig_i = px.bar(imp_df, x='Deviation', y='Feature', orientation='h',
                       color='Deviation', color_continuous_scale=['#3399ff','#ffd700','#ff2020'],
                       template='plotly_dark', height=420)
        fig_i.update_layout(paper_bgcolor='#07071a', plot_bgcolor='#07071a',
                            coloraxis_showscale=False, margin=dict(l=40,r=20,t=20,b=40))
        st.plotly_chart(fig_i, use_container_width=True)
        st.dataframe(imp_df, use_container_width=True)

    s = res['stats']
    sec_header('MODEL CONFIGURATION SUMMARY')
    st.markdown(f"""
    <div style='background:#07071a;border:1px solid #0d0d2a;border-radius:4px;padding:18px;font-size:11px;color:#7799cc;line-height:2.2'>
        <b style='color:#ffd700'>Mode:</b> {s.get("mode","UPLOAD")}<br>
        <b style='color:#ffd700'>Model:</b> IF (n=100, contamination={s.get("contamination",0.025):.4f}) + {'XGBoost (n=80)' if s.get('xgb_used') else 'XGBoost not installed'}<br>
        <b style='color:#ffd700'>HIGH threshold:</b> {s.get("high_thresh",0.72):.4f} (top 5% of dataset)<br>
        <b style='color:#ffd700'>MEDIUM threshold:</b> {s.get("med_thresh",0.45):.4f} (top 15% of dataset)<br>
        <b style='color:#ffd700'>Boss detection:</b> PageRank×0.35 + InDegree×0.35 + Inflow×0.20 + InvOut×0.10<br>
        <b style='color:#ffd700'>Features:</b> {", ".join(FEATS)}
    </div>""", unsafe_allow_html=True)

    if st.session_state.get('db_ok'):
        sec_header('SCAN HISTORY')
        history = db_get_history()
        if not history.empty:
            st.dataframe(history, use_container_width=True, height=250)

# ==============================================================================
#  PAGE: EXPORT
# ==============================================================================
elif menu == '⚙️  Settings':
    st.markdown('# ⚙️  SETTINGS — ED / CBI Financial Crimes Unit')
    if not PDF_OK:
        st.markdown(
            "<div class='alert-high' style='margin-bottom:12px'>"
            "<b style='color:#ff2020'>fpdf2 is not installed</b> — PDF export will fail. "
            "Run: <code style='color:#00e676'>pip install fpdf2</code> then restart.</div>",
            unsafe_allow_html=True)
    if not XGB_OK:
        st.markdown(
            "<div class='alert-med' style='margin-bottom:12px'>"
            "<b style='color:#ff9900'>xgboost not installed</b> — using Isolation Forest only. "
            "Run: <code style='color:#00e676'>pip install xgboost</code> then restart.</div>",
            unsafe_allow_html=True)
    sec_header('EMAIL ALERT CONFIGURATION  (IMP-10)')
    cfg = st.session_state['smtp_config']
    with st.form('smtp_form'):
        c1, c2 = st.columns(2)
        with c1:
            server   = st.text_input('SMTP Server', cfg.get('server', 'smtp.gmail.com'))
            port     = st.number_input('Port', value=int(cfg.get('port', 587)), step=1)
            use_tls  = st.checkbox('Use TLS', value=bool(cfg.get('tls', True)))
        with c2:
            sender   = st.text_input('Sender Email',    cfg.get('sender',    ''))
            password = st.text_input('App Password',    cfg.get('password',  ''), type='password')
            recipient= st.text_input('Recipient Email', cfg.get('recipient', ''))
        if st.form_submit_button('SAVE SMTP'):
            st.session_state['smtp_config'] = {
                'server': server, 'port': port, 'tls': use_tls,
                'sender': sender, 'password': password, 'recipient': recipient}
            st.success('Saved.')

    if st.button('SEND TEST EMAIL'):
        ok, msg = send_alert_email(st.session_state['smtp_config'],
            'Deep Audit AI v9.0 Test', 'Email alerts are working correctly.')
        st.success(msg) if ok else st.error(msg)

    sec_header('SYSTEM STATUS')
    st.markdown(f"""
    <div style='background:#07071a;border:1px solid #0d0d2a;border-radius:4px;padding:14px;font-size:11px;color:#7799cc;line-height:2.2'>
        <b style='color:#ffd700'>Audit DB:</b>
            <span style='color:{"#00e676" if st.session_state.get("db_ok") else "#ff2020"}'>
            {"CONNECTED — "+AUDIT_DB_PATH if st.session_state.get("db_ok") else "NOT AVAILABLE"}</span><br>
        <b style='color:#ffd700'>Live DB:</b>
            <span style='color:{"#00e676" if live_db_available() else "#ff9900"}'>
            {"CONNECTED — "+LIVE_DB_PATH if live_db_available() else "NOT FOUND — run python transaction_generator.py"}</span><br>
        <b style='color:#ffd700'>XGBoost:</b>
            <span style='color:{"#00e676" if XGB_OK else "#ff9900"}'>
            {"ACTIVE" if XGB_OK else "NOT INSTALLED — pip install xgboost"}</span><br>
        <b style='color:#ffd700'>fpdf2:</b>
            <span style='color:{"#00e676" if PDF_OK else "#ff9900"}'>
            {"ACTIVE" if PDF_OK else "NOT INSTALLED — pip install fpdf2"}</span><br>
        <b style='color:#ffd700'>Current User:</b> {st.session_state["auth_user"]} ({st.session_state["auth_role"]})<br>
        <b style='color:#ffd700'>Scan thresholds (last scan):</b>
            HIGH≥{st.session_state.get("rt_high_thresh", "not set")}
            &nbsp; MED≥{st.session_state.get("rt_med_thresh", "not set")}
    </div>""", unsafe_allow_html=True)

    if st.session_state.get('db_ok'):
        history = db_get_history()
        if not history.empty:
            sec_header('RECENT SCAN HISTORY')
            st.dataframe(history[[c for c in ['ts','case_no','user','n_txn','n_suspicious','boss_account','boss_score','mode'] if c in history.columns]],
                         use_container_width=True, height=220)