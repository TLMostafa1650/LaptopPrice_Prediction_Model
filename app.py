import streamlit as st
import pandas as pd
import numpy as np
import os, time

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Syne', sans-serif; }

/* Background */
.stApp { background-color: #0d0d0d; color: #f0ede6; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #141414;
    border-right: 1px solid #2a2a2a;
}
section[data-testid="stSidebar"] * { color: #c8c4bc !important; }
section[data-testid="stSidebar"] label { font-size: 0.75rem !important; letter-spacing: 0.05em; text-transform: uppercase; color: #888 !important; }

/* Inputs */
.stSelectbox > div > div, .stNumberInput > div > div > input, .stSlider {
    background: #1c1c1c !important;
    border: 1px solid #2e2e2e !important;
    border-radius: 6px !important;
    color: #f0ede6 !important;
    font-family: 'DM Mono', monospace !important;
}

/* Main title */
.main-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3rem;
    letter-spacing: -0.03em;
    color: #f0ede6;
    margin-bottom: 0;
    line-height: 1;
}
.main-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #555;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.4rem;
    margin-bottom: 2rem;
}

/* Price result box */
.price-box {
    background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
    border: 1px solid #333;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
}
.price-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #666;
    margin-bottom: 0.5rem;
}
.price-value {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 3.5rem;
    color: #e8d5a3;
    letter-spacing: -0.02em;
    line-height: 1;
}
.price-range {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #555;
    margin-top: 0.6rem;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin: 1rem 0; }
.metric-card {
    flex: 1;
    background: #141414;
    border: 1px solid #222;
    border-radius: 8px;
    padding: 1rem;
    text-align: center;
}
.metric-val { font-family: 'DM Mono', monospace; font-size: 1.2rem; font-weight: 500; color: #f0ede6; }
.metric-lbl { font-family: 'DM Mono', monospace; font-size: 0.65rem; color: #555; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }

/* Section headers */
.section-head {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #555;
    border-bottom: 1px solid #222;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* Status badge */
.status-ok { background: #1a2e1a; color: #5a9e5a; border: 1px solid #2a4a2a; border-radius: 20px; padding: 3px 12px; font-family: 'DM Mono', monospace; font-size: 0.7rem; display: inline-block; }
.status-wait { background: #2a2a1a; color: #9e9e5a; border: 1px solid #4a4a2a; border-radius: 20px; padding: 3px 12px; font-family: 'DM Mono', monospace; font-size: 0.7rem; display: inline-block; }

/* Button */
.stButton > button {
    background: #e8d5a3 !important;
    color: #0d0d0d !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.7rem 2rem !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stButton > button:hover { background: #f5e4b5 !important; transform: translateY(-1px); }

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Divider */
hr { border-color: #222 !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Feature definitions ─────────────────────────────────────────────────────────
NUM_COLS = ['screen_size','ram_gb','storage_gb','year','weight_kg','battery_wh',
            'refresh_rate_hz','cpu_cores','touchscreen','backlit_keyboard',
            'fingerprint_reader','usb_ports','has_thunderbolt','review_count','rating']

CAT_COLS = ['brand','category','resolution','cpu','gpu','storage_type',
            'operating_system','color','country_of_origin','warranty']

# Known unique values from the dataset
CAT_OPTIONS = {
    'brand': ['Dell','HP','Lenovo','Apple','Asus','Acer','Microsoft','Razer','Samsung','MSI'],
    'category': ['Notebook','Gaming','Ultrabook','Business','2-in-1','Workstation','Chromebook'],
    'resolution': ['1920x1080','2560x1440','3840x2160','1366x768','2560x1600','1920x1200'],
    'cpu': ['Intel Core i5','Intel Core i7','Intel Core i9','AMD Ryzen 5','AMD Ryzen 7',
            'AMD Ryzen 9','Intel Core i3','Apple M1','Apple M2','Apple M3',
            'Intel Core Ultra 5','Intel Core Ultra 7'],
    'gpu': ['Integrated','NVIDIA GTX 1650','NVIDIA RTX 3050','NVIDIA RTX 3060',
            'NVIDIA RTX 3070','NVIDIA RTX 4060','NVIDIA RTX 4070','AMD Radeon RX 6600',
            'Apple M-series GPU','Intel Iris Xe'],
    'storage_type': ['SSD','HDD','NVMe','eMMC'],
    'operating_system': ['Windows 11','Windows 10','macOS','Ubuntu','Chrome OS','No OS'],
    'color': ['Black','Silver','White','Gray','Blue','Red'],
    'country_of_origin': ['China','Taiwan','South Korea','USA','Japan','Vietnam'],
    'warranty': ['1 year','2 years','3 years','No warranty'],
}

TARGET = 'price_usd'

# ── Model training ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model(csv_path):
    df = pd.read_csv(csv_path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, NUM_COLS),
        ('cat', cat_pipeline, CAT_COLS)
    ])

    model = Pipeline([
        ('preprocessor', preprocessor),
        ('pca', PCA(n_components=0.95)),
        ('svr', SVR(kernel='linear', C=10, epsilon=0.2))
    ])
    model.fit(X_train, y_train)

    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    y_pred = model.predict(X_test)
    metrics = {
        'r2':   round(r2_score(y_test, y_pred), 4),
        'mae':  round(mean_absolute_error(y_test, y_pred), 2),
        'rmse': round(mean_squared_error(y_test, y_pred)**0.5, 2),
        'n_train': len(X_train),
    }
    return model, metrics

# ── Header ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">Laptop Price<br>Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="main-sub">SVM Regression · Data Computation Project</div>', unsafe_allow_html=True)

# ── Dataset path ────────────────────────────────────────────────────────────────
DEFAULT_PATH = r'C:\Users\TLMO\Downloads\laptop_price_dataset.csv'

with st.expander("⚙️  Dataset path (click to change)", expanded=False):
    csv_path = st.text_input("CSV file path", value=DEFAULT_PATH, label_visibility="collapsed")

# ── Load model ──────────────────────────────────────────────────────────────────
model_ready = False
metrics = {}

if os.path.exists(csv_path):
    with st.spinner("Training model on your dataset…"):
        model, metrics = train_model(csv_path)
    model_ready = True
    st.markdown('<span class="status-ok">● model ready</span>', unsafe_allow_html=True)
else:
    st.markdown('<span class="status-wait">● awaiting dataset</span>', unsafe_allow_html=True)
    st.info(f"Place your CSV at the path above, or update the path.")

# ── Model metrics ───────────────────────────────────────────────────────────────
if model_ready:
    st.markdown('<div class="metric-row">'
        f'<div class="metric-card"><div class="metric-val">{metrics["r2"]}</div><div class="metric-lbl">R² score</div></div>'
        f'<div class="metric-card"><div class="metric-val">${metrics["mae"]:,.0f}</div><div class="metric-lbl">MAE</div></div>'
        f'<div class="metric-card"><div class="metric-val">${metrics["rmse"]:,.0f}</div><div class="metric-lbl">RMSE</div></div>'
        f'<div class="metric-card"><div class="metric-val">{metrics["n_train"]:,}</div><div class="metric-lbl">training rows</div></div>'
        '</div>', unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# ── Sidebar: input form ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configure Laptop")
    st.markdown('<div class="section-head">Brand & Category</div>', unsafe_allow_html=True)

    brand    = st.selectbox("Brand",    CAT_OPTIONS['brand'])
    category = st.selectbox("Category", CAT_OPTIONS['category'])
    color    = st.selectbox("Color",    CAT_OPTIONS['color'])

    st.markdown('<div class="section-head">Display</div>', unsafe_allow_html=True)
    screen_size = st.slider("Screen size (inches)", 10.0, 18.0, 15.6, 0.1)
    resolution  = st.selectbox("Resolution", CAT_OPTIONS['resolution'])
    refresh_rate_hz = st.selectbox("Refresh rate (Hz)", [60, 90, 120, 144, 165, 240], index=0)

    st.markdown('<div class="section-head">Processor</div>', unsafe_allow_html=True)
    cpu       = st.selectbox("CPU", CAT_OPTIONS['cpu'])
    cpu_cores = st.selectbox("CPU cores", [2, 4, 6, 8, 10, 12, 16], index=1)

    st.markdown('<div class="section-head">Memory & Storage</div>', unsafe_allow_html=True)
    ram_gb       = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64], index=1)
    storage_gb   = st.selectbox("Storage (GB)", [128, 256, 512, 1024, 2048], index=2)
    storage_type = st.selectbox("Storage type", CAT_OPTIONS['storage_type'])

    st.markdown('<div class="section-head">Graphics & OS</div>', unsafe_allow_html=True)
    gpu              = st.selectbox("GPU", CAT_OPTIONS['gpu'])
    operating_system = st.selectbox("Operating system", CAT_OPTIONS['operating_system'])

    st.markdown('<div class="section-head">Physical</div>', unsafe_allow_html=True)
    weight_kg  = st.slider("Weight (kg)", 0.8, 5.0, 2.0, 0.1)
    battery_wh = st.slider("Battery (Wh)", 30, 120, 60, 5)
    year       = st.selectbox("Release year", list(range(2015, 2026))[::-1], index=1)

    st.markdown('<div class="section-head">Connectivity & Features</div>', unsafe_allow_html=True)
    usb_ports       = st.selectbox("USB ports", [1, 2, 3, 4, 5, 6], index=2)
    has_thunderbolt = st.selectbox("Thunderbolt", [0, 1], format_func=lambda x: "Yes" if x else "No")
    touchscreen        = st.selectbox("Touchscreen",        [0, 1], format_func=lambda x: "Yes" if x else "No")
    backlit_keyboard   = st.selectbox("Backlit keyboard",   [0, 1], format_func=lambda x: "Yes" if x else "No")
    fingerprint_reader = st.selectbox("Fingerprint reader", [0, 1], format_func=lambda x: "Yes" if x else "No")

    st.markdown('<div class="section-head">Reviews & Origin</div>', unsafe_allow_html=True)
    review_count     = st.number_input("Review count", 0, 50000, 500, step=100)
    rating           = st.slider("Rating", 1.0, 5.0, 4.2, 0.1)
    country_of_origin = st.selectbox("Country of origin", CAT_OPTIONS['country_of_origin'])
    warranty          = st.selectbox("Warranty",          CAT_OPTIONS['warranty'])

    predict_btn = st.button("Predict Price →")

# ── Prediction ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1.2, 1])

with col1:
    st.markdown('<div class="section-head">Laptop Summary</div>', unsafe_allow_html=True)

    summary_data = {
        "Brand / Category": f"{brand} · {category}",
        "CPU": f"{cpu} ({cpu_cores} cores)",
        "GPU": gpu,
        "RAM / Storage": f"{ram_gb} GB · {storage_gb} GB {storage_type}",
        "Display": f"{screen_size}\" {resolution} @ {refresh_rate_hz}Hz",
        "OS": operating_system,
        "Weight / Battery": f"{weight_kg} kg · {battery_wh} Wh",
        "Year": str(year),
        "Color": color,
        "Origin / Warranty": f"{country_of_origin} · {warranty}",
        "Extras": f"Touch:{bool(touchscreen)} | Backlit:{bool(backlit_keyboard)} | FP:{bool(fingerprint_reader)} | TB:{bool(has_thunderbolt)}",
        "Rating / Reviews": f"{rating} ★ ({review_count:,} reviews)",
    }

    for k, v in summary_data.items():
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #1c1c1c;">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.72rem;color:#555;text-transform:uppercase;letter-spacing:0.05em">{k}</span>'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.78rem;color:#c8c4bc;text-align:right;max-width:60%">{v}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

with col2:
    st.markdown('<div class="section-head">Predicted Price</div>', unsafe_allow_html=True)

    if predict_btn or 'last_pred' in st.session_state:
        if predict_btn:
            if not model_ready:
                st.error("Model not loaded. Check the dataset path.")
            else:
                input_df = pd.DataFrame([{
                    'screen_size': screen_size, 'ram_gb': ram_gb, 'storage_gb': storage_gb,
                    'year': year, 'weight_kg': weight_kg, 'battery_wh': battery_wh,
                    'refresh_rate_hz': refresh_rate_hz, 'cpu_cores': cpu_cores,
                    'touchscreen': touchscreen, 'backlit_keyboard': backlit_keyboard,
                    'fingerprint_reader': fingerprint_reader, 'usb_ports': usb_ports,
                    'has_thunderbolt': has_thunderbolt, 'review_count': review_count,
                    'rating': rating, 'brand': brand, 'category': category,
                    'resolution': resolution, 'cpu': cpu, 'gpu': gpu,
                    'storage_type': storage_type, 'operating_system': operating_system,
                    'color': color, 'country_of_origin': country_of_origin, 'warranty': warranty,
                }])

                with st.spinner("Computing…"):
                    pred = float(model.predict(input_df)[0])
                    time.sleep(0.3)

                st.session_state['last_pred'] = pred
                st.session_state['last_mae']  = metrics['mae']

        pred = st.session_state.get('last_pred', 0)
        mae  = st.session_state.get('last_mae', metrics.get('mae', 0))
        lo   = max(0, pred - mae)
        hi   = pred + mae

        st.markdown(
            f'<div class="price-box">'
            f'<div class="price-label">estimated price</div>'
            f'<div class="price-value">${pred:,.0f}</div>'
            f'<div class="price-range">confidence range  ${lo:,.0f} – ${hi:,.0f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Tier indicator
        if pred < 800:
            tier, tc = "Budget", "#5a9e5a"
        elif pred < 1500:
            tier, tc = "Mid-range", "#e8d5a3"
        elif pred < 2500:
            tier, tc = "Premium", "#c8a87a"
        else:
            tier, tc = "Flagship", "#e87a5a"

        st.markdown(
            f'<div style="text-align:center;margin-top:1rem;">'
            f'<span style="font-family:\'DM Mono\',monospace;font-size:0.75rem;letter-spacing:0.1em;'
            f'text-transform:uppercase;color:{tc}">▲ {tier} tier</span>'
            f'</div>',
            unsafe_allow_html=True
        )

        # MAE explanation
        st.markdown(
            f'<p style="font-family:\'DM Mono\',monospace;font-size:0.68rem;color:#444;'
            f'text-align:center;margin-top:1rem;line-height:1.6">'
            f'Range based on model MAE of ${mae:,.0f}<br>'
            f'Model R² = {metrics.get("r2","–")} on held-out test set</p>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="price-box" style="opacity:0.4">'
            '<div class="price-label">estimated price</div>'
            '<div class="price-value">$—</div>'
            '<div class="price-range">configure specs and click predict</div>'
            '</div>',
            unsafe_allow_html=True
        )
