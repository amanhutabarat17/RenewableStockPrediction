import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import sqlite3
import os
import io
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.tree import plot_tree
import matplotlib.patches as patches
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# =========================
# DATABASE SETUP
# =========================
def init_database():
    """Inisialisasi database SQLite untuk menyimpan history prediksi"""
    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            ticker TEXT NOT NULL,
            last_price REAL,
            pred_rf REAL,
            pred_ema5 REAL,
            pred_ema10 REAL,
            mape_rf REAL,
            mape_ema5 REAL,
            mape_ema10 REAL,
            r2_rf REAL,
            r2_ema5 REAL,
            r2_ema10 REAL,
            rmse_rf REAL,
            rmse_ema5 REAL,
            rmse_ema10 REAL,
            train_ratio REAL,
            rf_estimators INTEGER,
            rf_max_depth INTEGER,
            total_data_rf INTEGER,
            total_data_ema INTEGER,
            test_samples_count INTEGER,
            notes TEXT
        )
    ''')
    
    # Migration: Add new column if it doesn't exist
    try:
        cursor.execute("SELECT test_samples_count FROM prediction_history LIMIT 1")
    except sqlite3.OperationalError:
        # Column doesn't exist, add it
        cursor.execute("ALTER TABLE prediction_history ADD COLUMN test_samples_count INTEGER")
        print("✅ Database migrated: Added test_samples_count column")
    
    conn.commit()
    conn.close()

def save_prediction_to_db(ticker, res, params):
    """Simpan hasil prediksi ke database"""
    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO prediction_history (
            ticker, last_price, pred_rf, pred_ema5, pred_ema10,
            mape_rf, mape_ema5, mape_ema10,
            r2_rf, r2_ema5, r2_ema10,
            rmse_rf, rmse_ema5, rmse_ema10,
            train_ratio, rf_estimators, rf_max_depth, total_data_rf, total_data_ema,
            test_samples_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        ticker,
        float(res['df_full']['Price'].iloc[-1]),
        float(res['pred_next']['rf']),
        float(res['pred_next']['ema5']),
        float(res['pred_next']['ema10']),
        float(res['mape_rf']),
        float(res['mape_ema5']),
        float(res['mape_ema10']),
        float(res['r2_rf']),
        float(res['r2_ema5']),
        float(res['r2_ema10']),
        float(res['rmse_rf']),
        float(res['rmse_ema5']),
        float(res['rmse_ema10']),
        float(params['train_ratio']),
        int(params['rf_estimators']),
        int(params['rf_max_depth']),
        int(res['total_data_rf']),
        int(res['total_data_ema']),
        int(res['test_samples_count'])
    ))
    
    conn.commit()
    conn.close()

def get_prediction_history(ticker=None, limit=50):
    """Ambil history prediksi dari database"""
    conn = sqlite3.connect('stock_predictions.db')
    
    if ticker:
        query = '''
            SELECT * FROM prediction_history 
            WHERE ticker = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(ticker, limit))
    else:
        query = '''
            SELECT * FROM prediction_history 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        df = pd.read_sql_query(query, conn, params=(limit,))
    
    conn.close()
    return df

def delete_prediction_history(prediction_id):
    """Hapus history prediksi berdasarkan ID"""
    conn = sqlite3.connect('stock_predictions.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM prediction_history WHERE id = ?', (prediction_id,))
    conn.commit()
    conn.close()

# =========================
# PAGE CONFIG + ENHANCED CSS
# =========================
st.set_page_config(layout="wide", page_title="Sistem Prediksi Saham Multi Ticker", page_icon="📈")

st.markdown(
    """
<style>
    .stApp { 
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 2px solid #e3e8ef;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .equation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        font-family: 'Courier New', monospace;
        color: white;
        margin: 15px 0;
        font-size: 16px;
        font-weight: bold;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #00a8cc;
        margin: 10px 0;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 10px 0;
    }
    
    .success-box {
        background: #d4edda;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 10px 0;
    }
    
    .stTabs [data-baseweb="tab-list"] { 
        gap: 15px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background: white;
        border-radius: 10px 10px 0 0;
        padding: 0 25px;
        font-weight: 600;
        border: 2px solid #e3e8ef;
        transition: all 0.3s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8f9fa;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-color: #667eea !important;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 700;
    }
    
    .preprocessing-table {
        background: white;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    
    .bootstrap-sample {
        background: #f8f9fa;
        border: 2px solid #dee2e6;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
</style>
""",
    unsafe_allow_html=True
)

# Initialize database
init_database()

# =========================
# UTILITY FUNCTIONS
# =========================
def parse_number(x):
    """Parse berbagai format angka dari CSV termasuk K/M notasi dan separator berbeda"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    
    s = str(x).replace('"', "").replace("'", "").strip()
    if s == "": 
        return np.nan
    
    # Handle K/M multipliers
    mult = 1.0
    su = s.upper()
    if su.endswith("M"): 
        mult = 1_000_000.0
        s = s[:-1].strip()
    elif su.endswith("K"): 
        mult = 1_000.0
        s = s[:-1].strip()
    
    # Remove percentage
    s = s.replace("%", "").strip()
    
    # Handle different decimal/thousand separators
    if "." in s and "," in s:
        if s.rfind(".") > s.rfind(","): 
            s = s.replace(",", "")
        else: 
            s = s.replace(".", "").replace(",", ".")
    elif "," in s and "." not in s:
        if re.fullmatch(r"\d{1,3}(,\d{3})+", s): 
            s = s.replace(",", "")
        else: 
            s = s.replace(",", ".")
    elif "." in s and "," not in s:
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s): 
            s = s.replace(".", "")
    
    try:
        return float(s) * mult
    except:
        return np.nan

def baca_data_pintar(uploaded_file):
    """Membaca CSV dengan berbagai format separator dan encoding"""
    try:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file, encoding="utf-8-sig")
        
        if len(df.columns) <= 1:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8-sig")
        
        df.columns = df.columns.str.replace('"', "").str.replace("'", "").str.strip()
        
        col_date = next((c for c in df.columns if c.lower() in ["date", "tanggal", "time", "datetime"]), None)
        if not col_date: 
            return None
        
        df.rename(columns={col_date: "Date"}, inplace=True)
        return df
    except Exception as e:
        st.error(f"Error membaca file: {str(e)}")
        return None

def standardize_ohlc_columns(df):
    """Standardisasi nama kolom OHLC dan parsing nilai"""
    for c in list(df.columns):
        cl = c.lower().replace("_", "").replace(" ", "")
        if cl in ["vol.", "vol", "volume"]: 
            df.rename(columns={c: "Volume"}, inplace=True)
    
    required = ["Price", "Open", "High", "Low"]
    for c in required:
        if c in df.columns: 
            df[c] = df[c].apply(parse_number)
    
    if "Volume" in df.columns: 
        df["Volume"] = df["Volume"].apply(parse_number)
    
    return df

def process_train_one_ticker(df_one, train_ratio, rf_estimators, rf_max_depth):
    """Pipeline lengkap untuk training dan evaluasi model dengan SYNCHRONIZED TEST DATA"""
    df = df_one.copy()
    
    # Date processing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    
    # Simpan data sebelum preprocessing
    df_before_prep = df.copy()
    
    df = standardize_ohlc_columns(df)
    
    # ========================================
    # FEATURE ENGINEERING
    # ========================================
    
    # 1. Pastikan data terurut dari TANGGAL TERLAMA ke TERBARU
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date', ascending=True).reset_index(drop=True)
    
    # 2. Target t+1 (Harga besok)
    df["Target_Next_Price"] = df["Price"].shift(-1)
    
    # 3. Lag t-1 (Harga kemarin)
    df["Price_t_minus_1"] = df["Price"].shift(1)
    
    # 4. Fungsi EMA dengan SMA Start
    def calculate_ema_custom(series, span):
        alpha = 2 / (span + 1)
        ema_values = np.full(len(series), np.nan)
        
        sma_start = series.iloc[:span].mean()
        ema_values[span - 1] = sma_start
        
        curr_ema = sma_start
        for i in range(span, len(series)):
            curr_ema = (series.iloc[i] * alpha) + (curr_ema * (1 - alpha))
            ema_values[i] = curr_ema
            
        return pd.Series(ema_values, index=series.index)

    # 5. Eksekusi
    df["EMA_5"] = calculate_ema_custom(df["Price"], 5)
    df["EMA_10"] = calculate_ema_custom(df["Price"], 10)
    
    # ========================================
    # PREPARE DATASETS
    # ========================================
    features_rf = ["Price_t_minus_1", "Open", "High", "Low", "Volume"]
    
    # Dataset RF - mulai dari baris 1 (setelah drop lag)
    df_rf = df.dropna(subset=features_rf + ["Target_Next_Price"]).copy()
    
    # Dataset EMA - mulai dari baris 10 (setelah warmup EMA_10)
    df_ema = df.dropna(subset=["EMA_5", "EMA_10", "Target_Next_Price"]).copy()
    
    # ========================================
    # SYNCHRONIZED SPLIT STRATEGY
    # ========================================
    
    # 1. Split RF dulu menggunakan train_ratio
    split_idx_rf = int(len(df_rf) * train_ratio)
    
    # 2. Ambil tanggal awal test set dari RF
    test_start_date = df_rf.iloc[split_idx_rf]['Date']
    
    # 3. Cari index yang sama di df_ema berdasarkan tanggal
    split_idx_ema = df_ema[df_ema['Date'] < test_start_date].shape[0]
    
    # 4. Pastikan EMA punya data test (jika tidak, EMA dataset terlalu pendek)
    if split_idx_ema >= len(df_ema):
        # Fallback: gunakan ratio yang sama untuk EMA
        split_idx_ema = int(len(df_ema) * train_ratio)
        # Update test_start_date dari EMA (ambil yang lebih baru)
        if split_idx_ema < len(df_ema):
            test_start_date = df_ema.iloc[split_idx_ema]['Date']
            # Update split_idx_rf
            split_idx_rf = df_rf[df_rf['Date'] < test_start_date].shape[0]
    
    # ========================================
    # RANDOM FOREST MODEL
    # ========================================
    X_rf, y_rf = df_rf[features_rf], df_rf["Target_Next_Price"]
    X_train_rf, X_test_rf = X_rf.iloc[:split_idx_rf], X_rf.iloc[split_idx_rf:]
    y_train_rf, y_test_rf = y_rf.iloc[:split_idx_rf], y_rf.iloc[split_idx_rf:]
    
    rf = RandomForestRegressor(
        n_estimators=rf_estimators, 
        max_depth=rf_max_depth, 
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True
    )
    rf.fit(X_train_rf, y_train_rf)
    y_pred_rf = rf.predict(X_test_rf)
    
    mape_rf = mean_absolute_percentage_error(y_test_rf, y_pred_rf)
    r2_rf = r2_score(y_test_rf, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test_rf, y_pred_rf))
    
    last_row = df.iloc[-1]
    future_features_rf = [
        last_row["Price"],
        last_row["Open"],
        last_row["High"],
        last_row["Low"],
        last_row["Volume"] if "Volume" in df.columns else 0
    ]
    future_rf = rf.predict([future_features_rf])
    
    # ========================================
    # EMA PREDICTIONS
    # ========================================
    y_test_ema = df_ema["Target_Next_Price"].iloc[split_idx_ema:]
    y_pred_ema5 = df_ema["EMA_5"].iloc[split_idx_ema:].values
    y_pred_ema10 = df_ema["EMA_10"].iloc[split_idx_ema:].values
    
    mape_ema5 = mean_absolute_percentage_error(y_test_ema, y_pred_ema5)
    r2_ema5 = r2_score(y_test_ema, y_pred_ema5)
    rmse_ema5 = np.sqrt(mean_squared_error(y_test_ema, y_pred_ema5))
    
    mape_ema10 = mean_absolute_percentage_error(y_test_ema, y_pred_ema10)
    r2_ema10 = r2_score(y_test_ema, y_pred_ema10)
    rmse_ema10 = np.sqrt(mean_squared_error(y_test_ema, y_pred_ema10))
    
    future_ema5 = last_row["EMA_5"]
    future_ema10 = last_row["EMA_10"]
    
    dates_test_rf = df_rf["Date"].iloc[split_idx_rf:]
    dates_test_ema = df_ema["Date"].iloc[split_idx_ema:]
    
    feature_importance = pd.DataFrame({
        'feature': features_rf,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # ========================================
    # BOOTSTRAP SAMPLES INFO
    # ========================================
    bootstrap_info = []
    for tree_idx, estimator in enumerate(rf.estimators_[:min(5, len(rf.estimators_))]):
        n_samples_bootstrap = estimator.tree_.n_node_samples[0]
        bootstrap_info.append({
            'tree_idx': tree_idx,
            'n_samples': n_samples_bootstrap,
            'tree_depth': estimator.tree_.max_depth,
            'n_leaves': (estimator.tree_.children_left == -1).sum()
        })
    
    # ========================================
    # VERIFY SYNCHRONIZED TEST DATA
    # ========================================
    test_samples_count = len(y_test_rf)
    test_samples_count_ema = len(y_test_ema)
    
    return {
        "df_before": df_before_prep,
        "df_full": df,
        "df_rf": df_rf,
        "df_ema": df_ema,
        "rf_model": rf,
        "X_train_rf": X_train_rf,
        "y_train_rf": y_train_rf,
        "y_test_rf": y_test_rf,
        "y_test_ema": y_test_ema,
        "y_pred_rf": y_pred_rf,
        "y_pred_ema5": y_pred_ema5,
        "y_pred_ema10": y_pred_ema10,
        "dates_test_rf": dates_test_rf,
        "dates_test_ema": dates_test_ema,
        "mape_rf": mape_rf,
        "mape_ema5": mape_ema5,
        "mape_ema10": mape_ema10,
        "r2_rf": r2_rf,
        "r2_ema5": r2_ema5,
        "r2_ema10": r2_ema10,
        "rmse_rf": rmse_rf,
        "rmse_ema5": rmse_ema5,
        "rmse_ema10": rmse_ema10,
        "pred_next": {
            "rf": future_rf[0], 
            "ema5": future_ema5,
            "ema10": future_ema10
        },
        "features": features_rf,
        "feature_importance": feature_importance,
        "total_data_rf": len(df_rf),
        "total_data_ema": len(df_ema),
        "bootstrap_info": bootstrap_info,
        "split_idx_rf": split_idx_rf,
        "split_idx_ema": split_idx_ema,
        "test_samples_count": test_samples_count,
        "test_samples_count_ema": test_samples_count_ema,
        "test_start_date": test_start_date
    }

# =========================
# VISUALIZATION FUNCTIONS
# =========================
def plot_single_decision_tree(
    rf_model,
    features,
    tree_index=0,
    max_depth=None,
    figsize=(22, 13),
    fontsize=10,
    title=None,
    dpi=140
):
    """Visualisasi Decision Tree dengan custom warna berdasarkan tertiles"""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    tree_estimator = rf_model.estimators_[tree_index]
    
    anns = plot_tree(
        tree_estimator,
        feature_names=features,
        filled=True,
        rounded=True,
        precision=2,
        fontsize=fontsize,
        max_depth=max_depth,
        impurity=True,
        node_ids=False,
        proportion=False,
        ax=ax
    )
    
    # Custom Coloring
    all_values = tree_estimator.tree_.value.flatten()
    if len(all_values) > 0:
        low_tertile, high_tertile = np.percentile(all_values, [33, 66]) if len(all_values) > 2 else (min(all_values), max(all_values))
        
        for ann in anns:
            text = ann.get_text()
            match = re.search(r'value = ([\d\.]+)', text)
            if match:
                node_value = float(match.group(1))
                
                if node_value >= high_tertile:
                    color = '#228B22'  # Hijau
                elif node_value >= low_tertile:
                    color = '#FFD700'  # Kuning
                else:
                    color = '#FF8C00'  # Orange
                
                bbox = ann.get_bbox_patch()
                if bbox is not None:
                    bbox.set_facecolor(color)
                    bbox.set_alpha(0.82)

    if title is None:
        title = "Decision Tree Prediksi"
    
    ax.set_title(
        f"{title} #{tree_index+1}\n"
        f"(Kedalaman aktual: {tree_estimator.tree_.max_depth} | "
        f"Tree ke-{tree_index+1}/{len(rf_model.estimators_)})",
        fontsize=fontsize + 6,
        fontweight='bold',
        pad=20
    )
    
    plt.tight_layout()
    return fig

def visualize_bootstrap_samples(X_train, y_train, rf_model, n_trees_to_show=5):
    """Visualisasi bootstrap samples yang digunakan oleh Random Forest"""
    fig, axes = plt.subplots(n_trees_to_show, 1, figsize=(14, 4*n_trees_to_show))
    if n_trees_to_show == 1:
        axes = [axes]
    
    total_samples = len(X_train)
    
    for idx in range(n_trees_to_show):
        ax = axes[idx]
        
        np.random.seed(42 + idx)
        bootstrap_indices = np.random.choice(total_samples, total_samples, replace=True)
        unique_samples = len(np.unique(bootstrap_indices))
        oob_samples = total_samples - unique_samples
        
        sample_count = np.bincount(bootstrap_indices, minlength=total_samples)
        
        colors = ['#d4edda' if count > 0 else '#f8d7da' for count in sample_count]
        ax.bar(range(total_samples), sample_count, color=colors, edgecolor='black', linewidth=0.5)
        
        ax.set_title(f'Tree #{idx+1} - Bootstrap Sample Distribution\n'
                    f'Unique samples: {unique_samples}/{total_samples} ({unique_samples/total_samples*100:.1f}%) | '
                    f'OOB samples: {oob_samples} ({oob_samples/total_samples*100:.1f}%)',
                    fontweight='bold', fontsize=11)
        ax.set_xlabel('Sample Index', fontweight='bold')
        ax.set_ylabel('Times Selected', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#d4edda', edgecolor='black', label='In-bag'),
            Patch(facecolor='#f8d7da', edgecolor='black', label='Out-of-bag (OOB)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    return fig

def generate_bootstrap_details_for_excel(X_train, y_train, rf_model, n_trees=5, feature_names=None):
    """Generate detailed bootstrap information for each tree for Excel export"""
    total_samples = len(X_train)
    bootstrap_details = {}
    
    for tree_idx in range(min(n_trees, len(rf_model.estimators_))):
        np.random.seed(42 + tree_idx)
        bootstrap_indices = np.random.choice(total_samples, total_samples, replace=True)
        
        sample_count = np.bincount(bootstrap_indices, minlength=total_samples)
        
        tree_data = []
        for sample_idx in range(total_samples):
            count = sample_count[sample_idx]
            status = 'In-bag' if count > 0 else 'Out-of-bag (OOB)'
            
            row = {
                'Sample_Index': sample_idx,
                'Times_Selected': int(count),
                'Status': status,
                'Target_Value': float(y_train.iloc[sample_idx])
            }
            
            if feature_names:
                for feat_idx, feat_name in enumerate(feature_names):
                    row[feat_name] = float(X_train.iloc[sample_idx, feat_idx])
            
            tree_data.append(row)
        
        df_tree = pd.DataFrame(tree_data)
        
        summary_data = {
            'Sample_Index': ['SUMMARY', '', 'Total Samples', 'In-bag Samples', 'OOB Samples', 'Unique In-bag', 'In-bag %', 'OOB %'],
            'Times_Selected': ['', '', total_samples, (df_tree['Status'] == 'In-bag').sum(), 
                              (df_tree['Status'] == 'Out-of-bag (OOB)').sum(),
                              len(df_tree[df_tree['Times_Selected'] > 0]),
                              f"{(df_tree['Status'] == 'In-bag').sum() / total_samples * 100:.1f}%",
                              f"{(df_tree['Status'] == 'Out-of-bag (OOB)').sum() / total_samples * 100:.1f}%"],
            'Status': ['', '', '', '', '', '', '', '']
        }
        
        if feature_names:
            for feat_name in feature_names:
                summary_data[feat_name] = [''] * 8
        summary_data['Target_Value'] = [''] * 8
        
        df_summary = pd.DataFrame(summary_data)
        
        df_combined = pd.concat([df_summary, df_tree], ignore_index=True)
        
        bootstrap_details[f'Tree_{tree_idx + 1}'] = df_combined
    
    return bootstrap_details

def extract_tree_structure(tree_estimator, feature_names):
    """Extract complete tree structure including features and thresholds"""
    tree = tree_estimator.tree_
    
    structure_data = []
    
    def recurse(node_id, depth=0):
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        n_samples = tree.n_node_samples[node_id]
        value = tree.value[node_id][0][0]
        impurity = tree.impurity[node_id]
        
        is_leaf = (tree.children_left[node_id] == tree.children_right[node_id])
        
        if is_leaf:
            structure_data.append({
                'Node_ID': node_id,
                'Depth': depth,
                'Type': 'Leaf',
                'Feature': 'N/A',
                'Threshold': 'N/A',
                'Samples': n_samples,
                'Value': round(value, 2),
                'Impurity': round(impurity, 4),
                'Left_Child': 'N/A',
                'Right_Child': 'N/A'
            })
        else:
            feature_name = feature_names[feature_idx] if feature_names else f'X[{feature_idx}]'
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            structure_data.append({
                'Node_ID': node_id,
                'Depth': depth,
                'Type': 'Internal',
                'Feature': feature_name,
                'Threshold': round(threshold, 2),
                'Samples': n_samples,
                'Value': round(value, 2),
                'Impurity': round(impurity, 4),
                'Left_Child': left_child,
                'Right_Child': right_child
            })
            
            recurse(left_child, depth + 1)
            recurse(right_child, depth + 1)
    
    recurse(0)
    
    return pd.DataFrame(structure_data)

def get_tree_feature_importance(tree_estimator, feature_names):
    """Get feature importance for a single tree"""
    tree = tree_estimator.tree_
    
    feature_importance = np.zeros(len(feature_names))
    
    def recurse(node_id):
        feature_idx = tree.feature[node_id]
        
        if tree.children_left[node_id] != tree.children_right[node_id]:
            left_child = tree.children_left[node_id]
            right_child = tree.children_right[node_id]
            
            n_samples = tree.n_node_samples[node_id]
            n_left = tree.n_node_samples[left_child]
            n_right = tree.n_node_samples[right_child]
            
            impurity_decrease = (
                tree.impurity[node_id] -
                (n_left / n_samples) * tree.impurity[left_child] -
                (n_right / n_samples) * tree.impurity[right_child]
            )
            
            feature_importance[feature_idx] += n_samples * impurity_decrease
            
            recurse(left_child)
            recurse(right_child)
    
    recurse(0)
    
    if feature_importance.sum() > 0:
        feature_importance = feature_importance / feature_importance.sum()
    
    df_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance,
        'Percentage': feature_importance * 100
    }).sort_values('Importance', ascending=False)
    
    return df_importance

# =========================
# SIDEBAR CONFIGURATION
# =========================
st.sidebar.markdown("## ⚙️ Konfigurasi Model")
st.sidebar.markdown("---")

st.sidebar.markdown("### 📁 Upload Data")
file_arko = st.sidebar.file_uploader("📊 Upload ARKO.JK CSV", type="csv")
file_keen = st.sidebar.file_uploader("📊 Upload KEEN.JK CSV", type="csv")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Parameter Model")

t_ratio = st.sidebar.slider("📐 Ratio Training/Testing", 0.5, 0.9, 0.8, 0.05)
rf_est = st.sidebar.slider("🌲 Jumlah Trees (RF)", 5, 500, 5, 5)
rf_dep = st.sidebar.slider("📏 Kedalaman Maksimal (RF)", 2, 20, 10, 1)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Informasi")
st.sidebar.info(
    """
    **Sistem Prediksi Saham**
    
    **Random Forest:**
    - Bootstrap sampling untuk setiap tree
    - Data: Dari baris paling atas (setelah lag)
    - Fitur: Price(t-1), OHLC(t), Volume(t)
    
    **EMA Indicators:**
    - EMA 5 dan EMA 10
    - Data: Setelah warmup period
    - Prediksi: Nilai EMA langsung
    
    **⚖️ SYNCHRONIZED TEST DATA:**
    - Test set mengikuti RANDOM FOREST
    - Split RF dulu → cari tanggal yang sama di EMA
    - Test samples lebih banyak (RF punya data lebih awal)
    - Perbandingan lebih fair & valid
    
    © 2026 Stock Analysis System
    """
)

# =========================
# MAIN APPLICATION
# =========================
data_map = {}
if file_arko: 
    df_arko = baca_data_pintar(file_arko)
    if df_arko is not None:
        data_map["ARKO.JK"] = df_arko

if file_keen: 
    df_keen = baca_data_pintar(file_keen)
    if df_keen is not None:
        data_map["KEEN.JK"] = df_keen

# Header
st.markdown(
    """
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
    border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; margin: 0;'>📈 Sistem Prediksi Harga Saham</h1>
        <p style='color: #f0f0f0; margin: 10px 0 0 0; font-size: 18px;'>
            Time Series Forecasting: Random Forest (Bootstrap) + EMA Indicators
        </p>
        <p style='color: #ffd700; margin: 5px 0 0 0; font-size: 14px; font-weight: bold;'>
            ✅ SYNCHRONIZED TEST DATA - Mengikuti Random Forest (Lebih Panjang)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

if not data_map:
    st.markdown(
        """
        <div class='info-box'>
            <h3>👋 Selamat Datang!</h3>
            <p>Untuk memulai analisis, silakan upload file CSV saham di sidebar.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    history_df = get_prediction_history(limit=100)
    
    if len(history_df) > 0:
        st.markdown("### 📜 History Prediksi")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Prediksi", len(history_df))
        col2.metric("Ticker Berbeda", history_df['ticker'].nunique())
        if 'mape_rf' in history_df.columns:
            col3.metric("Avg MAPE (RF)", f"{history_df['mape_rf'].mean()*100:.2f}%")
        if 'mape_ema5' in history_df.columns:
            col4.metric("Avg MAPE (EMA5)", f"{history_df['mape_ema5'].mean()*100:.2f}%")
        
        st.dataframe(history_df, use_container_width=True, height=400)
else:
    ticker_choice = st.selectbox("🎯 Pilih Saham:", list(data_map.keys()))
    
    with st.spinner(f"🔄 Memproses data untuk {ticker_choice}..."):
        res = process_train_one_ticker(data_map[ticker_choice], t_ratio, rf_est, rf_dep)
    
    params = {'train_ratio': t_ratio, 'rf_estimators': rf_est, 'rf_max_depth': rf_dep}
    save_prediction_to_db(ticker_choice, res, params)
    
    # SYNCHRONIZED DATA VERIFICATION
    st.markdown(
        f"""
        <div class='success-box'>
            ✅ <strong>Model berhasil dilatih untuk {ticker_choice}</strong><br>
            📊 Random Forest: {res['total_data_rf']} baris | {rf_est} trees dengan bootstrap sampling<br>
            📈 EMA: {res['total_data_ema']} baris (setelah warmup period)<br><br>
            
            <strong>⚖️ SYNCHRONIZED TEST DATA (Mengikuti Random Forest):</strong><br>
            🔹 Test start date: {res['test_start_date'].strftime('%Y-%m-%d')}<br>
            🔹 RF test samples: {res['test_samples_count']}<br>
            🔹 EMA test samples: {res['test_samples_count_ema']}<br>
            🔹 Status: <span style='color: {'green' if res['test_samples_count'] == res['test_samples_count_ema'] else 'orange'}; font-weight: bold;'>{'✅ SAMA - Perbandingan Valid!' if res['test_samples_count'] == res['test_samples_count_ema'] else f"⚠️ BERBEDA {abs(res['test_samples_count'] - res['test_samples_count_ema'])} baris (EMA dataset lebih pendek)"}</span><br>
            💡 <em>Test samples mengikuti RF karena RF punya data lebih banyak dari awal (mulai baris 1 vs EMA baris 10)</em>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Warning jika tidak sama
    if res['test_samples_count'] != res['test_samples_count_ema']:
        st.markdown(
            f"""
            <div class='warning-box'>
                ⚠️ <strong>PERHATIAN:</strong> Jumlah test samples berbeda!<br>
                RF: {res['test_samples_count']} vs EMA: {res['test_samples_count_ema']}<br>
                Perbedaan: {abs(res['test_samples_count'] - res['test_samples_count_ema'])} samples<br>
                Ini bisa mempengaruhi perbandingan metrik.
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # TABS
    tab_prep, tab_bootstrap, tab_results, tab_comparison, tab_tree, tab_history = st.tabs([
        "🔧 Preprocessing Detail",
        "🎲 Bootstrap Samples",
        "📊 Hasil Prediksi",
        "⚖️ Perbandingan",
        "🌲 Decision Tree",
        "📜 History"
    ])
    
    # =====================
    # TAB: PREPROCESSING DETAIL
    # =====================
    with tab_prep:
        st.markdown(f"### 🔧 Detail Tahapan Preprocessing: {ticker_choice}")
        
        # Section 1: Data Mentah
        st.markdown("#### 📄 Step 1: Data Mentah (Raw Data)")
        st.markdown(
            """
            <div class='info-box'>
                Data yang baru di-upload, sebelum dilakukan parsing atau transformasi apapun.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(res['df_before'].head(20), use_container_width=True)
        
        st.markdown("---")
        
        # Section 2: Setelah Parsing & Standardisasi
        st.markdown("#### 🔢 Step 2: Setelah Parsing Number & Standardisasi Kolom")
        st.markdown(
            """
            <div class='info-box'>
                <ul>
                    <li>Parsing angka: K/M notation, berbagai decimal separator</li>
                    <li>Standardisasi nama kolom: Price, Open, High, Low, Volume</li>
                    <li>Konversi Date ke datetime format</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        clean_cols = ['Date', 'Price', 'Open', 'High', 'Low', 'Volume'] if 'Volume' in res['df_full'].columns else ['Date', 'Price', 'Open', 'High', 'Low']
        st.dataframe(res['df_full'][clean_cols].head(20), use_container_width=True)
        
        st.markdown("---")
        
        # Section 3: Feature Engineering
        st.markdown("#### ⚙️ Step 3: Feature Engineering")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🎯 Target & Lag Features")
            st.markdown(
                """
                <div class='equation-box'>
                    Target(t) = Price(t+1)<br>
                    Price_t_minus_1(t) = Price(t-1)
                </div>
                """,
                unsafe_allow_html=True
            )
            lag_cols = ['Date', 'Price', 'Price_t_minus_1', 'Target_Next_Price']
            st.dataframe(res['df_full'][lag_cols].head(20), use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 EMA Calculation")
            st.markdown(
                """
                <div class='equation-box'>
                    α = 2 / (span + 1)<br>
                    EMA(0) = SMA(span)<br>
                    EMA(t) = α × Price(t) + (1-α) × EMA(t-1)
                </div>
                """,
                unsafe_allow_html=True
            )
            ema_cols = ['Date', 'Price', 'EMA_5', 'EMA_10']
            st.dataframe(res['df_full'][ema_cols].head(20), use_container_width=True)
        
        st.markdown("---")
        
        # Section 4: Dataset Split dengan SYNCHRONIZED INFO
        st.markdown("#### 📊 Step 4: Dataset untuk Modeling (SYNCHRONIZED)")
        
        split_idx_rf = res['split_idx_rf']
        split_idx_ema = res['split_idx_ema']
        
        train_size_rf = split_idx_rf
        test_size_rf = len(res['df_rf']) - split_idx_rf
        train_size_ema = split_idx_ema
        test_size_ema = len(res['df_ema']) - split_idx_ema
        
        # Highlight synchronized info
        st.markdown(
            f"""
            <div class='success-box'>
                <h4>✅ SYNCHRONIZED TEST DATA (Mengikuti Random Forest)</h4>
                <ul>
                    <li><strong>Strategi:</strong> Split RF dulu → EMA menyesuaikan tanggal yang sama</li>
                    <li><strong>Test Start Date:</strong> {res['test_start_date'].strftime('%Y-%m-%d')}</li>
                    <li><strong>RF Test Samples:</strong> {test_size_rf} (dari total {res['total_data_rf']} baris)</li>
                    <li><strong>EMA Test Samples:</strong> {test_size_ema} (dari total {res['total_data_ema']} baris)</li>
                    <li><strong>Match Status:</strong> {'✅ SAMA' if test_size_rf == test_size_ema else '⚠️ BERBEDA ({} baris)'.format(abs(test_size_rf - test_size_ema))}</li>
                    <li><strong>Keuntungan:</strong> Test samples lebih panjang karena mengikuti RF yang punya data lebih awal</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🤖 Random Forest Dataset")
            st.markdown(
                f"""
                <div class='info-box'>
                    <strong>Total Data:</strong> {res['total_data_rf']} baris<br>
                    <strong>Mulai dari baris:</strong> {res['df_rf'].index[0]} (baris ke-{res['df_rf'].index[0] + 1})<br>
                    <strong>Training:</strong> {train_size_rf} baris ({train_size_rf/res['total_data_rf']*100:.1f}%)<br>
                    <strong>Testing:</strong> {test_size_rf} baris ({test_size_rf/res['total_data_rf']*100:.1f}%)<br>
                    <strong>Alasan drop baris 0:</strong> Price_t_minus_1 = NaN<br>
                    <strong>Features:</strong> {', '.join(res['features'])}<br>
                    <strong>Test Start:</strong> {res['test_start_date'].strftime('%Y-%m-%d')}
                </div>
                """,
                unsafe_allow_html=True
            )
            rf_display_cols = ['Date'] + res['features'] + ['Target_Next_Price']
            st.dataframe(res['df_rf'][rf_display_cols].head(15), use_container_width=True)
        
        with col2:
            st.markdown("##### 📈 EMA Dataset")
            st.markdown(
                f"""
                <div class='info-box'>
                    <strong>Total Data:</strong> {res['total_data_ema']} baris<br>
                    <strong>Mulai dari baris:</strong> {res['df_ema'].index[0]} (baris ke-{res['df_ema'].index[0] + 1})<br>
                    <strong>Training:</strong> {train_size_ema} baris ({train_size_ema/res['total_data_ema']*100:.1f}%)<br>
                    <strong>Testing:</strong> {test_size_ema} baris ({test_size_ema/res['total_data_ema']*100:.1f}%)<br>
                    <strong>Alasan drop baris 0-9:</strong> Warmup period untuk EMA_10<br>
                    <strong>Predictors:</strong> EMA_5, EMA_10<br>
                    <strong>Test Start:</strong> {res['test_start_date'].strftime('%Y-%m-%d')}
                </div>
                """,
                unsafe_allow_html=True
            )
            ema_display_cols = ['Date', 'Price', 'EMA_5', 'EMA_10', 'Target_Next_Price']
            st.dataframe(res['df_ema'][ema_display_cols].head(15), use_container_width=True)
        
        st.markdown("---")
        
        # Section 5: Complete Comparison Table
        st.markdown("#### 📋 Step 5: Tabel Lengkap dengan Status Validasi (SEMUA BARIS)")
        
        total_rows = len(res['df_full'])
        st.markdown(
            f"""
            <div class='warning-box'>
                <h4>⚠️ Legend Status (Total: {total_rows} baris):</h4>
                <ul>
                    <li><strong>❌ RF (lag):</strong> Baris 0 - tidak punya Price_t_minus_1</li>
                    <li><strong>✅ RF | ❌ EMA5:</strong> Baris 1-4 - valid untuk RF, tapi EMA_5 masih NaN</li>
                    <li><strong>✅ RF | ❌ EMA10:</strong> Baris 5-9 - valid untuk RF, tapi EMA_10 masih NaN</li>
                    <li><strong>✅ ALL:</strong> Baris 10+ - semua model bisa menggunakan</li>
                </ul>
                <p><strong>📊 Menampilkan {total_rows} baris lengkap dari baris 0 hingga baris {total_rows-1}</strong></p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        comprehensive_cols = [
            'Date', 'Price', 'Open', 'High', 'Low', 'Volume',
            'Price_t_minus_1', 'EMA_5', 'EMA_10', 'Target_Next_Price'
        ]
        available_cols = [c for c in comprehensive_cols if c in res['df_full'].columns]
        
        display_df = res['df_full'][available_cols].copy()
        display_df.insert(0, 'Baris', display_df.index)
        display_df.insert(1, 'Status', '')
        
        for idx in display_df.index:
            if idx == 0:
                display_df.loc[idx, 'Status'] = '❌ RF (lag)'
            elif idx < 5:
                display_df.loc[idx, 'Status'] = '✅ RF | ❌ EMA5'
            elif idx < 10:
                display_df.loc[idx, 'Status'] = '✅ RF | ❌ EMA10'
            else:
                display_df.loc[idx, 'Status'] = '✅ ALL'
        
        st.dataframe(display_df, use_container_width=True, height=600)
        
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Tabel Lengkap (CSV)",
            data=csv_data,
            file_name=f"preprocessing_full_{ticker_choice}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            type="secondary",
            use_container_width=True
        )
        
        st.markdown("---")
        
        # Section 5b: Row Status Summary
        st.markdown("#### 📊 Ringkasan Status Baris")
        
        n_rf_dropped = 1
        n_ema5_dropped = 5
        n_ema10_dropped = 10
        n_all_valid = total_rows - 10 if total_rows > 10 else 0
        
        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
        
        with col_sum1:
            st.markdown(
                f"""
                <div class='metric-card' style='background: #f8d7da; border-color: #f5c6cb;'>
                    <h4 style='color: #721c24; margin: 0;'>❌ Tidak Valid</h4>
                    <h2 style='margin: 10px 0; color: #721c24;'>{n_rf_dropped}</h2>
                    <p style='color: #721c24; margin: 0; font-size: 12px;'>Baris 0 (lag)</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_sum2:
            st.markdown(
                f"""
                <div class='metric-card' style='background: #fff3cd; border-color: #ffeaa7;'>
                    <h4 style='color: #856404; margin: 0;'>⚠️ RF Only</h4>
                    <h2 style='margin: 10px 0; color: #856404;'>{n_ema5_dropped - n_rf_dropped}</h2>
                    <p style='color: #856404; margin: 0; font-size: 12px;'>Baris 1-4 (EMA5 NaN)</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_sum3:
            st.markdown(
                f"""
                <div class='metric-card' style='background: #d1ecf1; border-color: #bee5eb;'>
                    <h4 style='color: #0c5460; margin: 0;'>⚠️ RF + EMA5</h4>
                    <h2 style='margin: 10px 0; color: #0c5460;'>{n_ema10_dropped - n_ema5_dropped}</h2>
                    <p style='color: #0c5460; margin: 0; font-size: 12px;'>Baris 5-9 (EMA10 NaN)</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_sum4:
            st.markdown(
                f"""
                <div class='metric-card' style='background: #d4edda; border-color: #c3e6cb;'>
                    <h4 style='color: #155724; margin: 0;'>✅ Semua Valid</h4>
                    <h2 style='margin: 10px 0; color: #155724;'>{n_all_valid}</h2>
                    <p style='color: #155724; margin: 0; font-size: 12px;'>Baris 10-{total_rows-1}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown(
            f"""
            <div class='info-box'>
                <h4>📈 Persentase Utilisasi Data:</h4>
                <ul>
                    <li><strong>Random Forest:</strong> {res['total_data_rf']}/{total_rows} baris = {res['total_data_rf']/total_rows*100:.1f}% (kehilangan {n_rf_dropped} baris)</li>
                    <li><strong>EMA Models:</strong> {res['total_data_ema']}/{total_rows} baris = {res['total_data_ema']/total_rows*100:.1f}% (kehilangan {n_ema10_dropped} baris)</li>
                    <li><strong>Efisiensi RF vs EMA:</strong> RF menggunakan {res['total_data_rf'] - res['total_data_ema']} baris lebih banyak ({(res['total_data_rf'] - res['total_data_ema'])/total_rows*100:.1f}% lebih efisien)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Section 6: Summary Statistics
        st.markdown("#### 📊 Step 6: Summary Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("##### 📉 Data Asli")
            st.write(res['df_before'].describe())
        
        with col2:
            st.markdown("##### 🤖 RF Dataset")
            st.write(res['df_rf'][res['features']].describe())
        
        with col3:
            st.markdown("##### 📈 EMA Dataset")
            st.write(res['df_ema'][['EMA_5', 'EMA_10']].describe())
    
    # =====================
    # TAB: BOOTSTRAP SAMPLES
    # =====================
    with tab_bootstrap:
        st.markdown(f"### 🎲 Bootstrap Sampling Analysis: {ticker_choice}")
        
        st.markdown(
            """
            <div class='info-box'>
                <h4>📚 Apa itu Bootstrap Sampling?</h4>
                <p>Random Forest menggunakan <strong>bootstrap sampling</strong> untuk membuat setiap decision tree:</p>
                <ul>
                    <li><strong>Bootstrap:</strong> Sampling dengan penggantian (with replacement)</li>
                    <li><strong>In-bag samples:</strong> Data yang terpilih untuk training tree tersebut</li>
                    <li><strong>Out-of-bag (OOB) samples:</strong> Data yang tidak terpilih (~36.8% dari total data)</li>
                    <li>Setiap tree melihat subset data yang berbeda → diversity → ensemble strength</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Bootstrap Info Table
        st.markdown("#### 📊 Informasi Bootstrap untuk Setiap Tree")
        
        bootstrap_df = pd.DataFrame(res['bootstrap_info'])
        bootstrap_df['tree'] = bootstrap_df['tree_idx'] + 1
        bootstrap_df = bootstrap_df[['tree', 'n_samples', 'tree_depth', 'n_leaves']]
        bootstrap_df.columns = ['Tree #', 'Bootstrap Samples', 'Tree Depth', 'Leaf Nodes']
        
        st.dataframe(bootstrap_df, use_container_width=True, hide_index=True)
        
        # Excel Download Button
        if EXCEL_AVAILABLE:
            st.markdown("---")
            st.markdown("#### 📥 Download Detail Bootstrap Samples")
            
            n_trees_export = min(rf_est, 10)
            total_sheets = n_trees_export * 3 + 1
            
            st.markdown(
                f"""
                <div class='success-box'>
                    <h4>📊 Isi File Excel:</h4>
                    <ul>
                        <li><strong>Summary Sheet:</strong> Ringkasan semua trees (samples, depth, leaf nodes)</li>
                        <li><strong>Bootstrap Sheets (Tree_1, Tree_2, ...):</strong> Detail per sample - index, berapa kali dipilih, status (In-bag/OOB), nilai target, semua features</li>
                        <li><strong>Structure Sheets (T1_Structure, T2_Structure, ...):</strong> ⭐ BARU! Struktur lengkap tree - setiap node, feature yang digunakan, threshold pemecah, jumlah samples di node</li>
                        <li><strong>Feature Importance Sheets (T1_Features, T2_Features, ...):</strong> ⭐ BARU! Importance setiap feature untuk tree spesifik ini</li>
                    </ul>
                    <p><strong>Total: {n_trees_export} trees × 3 sheets + 1 Summary = {total_sheets} sheets</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("🎲 Generate Excel Bootstrap Details", type="primary", use_container_width=True):
                with st.spinner("Generating detailed bootstrap information..."):
                    bootstrap_excel_data = generate_bootstrap_details_for_excel(
                        X_train=res['X_train_rf'],
                        y_train=res['y_train_rf'],
                        rf_model=res['rf_model'],
                        n_trees=min(rf_est, 10),
                        feature_names=res['features']
                    )
                    
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        summary_list = []
                        for tree_idx in range(min(rf_est, 10)):
                            tree_info = res['bootstrap_info'][tree_idx] if tree_idx < len(res['bootstrap_info']) else None
                            if tree_info:
                                summary_list.append({
                                    'Tree': tree_idx + 1,
                                    'Bootstrap_Samples': tree_info['n_samples'],
                                    'Tree_Depth': tree_info['tree_depth'],
                                    'Leaf_Nodes': tree_info['n_leaves']
                                })
                        
                        df_summary_all = pd.DataFrame(summary_list)
                        df_summary_all.to_excel(writer, sheet_name='Summary', index=False)
                        
                        for tree_idx in range(min(rf_est, 10)):
                            if tree_idx >= len(res['rf_model'].estimators_):
                                break
                            
                            tree_estimator = res['rf_model'].estimators_[tree_idx]
                            
                            sheet_name = f'Tree_{tree_idx + 1}'
                            if sheet_name in bootstrap_excel_data:
                                bootstrap_excel_data[sheet_name].to_excel(writer, sheet_name=sheet_name, index=False)
                            
                            df_structure = extract_tree_structure(tree_estimator, res['features'])
                            structure_sheet_name = f'T{tree_idx + 1}_Structure'
                            df_structure.to_excel(writer, sheet_name=structure_sheet_name, index=False)
                            
                            df_feat_imp = get_tree_feature_importance(tree_estimator, res['features'])
                            feat_imp_sheet_name = f'T{tree_idx + 1}_Features'
                            df_feat_imp.to_excel(writer, sheet_name=feat_imp_sheet_name, index=False)
                    
                    output.seek(0)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"bootstrap_details_{ticker_choice}_{timestamp}.xlsx"
                    
                    st.success("✅ Excel file generated successfully!")
                    
                    st.download_button(
                        label="⬇️ Download Bootstrap Details (Excel)",
                        data=output.getvalue(),
                        file_name=filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary",
                        use_container_width=True
                    )
                    
                    n_trees_exported = min(rf_est, 10)
                    st.info(
                        f"📄 File berisi {n_trees_exported * 3 + 1} sheets:\n"
                        f"- 1 Summary sheet\n"
                        f"- {n_trees_exported} Bootstrap sheets (Tree_1 sampai Tree_{n_trees_exported})\n"
                        f"- {n_trees_exported} Structure sheets (T1_Structure sampai T{n_trees_exported}_Structure)\n"
                        f"- {n_trees_exported} Feature importance sheets (T1_Features sampai T{n_trees_exported}_Features)"
                    )
        else:
            st.warning("⚠️ openpyxl not installed. Install with: pip install openpyxl")
        
        st.markdown("---")
        
        # Visualize Bootstrap Samples
        st.markdown("#### 📈 Visualisasi Bootstrap Sampling Pattern")
        
        st.markdown(
            """
            <div class='warning-box'>
                <strong>Interpretasi Grafik:</strong>
                <ul>
                    <li><strong>Hijau (In-bag):</strong> Sample yang dipilih untuk tree ini (bisa 1x, 2x, atau lebih)</li>
                    <li><strong>Merah (OOB):</strong> Sample yang tidak terpilih sama sekali</li>
                    <li><strong>Tinggi bar:</strong> Berapa kali sample tersebut dipilih</li>
                    <li>Setiap tree punya kombinasi in-bag/OOB yang berbeda</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        n_trees_vis = min(5, rf_est)
        fig_bootstrap = visualize_bootstrap_samples(
            res['X_train_rf'], 
            res['y_train_rf'], 
            res['rf_model'],
            n_trees_to_show=n_trees_vis
        )
        st.pyplot(fig_bootstrap)
        plt.close(fig_bootstrap)
        
        st.markdown("---")
        
        # Mathematical Explanation
        st.markdown("#### 🧮 Matematika Bootstrap Sampling")
        
        st.markdown(
            """
            <div class='equation-box'>
                Probability(sample not selected in one draw) = (n-1)/n<br>
                Probability(sample not selected in n draws) = ((n-1)/n)^n<br>
                Limit as n→∞: ((n-1)/n)^n ≈ 1/e ≈ 0.368 = 36.8%<br><br>
                
                Expected proportion of unique samples = 1 - 1/e ≈ 63.2%<br>
                Expected OOB samples ≈ 36.8% of training data
            </div>
            """,
            unsafe_allow_html=True
        )
        
        n_train = len(res['X_train_rf'])
        expected_unique = int(n_train * 0.632)
        expected_oob = int(n_train * 0.368)
        
        st.markdown(
            f"""
            <div class='success-box'>
                <h4>📊 Theoretical vs Actual untuk dataset ini:</h4>
                <ul>
                    <li><strong>Total training samples:</strong> {n_train}</li>
                    <li><strong>Expected unique samples per tree:</strong> ~{expected_unique} ({63.2:.1f}%)</li>
                    <li><strong>Expected OOB samples per tree:</strong> ~{expected_oob} ({36.8:.1f}%)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Feature Subsampling
        st.markdown("#### 🎯 Feature Subsampling (sqrt features)")
        
        n_features = len(res['features'])
        n_features_per_split = int(np.sqrt(n_features))
        
        st.markdown(
            f"""
            <div class='info-box'>
                <h4>Random Forest juga melakukan feature subsampling:</h4>
                <ul>
                    <li><strong>Total features:</strong> {n_features} ({', '.join(res['features'])})</li>
                    <li><strong>Features per split:</strong> √{n_features} = {n_features_per_split}</li>
                    <li><strong>Tujuan:</strong> Mengurangi korelasi antar trees → ensemble lebih robust</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # =====================
    # TAB: HASIL PREDIKSI
    # =====================
    with tab_results:
        st.markdown(f"### 🎯 Hasil Prediksi: {ticker_choice}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);'>
                    <h4 style='color: white; margin: 0;'>🤖 RF ({rf_est} trees)</h4>
                    <h2 style='margin: 10px 0; color: white;'>Rp {res['pred_next']['rf']:,.0f}</h2>
                    <p style='color: #f0f0f0; margin: 0;'>MAPE: {res['mape_rf']*100:.2f}%</p>
                    <p style='color: #f0f0f0; margin: 0; font-size: 12px;'>Test: {res['test_samples_count']} samples</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #ff6b6b 0%, #ff8e53 100%);'>
                    <h4 style='color: white; margin: 0;'>📊 EMA 5</h4>
                    <h2 style='margin: 10px 0; color: white;'>Rp {res['pred_next']['ema5']:,.0f}</h2>
                    <p style='color: #f0f0f0; margin: 0;'>MAPE: {res['mape_ema5']*100:.2f}%</p>
                    <p style='color: #f0f0f0; margin: 0; font-size: 12px;'>Test: {res['test_samples_count_ema']} samples</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class='metric-card' style='background: linear-gradient(135deg, #4ecdc4 0%, #44a08d 100%);'>
                    <h4 style='color: white; margin: 0;'>📈 EMA 10</h4>
                    <h2 style='margin: 10px 0; color: white;'>Rp {res['pred_next']['ema10']:,.0f}</h2>
                    <p style='color: #f0f0f0; margin: 0;'>MAPE: {res['mape_ema10']*100:.2f}%</p>
                    <p style='color: #f0f0f0; margin: 0; font-size: 12px;'>Test: {res['test_samples_count_ema']} samples</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        st.markdown("#### 📈 Visualisasi Prediksi vs Aktual (SYNCHRONIZED TEST DATA)")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        ax1.plot(res['dates_test_rf'], res['y_test_rf'], 
                label='Harga Aktual', color='black', linewidth=2.5, marker='o', markersize=3)
        ax1.plot(res['dates_test_rf'], res['y_pred_rf'], 
                label=f'Prediksi Random Forest ({rf_est} trees)', linestyle='--', color='#11998e', linewidth=2, marker='s', markersize=3)
        ax1.set_title(f'Random Forest ({res["test_samples_count"]} test samples from {res["test_start_date"].strftime("%Y-%m-%d")})', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(res['dates_test_ema'], res['y_test_ema'], 
                label='Harga Aktual', color='black', linewidth=2.5, marker='o', markersize=3)
        ax2.plot(res['dates_test_ema'], res['y_pred_ema5'], 
                label='EMA 5', linestyle='-.', color='#ff6b6b', linewidth=2, marker='^', markersize=3)
        ax2.plot(res['dates_test_ema'], res['y_pred_ema10'], 
                label='EMA 10', linestyle=':', color='#4ecdc4', linewidth=2, marker='d', markersize=3)
        ax2.set_title(f'EMA Indicators ({res["test_samples_count_ema"]} test samples from {res["test_start_date"].strftime("%Y-%m-%d")})', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("---")
        
        st.markdown("#### 🎯 Feature Importance (Random Forest)")
        
        fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
        colors_imp = plt.cm.viridis(np.linspace(0.3, 0.9, len(res['feature_importance'])))
        ax_imp.barh(res['feature_importance']['feature'], 
                   res['feature_importance']['importance'],
                   color=colors_imp)
        ax_imp.set_xlabel('Importance Score', fontweight='bold')
        ax_imp.set_ylabel('Features', fontweight='bold')
        ax_imp.set_title(f'Feature Importance Analysis ({rf_est} trees ensemble)', fontweight='bold')
        ax_imp.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        st.pyplot(fig_imp)
        plt.close(fig_imp)
    
    # =====================
    # TAB: PERBANDINGAN
    # =====================
    with tab_comparison:
        st.markdown("### ⚖️ Perbandingan Model (SYNCHRONIZED TEST DATA)")
        
        comparison_df = pd.DataFrame({
            'Model': [f'Random Forest ({rf_est} trees)', 'EMA 5', 'EMA 10'],
            'MAPE (%)': [
                f"{res['mape_rf']*100:.2f}",
                f"{res['mape_ema5']*100:.2f}",
                f"{res['mape_ema10']*100:.2f}"
            ],
            'R² Score': [
                f"{res['r2_rf']:.4f}",
                f"{res['r2_ema5']:.4f}",
                f"{res['r2_ema10']:.4f}"
            ],
            'RMSE': [
                f"{res['rmse_rf']:.2f}",
                f"{res['rmse_ema5']:.2f}",
                f"{res['rmse_ema10']:.2f}"
            ],
            'Data Points': [
                f"{res['total_data_rf']}",
                f"{res['total_data_ema']}",
                f"{res['total_data_ema']}"
            ],
            'Test Samples': [
                f"{res['test_samples_count']}",
                f"{res['test_samples_count_ema']}",
                f"{res['test_samples_count_ema']}"
            ],
            'Prediksi (Rp)': [
                f"{res['pred_next']['rf']:,.0f}",
                f"{res['pred_next']['ema5']:,.0f}",
                f"{res['pred_next']['ema10']:,.0f}"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        mape_dict = {
            f'Random Forest ({rf_est} trees)': res['mape_rf'],
            'EMA 5': res['mape_ema5'],
            'EMA 10': res['mape_ema10']
        }
        winner = min(mape_dict, key=mape_dict.get)
        
        # Check if test samples match
        test_match = res['test_samples_count'] == res['test_samples_count_ema']
        match_message = (
            f"Semua model di-evaluasi pada {res['test_samples_count']} test samples yang sama" 
            if test_match 
            else f"RF: {res['test_samples_count']} test samples | EMA: {res['test_samples_count_ema']} test samples (EMA dataset lebih pendek)"
        )
        
        st.markdown(
            f"""
            <div class='success-box'>
                <h3>🏆 Model Terbaik: {winner}</h3>
                <p>MAPE terendah: {mape_dict[winner]*100:.2f}%</p>
                <p><strong>{'✅' if test_match else '⚠️'} Perbandingan:</strong> {match_message}</p>
                <p><strong>📅 Test period:</strong> Mulai dari {res['test_start_date'].strftime('%Y-%m-%d')}</p>
                {'' if test_match else '<p><em>💡 Catatan: Perbedaan jumlah test samples karena EMA memerlukan warmup period lebih panjang (10 baris vs RF 1 baris)</em></p>'}
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Comparison Charts
        st.markdown("---")
        st.markdown("#### 📊 Visualisasi Perbandingan Metrik")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_mape, ax_mape = plt.subplots(figsize=(8, 6))
            models = ['RF', 'EMA5', 'EMA10']
            mapes = [res['mape_rf']*100, res['mape_ema5']*100, res['mape_ema10']*100]
            colors_bar = ['#11998e', '#ff6b6b', '#4ecdc4']
            ax_mape.bar(models, mapes, color=colors_bar)
            ax_mape.set_ylabel('MAPE (%)', fontweight='bold')
            ax_mape.set_title('Mean Absolute Percentage Error\n(Synchronized Test Data)', fontweight='bold')
            ax_mape.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(mapes):
                ax_mape.text(i, v + 0.1, f'{v:.2f}%', ha='center', fontweight='bold')
            st.pyplot(fig_mape)
            plt.close(fig_mape)
        
        with col2:
            fig_r2, ax_r2 = plt.subplots(figsize=(8, 6))
            r2s = [res['r2_rf'], res['r2_ema5'], res['r2_ema10']]
            ax_r2.bar(models, r2s, color=colors_bar)
            ax_r2.set_ylabel('R² Score', fontweight='bold')
            ax_r2.set_title('R² Score (Coefficient of Determination)\n(Synchronized Test Data)', fontweight='bold')
            ax_r2.grid(True, alpha=0.3, axis='y')
            for i, v in enumerate(r2s):
                ax_r2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
            st.pyplot(fig_r2)
            plt.close(fig_r2)
    
    # =====================
    # TAB: DECISION TREE
    # =====================
    with tab_tree:
        st.markdown("### 🌲 Visualisasi Decision Trees dari Random Forest")
        
        st.markdown(
            f"""
            <div class='info-box'>
                <h4>ℹ️ Informasi Model Random Forest</h4>
                <ul>
                    <li><strong>Jumlah Trees:</strong> {rf_est} estimators</li>
                    <li><strong>Max Depth:</strong> {rf_dep}</li>
                    <li><strong>Training Samples:</strong> {len(res['X_train_rf'])}</li>
                    <li><strong>Testing Samples:</strong> {len(res['y_test_rf'])}</li>
                    <li><strong>Features:</strong> {', '.join(res['features'])}</li>
                    <li><strong>Bootstrap:</strong> Enabled (setiap tree dilatih pada subset data yang berbeda)</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        st.markdown("### 🌳 Render Individual Tree (Gaya Textbook / Jurnal)")

        st.info("Menggunakan sklearn.plot_tree → font lebih jelas, warna gradient berdasarkan prediksi, cocok untuk laporan/skripsi/paper")

        col_tree1, col_tree2, col_tree3 = st.columns([4, 2, 2])

        with col_tree1:
            single_tree_idx = st.selectbox(
                "Pilih nomor pohon:",
                [f"Tree #{i+1}" for i in range(min(30, rf_est))],
                index=0,
                key="single_tree_select_v2"
            )

        with col_tree2:
            single_depth_options = [None, 3, 4, 5, 6, 7, 8] + ["Full Depth"]
            single_tree_depth = st.selectbox(
                "Maks kedalaman tampilan:",
                single_depth_options,
                index=2,
                format_func=lambda x: "Full Depth" if x == "Full Depth" else f"Depth {x}" if x is not None else "Default",
                key="single_depth_select_v2"
            )

        with col_tree3:
            font_size_choice = st.select_slider(
                "Ukuran font",
                options=[8, 9, 10, 11, 12, 13],
                value=10,
                key="font_size_slider_v2"
            )

        if st.button("📊 Gambar Pohon Ini", type="primary", use_container_width=True, key="btn_render_single_v2"):
            tree_num = int(single_tree_idx.split('#')[1]) - 1
            
            display_depth = None if single_tree_depth == "Full Depth" else single_tree_depth
            
            with st.spinner(f"Menyiapkan Decision Tree #{tree_num+1} (depth {display_depth or 'full'})..."):
                fig = plot_single_decision_tree(
                    rf_model=res['rf_model'],
                    features=res['features'],
                    tree_index=tree_num,
                    max_depth=display_depth,
                    figsize=(22, 13),
                    fontsize=font_size_choice,
                    dpi=140,
                    title=f"Decision Tree Prediksi Harga {ticker_choice}"
                )
                
                st.pyplot(fig)
                plt.close(fig)
                
                buf = io.BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    bbox_inches="tight",
                    dpi=300,
                    pad_inches=0.4
                )
                buf.seek(0)
                
                depth_text = "full" if display_depth is None else f"d{int(display_depth)}"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"decision_tree_{tree_num+1}_{ticker_choice}_{depth_text}_fs{font_size_choice}_{timestamp}.png"
                
                st.download_button(
                    label="⬇️ Download PNG (resolusi tinggi 300 dpi)",
                    data=buf,
                    file_name=filename,
                    mime="image/png",
                    type="primary",
                    use_container_width=True,
                    key=f"dl_single_tree_{tree_num}"
                )
                
                tree = res['rf_model'].estimators_[tree_num]
                n_nodes = tree.tree_.node_count
                n_leaves = (tree.tree_.children_left == -1).sum()
                n_internal = n_nodes - n_leaves
                actual_depth = tree.tree_.max_depth
                
                st.markdown("#### 📊 Statistik Pohon Ini")
                cols_stat = st.columns(4)
                cols_stat[0].metric("Total Node", n_nodes)
                cols_stat[1].metric("Leaf Node", n_leaves)
                cols_stat[2].metric("Internal Node", n_internal)
                cols_stat[3].metric("Kedalaman Aktual", actual_depth)
    
    # =====================
    # TAB: HISTORY
    # =====================
    with tab_history:
        st.markdown(f"### 📜 History: {ticker_choice}")
        
        history_df = get_prediction_history(ticker=ticker_choice, limit=100)
        
        if len(history_df) > 0:
            st.dataframe(history_df, use_container_width=True, height=400)
        else:
            st.info("Belum ada history")

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: #666;'>
        <p style='margin: 0;'><strong>📈 Stock Prediction System v6.1</strong></p>
    </div>
    """,
    unsafe_allow_html=True
)