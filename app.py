"""
Dashboard Forecasting QTY per Kategori Barang
Sumber data : Google Spreadsheet (sheet 'MASTER DATA')
Model       : Holt-Winters, XGBoost, Croston, SBA  (pipeline dari notebook
              FORECASTING_DAILY_ERRATIC & LUMPY, diperluas ke harian/mingguan/bulanan)
"""
import re
import warnings

import gspread
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from google.oauth2.service_account import Credentials
from scipy.stats import pearsonr
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Dashboard Forecasting QTY", layout="wide")

# ------------------------------------------------------------------
# KONFIGURASI
# ------------------------------------------------------------------
SHEET_NAME = "MASTER DATA"
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
]

COL_DATE = "Tanggal"
COL_BRAND = "Brand"
COL_CATEGORY = "Nama Kategori Barang Barang & Jasa"
COL_QTY = "QTY"

ALPHA_CROSTON = 0.1  # sama seperti notebook

# Parameter tiap periode. Harian = persis notebook (lag 1,2,3,7,14; backtest 7 hari).
PERIODS = {
    "Harian": dict(freq="D", season=7, n_test=7, lags=[1, 2, 3, 7, 14],
                   rolls=(7, 14), horizon=7, cal="D", fmt="%Y-%m-%d", unit="hari"),
    "Mingguan": dict(freq="W", season=52, n_test=8, lags=[1, 2, 3, 4],
                     rolls=(4, 8), horizon=8, cal="W", fmt="%Y-%m-%d", unit="minggu"),
    "Bulanan": dict(freq="MS", season=12, n_test=6, lags=[1, 2, 3, 6],
                    rolls=(3, 6), horizon=6, cal="M", fmt="%Y-%m", unit="bulan"),
}


def min_length(cfg) -> int:
    need = max(max(cfg["lags"]), cfg["rolls"][1])
    return need + cfg["n_test"] + 10


# ------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------
def to_number(x) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    s = re.sub(r"[^\d,.\-]", "", str(x).strip())
    if s in ("", "-"):
        return np.nan
    if re.fullmatch(r"-?\d{1,3}(\.\d{3})+", s):
        s = s.replace(".", "")
    elif "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


@st.cache_data(ttl=600, show_spinner="Memuat data dari Google Spreadsheet...")
def load_data() -> pd.DataFrame:
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES
    )
    client = gspread.authorize(creds)
    ws = client.open_by_key(st.secrets["spreadsheet_id"]).worksheet(SHEET_NAME)

    values = ws.get_all_values()
    header = [h.strip() for h in values[0]]
    df = pd.DataFrame(values[1:], columns=header)
    df = df.loc[:, [c for c in df.columns if c != ""]]

    df[COL_DATE] = pd.to_datetime(df[COL_DATE], dayfirst=True, errors="coerce")
    df[COL_QTY] = df[COL_QTY].apply(to_number)
    df[COL_BRAND] = df[COL_BRAND].astype(str).str.strip()
    df[COL_CATEGORY] = df[COL_CATEGORY].astype(str).str.strip()

    df = df.dropna(subset=[COL_DATE, COL_QTY])
    df = df[(df[COL_BRAND] != "") & (df[COL_CATEGORY] != "")]
    return df


def build_pivot(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Pivot: index = periode, kolom = kategori barang, isi = total QTY."""
    pivot = (
        df.groupby([pd.Grouper(key=COL_DATE, freq=freq), COL_CATEGORY])[COL_QTY]
        .sum()
        .unstack(fill_value=0)
    )
    full_idx = pd.date_range(pivot.index.min(), pivot.index.max(), freq=freq)
    return pivot.reindex(full_idx, fill_value=0)


# ------------------------------------------------------------------
# HELPER MODEL (dari notebook, digeneralisasi per periode)
# ------------------------------------------------------------------
def calendar_frame(idx: pd.DatetimeIndex, kind: str) -> pd.DataFrame:
    c = pd.DataFrame(index=idx)
    if kind == "D":
        c["dow"] = idx.dayofweek
        c["day"] = idx.day
        c["month"] = idx.month
        c["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    elif kind == "W":
        c["week"] = idx.isocalendar().week.astype(int).values
        c["month"] = idx.month
    else:
        c["month"] = idx.month
    return c


def make_features(series: pd.Series, cfg) -> pd.DataFrame:
    d = pd.DataFrame({"qty": series})
    d["log_qty"] = np.log1p(d["qty"].clip(lower=0))
    for lag in cfg["lags"]:
        d[f"lag_{lag}"] = d["log_qty"].shift(lag)
    w1, w2 = cfg["rolls"]
    d[f"roll_mean_{w1}"] = d["log_qty"].shift(1).rolling(w1).mean()
    d[f"roll_mean_{w2}"] = d["log_qty"].shift(1).rolling(w2).mean()
    d[f"roll_std_{w1}"] = d["log_qty"].shift(1).rolling(w1).std()
    d[f"roll_median_{w1}"] = d["log_qty"].shift(1).rolling(w1).median()
    return d.join(calendar_frame(d.index, cfg["cal"]))


def feature_row(log_hist: pd.Series, date, cfg) -> dict:
    w1, w2 = cfg["rolls"]
    row = {f"lag_{lag}": log_hist.iloc[-lag] for lag in cfg["lags"]}
    row[f"roll_mean_{w1}"] = log_hist.iloc[-w1:].mean()
    row[f"roll_mean_{w2}"] = log_hist.iloc[-w2:].mean()
    row[f"roll_std_{w1}"] = log_hist.iloc[-w1:].std()
    row[f"roll_median_{w1}"] = log_hist.iloc[-w1:].median()
    row.update(calendar_frame(pd.DatetimeIndex([date]), cfg["cal"]).iloc[0].to_dict())
    return row


def new_xgb() -> XGBRegressor:
    return XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        n_jobs=1,  # hemat CPU di Streamlit Cloud
    )


def xgb_recursive(model, history: pd.Series, dates, cfg, feature_cols) -> pd.Series:
    hist = history.copy()
    preds = []
    for date in dates:
        log_hist = np.log1p(hist.clip(lower=0))
        x = pd.DataFrame([feature_row(log_hist, date, cfg)])[feature_cols]
        p = max(np.expm1(model.predict(x)[0]), 0)
        preds.append(p)
        hist.loc[date] = p
    return pd.Series(preds, index=dates)


def fit_hw(qty: pd.Series, cfg):
    y = np.log1p(qty.clip(lower=0))
    kw = dict(trend="add", damped_trend=True, initialization_method="estimated")
    m = cfg["season"]
    if len(y) >= 2 * m + 2:  # musiman hanya dipakai kalau data cukup
        kw.update(seasonal="add", seasonal_periods=m)
    return ExponentialSmoothing(y, **kw).fit(optimized=True)


def croston_method(ts, alpha=0.1, sba=False):
    d = np.asarray(ts, dtype=float)
    n = len(d)
    a, p, f = np.zeros(n + 1), np.zeros(n + 1), np.zeros(n + 1)

    first_idx = np.argmax(d > 0)
    a[0] = d[first_idx] if d[first_idx] > 0 else 1e-6
    p[0] = first_idx + 1
    f[0] = a[0] / p[0]

    q = 1
    for t in range(n):
        if d[t] > 0:
            a[t + 1] = alpha * d[t] + (1 - alpha) * a[t]
            p[t + 1] = alpha * q + (1 - alpha) * p[t]
            q = 1
        else:
            a[t + 1], p[t + 1] = a[t], p[t]
            q += 1
        f[t + 1] = a[t + 1] / p[t + 1]

    nxt = f[-1]
    if sba:
        nxt *= 1 - alpha / 2
    return f[1:], nxt


def evaluate_model(actual, predicted, max_lag=3):
    """Best Lag (pergeseran periode dgn korelasi tertinggi), MAE, RMSE, MAPE, Correlation."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    n = len(actual)

    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, p = actual[-lag:], predicted[: n + lag]
        elif lag > 0:
            a, p = actual[: n - lag], predicted[lag:]
        else:
            a, p = actual, predicted
        if len(a) < 3 or np.std(a) == 0 or np.std(p) == 0:
            continue
        c = np.corrcoef(a, p)[0, 1]
        if not np.isnan(c) and c > best_corr:
            best_corr, best_lag = c, lag

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mask = actual != 0
    mape = (np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
            if mask.sum() > 0 else np.nan)
    if np.std(actual) == 0 or np.std(predicted) == 0:
        corr0 = np.nan
    else:
        corr0, _ = pearsonr(actual, predicted)

    return {
        "Best Lag": best_lag,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "MAPE (%)": round(mape, 2) if not np.isnan(mape) else np.nan,
        "Correlation": round(corr0, 3) if not np.isnan(corr0) else np.nan,
    }


# ------------------------------------------------------------------
# PIPELINE: backtest 4 model -> pilih MAE terkecil -> forecast
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Melatih model...")
def run_pipeline(series: pd.Series, period: str, horizon: int):
    cfg = PERIODS[period]
    n_test = cfg["n_test"]
    train, test = series.iloc[:-n_test], series.iloc[-n_test:]

    preds, skipped = {}, {}

    # A. Holt-Winters
    try:
        fit = fit_hw(train, cfg)
        preds["Holt-Winters"] = pd.Series(
            np.expm1(fit.forecast(n_test)).clip(lower=0).values, index=test.index
        )
    except Exception as e:
        skipped["Holt-Winters"] = str(e)

    # B. XGBoost (backtest rekursif)
    feature_cols, feat_clean = None, None
    try:
        feat = make_features(series, cfg)
        feature_cols = [c for c in feat.columns if c not in ("qty", "log_qty")]
        feat_clean = feat.dropna(subset=feature_cols)
        train_feat = feat_clean.loc[: train.index.max()]
        model = new_xgb().fit(train_feat[feature_cols], train_feat["log_qty"])
        preds["XGBoost"] = xgb_recursive(model, train, test.index, cfg, feature_cols)
    except Exception as e:
        skipped["XGBoost"] = str(e)

    # C & D. Croston dan SBA
    try:
        _, c_next = croston_method(train, ALPHA_CROSTON, sba=False)
        _, s_next = croston_method(train, ALPHA_CROSTON, sba=True)
        preds["Croston"] = pd.Series([c_next] * n_test, index=test.index)
        preds["SBA"] = pd.Series([s_next] * n_test, index=test.index)
    except Exception as e:
        skipped["Croston/SBA"] = str(e)

    if not preds:
        return None

    # Tabel backtest
    cmp = pd.DataFrame({"Aktual": test.values}, index=test.index)
    for name, p in preds.items():
        cmp[name] = p.round(1).values
    for name in preds:
        cmp[f"Err {name}"] = (cmp["Aktual"] - cmp[name]).abs()

    mae_s = pd.Series({m: cmp[f"Err {m}"].mean() for m in preds}).sort_values()
    mae_df = mae_s.rename("MAE").rename_axis("Model").reset_index()
    eval_df = pd.DataFrame(
        {m: evaluate_model(cmp["Aktual"], cmp[m]) for m in preds}
    ).T.rename_axis("Model").reset_index()

    best = mae_s.index[0]
    best_mae = float(mae_s.iloc[0])
    best_lag = int(eval_df.loc[eval_df["Model"] == best, "Best Lag"].iloc[0])

    # Forecast ke depan dengan model terbaik (fit ulang di seluruh data)
    future = pd.date_range(series.index[-1], periods=horizon + 1, freq=cfg["freq"])[1:]
    if best == "Holt-Winters":
        hw = fit_hw(series, cfg)
        q10, q90 = hw.resid.quantile([0.1, 0.9])
        fc_log = hw.forecast(horizon)
        vals = np.expm1(fc_log).clip(lower=0).values
        lower = np.expm1(fc_log + q10).clip(lower=0).values
        upper = np.expm1(fc_log + q90).values
    else:
        if best == "XGBoost":
            model = new_xgb().fit(feat_clean[feature_cols], feat_clean["log_qty"])
            vals = xgb_recursive(model, series, future, cfg, feature_cols).values
        else:
            _, nxt = croston_method(series, ALPHA_CROSTON, sba=(best == "SBA"))
            vals = np.array([nxt] * horizon)
        lower = np.clip(vals - best_mae, 0, None)
        upper = vals + best_mae

    fc_df = pd.DataFrame({
        "Periode": future,
        "Model": best,
        "Forecast (P50)": np.round(vals, 1),
        "Lower (P10)": np.round(lower, 1),
        "Upper (P90)": np.round(upper, 1),
    })

    return dict(cmp=cmp, models=list(preds), mae_df=mae_df, eval_df=eval_df,
                fc_df=fc_df, best=best, best_mae=best_mae, best_lag=best_lag,
                skipped=skipped)


# ------------------------------------------------------------------
# UI
# ------------------------------------------------------------------
st.title("📈 Dashboard Forecasting QTY")

try:
    df_all = load_data()
except Exception as e:
    st.error(f"Gagal memuat data dari Google Spreadsheet: {e}")
    st.stop()

if df_all.empty:
    st.warning("Data kosong setelah dibersihkan. Periksa isi sheet 'MASTER DATA'.")
    st.stop()

# ---- Filter ----
c1, c2, c3, c4 = st.columns([2, 3, 2, 1.5])

brand_in = c1.selectbox("Brand", ["Semua Brand"] + sorted(df_all[COL_BRAND].unique()))
df_f = df_all if brand_in == "Semua Brand" else df_all[df_all[COL_BRAND] == brand_in]

category_in = c2.selectbox(
    "Kategori Barang", ["Semua Kategori"] + sorted(df_f[COL_CATEGORY].unique())
)
period_in = c3.selectbox("Periode Forecast", list(PERIODS.keys()))
cfg_in = PERIODS[period_in]
horizon_in = int(c4.number_input(
    f"Forecast ke depan ({cfg_in['unit']})", 1, 365, cfg_in["horizon"]
))

if category_in != "Semua Kategori":
    df_f = df_f[df_f[COL_CATEGORY] == category_in]
if df_f.empty:
    st.warning("Tidak ada data untuk filter yang dipilih.")
    st.stop()

# ---- Tombol mulai ----
run_clicked = st.button("🚀 Mulai FORECAST", type="primary")

if run_clicked:
    # Komputasi berat HANYA jalan saat tombol diklik
    pivot = build_pivot(df_f[[COL_DATE, COL_CATEGORY, COL_QTY]], cfg_in["freq"])
    series_new = (pivot.sum(axis=1) if category_in == "Semua Kategori"
                  else pivot[category_in])
    series_new.name = "QTY"

    if len(series_new) < min_length(cfg_in):
        st.session_state.pop("fc", None)
        st.warning(
            f"Data hanya {len(series_new)} periode {period_in.lower()}; minimal "
            f"{min_length(cfg_in)} periode dibutuhkan untuk backtest dan pelatihan model. "
            "Coba periode yang lebih kecil atau filter yang lebih luas."
        )
        st.stop()

    res_new = run_pipeline(series_new, period_in, horizon_in)
    if res_new is None:
        st.session_state.pop("fc", None)
        st.error("Semua model gagal dilatih untuk data ini.")
        st.stop()

    # Simpan hasil + parameter yang dipakai, supaya tampilan tidak hilang
    # saat rerun (mis. klik tombol unduh) dan tidak berubah sebelum klik lagi.
    st.session_state["fc"] = dict(
        res=res_new, series=series_new, brand=brand_in, category=category_in,
        period=period_in, horizon=horizon_in,
        total_qty=float(df_f[COL_QTY].sum()),
    )

# Belum pernah klik -> berhenti (tidak ada komputasi berat)
if "fc" not in st.session_state:
    st.info("Pilih Brand, Kategori Barang, dan Periode Forecast, lalu klik **Mulai FORECAST**.")
    st.stop()

# Ambil hasil tersimpan
S = st.session_state["fc"]
res, series = S["res"], S["series"]
brand, category = S["brand"], S["category"]
period, horizon = S["period"], S["horizon"]
cfg = PERIODS[period]

# Peringatan kalau filter berbeda dari hasil yang sedang ditampilkan
if (brand_in, category_in, period_in, horizon_in) != (brand, category, period, horizon):
    st.warning("Filter sudah berubah. Klik **Mulai FORECAST** untuk memperbarui hasil.")

# ---- Sidebar ----
with st.sidebar:
    st.header("Ringkasan")
    st.metric("Total Data", f"{len(series):,}")
    st.metric("Total QTY", f"{S['total_qty']:,.0f}")
    st.metric("Model yang digunakan", res["best"])
    st.metric("Best Lag", f"{res['best_lag']:+d}")
    st.metric("MAE", f"{res['best_mae']:,.2f}")
    st.caption(
        f"Total Data = jumlah periode {period.lower()}. MAE dari backtest "
        f"{cfg['n_test']} {cfg['unit']} terakhir; model terbaik = MAE terkecil. "
        "Best Lag = pergeseran periode (±3) dengan korelasi aktual-prediksi tertinggi."
    )
    if st.button("🔄 Muat ulang data"):
        st.cache_data.clear()
        st.session_state.pop("fc", None)
        st.rerun()

st.markdown(f"### {brand} - {category}")
st.info(f"Rata-rata kesalahan prediksi (MAE) adalah **{res['best_mae']:,.2f}** QTY per {cfg['unit']}")
if res["skipped"]:
    st.caption("Model dilewati: " + "; ".join(f"{k} ({v[:60]})" for k, v in res["skipped"].items()))

tab1, tab2 = st.tabs(["Actual vs Prediksi", "Forecast"])

# ---- Tab 1 ----
with tab1:
    st.subheader(f"Grafik Actual vs Prediksi – backtest {cfg['n_test']} {cfg['unit']} terakhir")
    cmp = res["cmp"]
    tail = series.iloc[-(cfg["n_test"] * 4):]

    fig1 = go.Figure()
    fig1.add_vrect(x0=cmp.index[0], x1=cmp.index[-1], fillcolor="gray",
                   opacity=0.15, line_width=0,
                   annotation_text="Data uji", annotation_position="top left")
    fig1.add_trace(go.Scatter(x=tail.index, y=tail.values, name="Aktual",
                              mode="lines+markers", line=dict(color="#1f77b4")))
    for m in res["models"]:
        is_best = m == res["best"]
        fig1.add_trace(go.Scatter(
            x=cmp.index, y=cmp[m], name=f"{m} (terbaik)" if is_best else m,
            mode="lines+markers",
            line=dict(width=3 if is_best else 1.5, dash="solid" if is_best else "dot"),
            visible=True if is_best else "legendonly",
        ))
    fig1.update_layout(hovermode="x unified", yaxis_title="QTY", height=500)
    st.plotly_chart(fig1, use_container_width=True)
    st.caption("Klik nama model di legenda untuk menampilkan model lain.")

    st.subheader("Data Prediction Test")
    tbl = cmp.copy()
    tbl.insert(0, "Periode", tbl.index.strftime(cfg["fmt"]))
    st.dataframe(tbl.reset_index(drop=True), use_container_width=True, hide_index=True)

    with st.expander("Perbandingan model (MAE & matriks evaluasi)"):
        st.dataframe(res["mae_df"].round(2), use_container_width=True, hide_index=True)
        st.dataframe(res["eval_df"], use_container_width=True, hide_index=True)

# ---- Tab 2 ----
with tab2:
    fc = res["fc_df"]
    st.subheader(f"Forecast {int(horizon)} {cfg['unit']} ke depan – {res['best']}")
    hist = series.iloc[-max(int(horizon) * 3, 30):]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=hist.index, y=hist.values, name="Aktual",
                              mode="lines+markers"))
    fig2.add_trace(go.Scatter(x=fc["Periode"], y=fc["Upper (P90)"], line=dict(width=0),
                              showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(x=fc["Periode"], y=fc["Lower (P10)"], line=dict(width=0),
                              fill="tonexty", fillcolor="rgba(220,20,60,0.15)",
                              name="Rentang P10–P90", hoverinfo="skip"))
    fig2.add_trace(go.Scatter(
        x=[hist.index[-1], *fc["Periode"]],
        y=[hist.values[-1], *fc["Forecast (P50)"]],
        name="Forecast", mode="lines+markers", line=dict(dash="dash", color="crimson"),
    ))
    fig2.update_layout(hovermode="x unified", yaxis_title="QTY", height=500)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Tabel Forecast")
    tbl_fc = fc.copy()
    tbl_fc["Periode"] = tbl_fc["Periode"].dt.strftime(cfg["fmt"])
    st.dataframe(tbl_fc, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Unduh hasil forecast (CSV)",
                       tbl_fc.to_csv(index=False).encode("utf-8"),
                       file_name="forecast_qty.csv", mime="text/csv")
