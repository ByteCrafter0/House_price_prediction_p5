"""House Price Predictor – Streamlit app.

Run with:  streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE = Path(__file__).parent
MODEL_PATH = BASE / "final_house_price_gb_model.joblib"
DATA_PATH = BASE / "House_price.csv"

FEATURES = [
    "Avg. Area Income",
    "House Age",
    "Number of Rooms",
    "Number of Bedrooms",
    "Area Population",
]

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")


# ---------- Loaders ----------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def test_metrics():
    """Re-create the notebook's split (test_size=0.33, random_state=42) and score the model."""
    df = load_data()
    X, y = df[FEATURES], df["Price"]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.33, random_state=42)
    pred = load_model().predict(X_te)
    return {
        "R²": r2_score(y_te, pred),
        "MAE": mean_absolute_error(y_te, pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_te, pred))),
        "y_true": y_te.values,
        "y_pred": pred,
    }


def predict(model, row: dict) -> float:
    return float(model.predict(pd.DataFrame([row], columns=FEATURES))[0])


model = load_model()
df = load_data()

# ---------- Sidebar inputs ----------
st.sidebar.header("🏡 House details")
st.sidebar.caption("Defaults are the dataset medians. Ranges follow the training data.")


def slider(label, fmt, step, help_=None):
    lo, hi, med = float(df[label].min()), float(df[label].max()), float(df[label].median())
    return st.sidebar.slider(label, lo, hi, med, step=step, format=fmt, help=help_)


inputs = {
    "Avg. Area Income": slider("Avg. Area Income", "$%.0f", 500.0, "Average income of residents in the area"),
    "House Age": slider("House Age", "%.1f yrs", 0.1, "Average age of houses in the area"),
    "Number of Rooms": slider("Number of Rooms", "%.1f", 0.1, "Average number of rooms"),
    "Number of Bedrooms": slider("Number of Bedrooms", "%.1f", 0.1, "Average number of bedrooms"),
    "Area Population": slider("Area Population", "%.0f", 100.0, "Population of the area"),
}

if inputs["Number of Bedrooms"] > inputs["Number of Rooms"]:
    st.sidebar.warning("Bedrooms exceed total rooms – double-check your inputs.")

# ---------- Header ----------
st.title("🏠 House Price Predictor")
st.caption("Tuned Gradient Boosting model · predicts price from area & house characteristics")

tab_pred, tab_whatif, tab_model, tab_data = st.tabs(
    ["💰 Prediction", "🎚️ What-if analysis", "📊 Model insights", "🗂️ Data explorer"]
)

price = predict(model, inputs)
metrics = test_metrics()

# ---------- Tab 1: Prediction ----------
with tab_pred:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted price", f"${price:,.0f}")
    avg = df["Price"].mean()
    c2.metric("vs. dataset average", f"${price - avg:+,.0f}", f"{(price / avg - 1) * 100:+.1f}%")
    pct = (df["Price"] < price).mean() * 100
    c3.metric("Price percentile", f"{pct:.0f}th")

    st.markdown(
        f"Typical error on unseen data is about **±${metrics['MAE']:,.0f}** (MAE), "
        f"so a reasonable range is roughly **${price - metrics['MAE']:,.0f} – ${price + metrics['MAE']:,.0f}**."
    )

    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x="Price", nbins=50, title="Where this house sits in the market",
                           color_discrete_sequence=["#9db4d6"])
        fig.add_vline(x=price, line_color="#d62728", line_width=3,
                      annotation_text="Prediction", annotation_position="top")
        fig.update_layout(showlegend=False, margin=dict(t=50, b=10))
        st.plotly_chart(fig, width='stretch')
    with right:
        # Input profile vs dataset: percentile of each input
        prof = pd.DataFrame({
            "Feature": FEATURES,
            "Percentile": [(df[f] < inputs[f]).mean() * 100 for f in FEATURES],
        })
        fig = px.bar(prof, x="Percentile", y="Feature", orientation="h", range_x=[0, 100],
                     title="Your inputs vs. the dataset (percentile)",
                     color_discrete_sequence=["#4c78a8"])
        fig.update_layout(margin=dict(t=50, b=10), yaxis_title=None)
        st.plotly_chart(fig, width='stretch')

    with st.expander("Batch predictions from a CSV"):
        st.write("Upload a CSV containing these columns: " + ", ".join(f"`{f}`" for f in FEATURES))
        up = st.file_uploader("CSV file", type="csv")
        if up is not None:
            try:
                batch = pd.read_csv(up)
                missing = [f for f in FEATURES if f not in batch.columns]
                if missing:
                    st.error(f"Missing columns: {missing}")
                else:
                    batch["Predicted Price"] = model.predict(batch[FEATURES])
                    st.dataframe(batch, width='stretch')
                    st.download_button("Download predictions", batch.to_csv(index=False).encode(),
                                       "predictions.csv", "text/csv")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not process file: {e}")

# ---------- Tab 2: What-if ----------
with tab_whatif:
    st.subheader("How does each feature move the price?")
    st.caption("Vary one feature across its range while holding the others at your sidebar values.")
    feat = st.selectbox("Feature to vary", FEATURES)
    grid = np.linspace(df[feat].min(), df[feat].max(), 60)
    sweep = pd.DataFrame([{**inputs, feat: v} for v in grid])
    sweep["Predicted Price"] = model.predict(sweep[FEATURES])
    fig = px.line(sweep, x=feat, y="Predicted Price")
    fig.add_trace(go.Scatter(x=[inputs[feat]], y=[price], mode="markers",
                             marker=dict(size=12, color="#d62728"), name="Current"))
    fig.update_layout(margin=dict(t=20), yaxis_tickprefix="$")
    st.plotly_chart(fig, width='stretch')

    st.subheader("Compare two scenarios")
    a, b = st.columns(2)
    with a:
        st.markdown("**Scenario A** (sidebar values)")
        st.metric("Price", f"${price:,.0f}")
    with b:
        st.markdown("**Scenario B**")
        b_in = {}
        for f in FEATURES:
            b_in[f] = st.number_input(f, float(df[f].min()), float(df[f].max()),
                                      float(inputs[f]), key=f"b_{f}")
        price_b = predict(model, b_in)
        st.metric("Price", f"${price_b:,.0f}", f"${price_b - price:+,.0f} vs A")

# ---------- Tab 3: Model insights ----------
with tab_model:
    st.subheader("Held-out test performance")
    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{metrics['R²']:.3f}")
    m2.metric("MAE", f"${metrics['MAE']:,.0f}")
    m3.metric("RMSE", f"${metrics['RMSE']:,.0f}")
    st.caption("Same split as the notebook (test_size=0.33, random_state=42).")

    l, r = st.columns(2)
    with l:
        imp = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": model.named_steps["classifier"].feature_importances_,
        }).sort_values("Importance")
        fig = px.bar(imp, x="Importance", y="Feature", orientation="h",
                     title="Feature importance", color_discrete_sequence=["#4c78a8"])
        fig.update_layout(margin=dict(t=50), yaxis_title=None)
        st.plotly_chart(fig, width='stretch')
    with r:
        fig = px.scatter(x=metrics["y_true"], y=metrics["y_pred"], opacity=0.4,
                         labels={"x": "Actual price", "y": "Predicted price"},
                         title="Predicted vs. actual (test set)")
        lo, hi = metrics["y_true"].min(), metrics["y_true"].max()
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                 line=dict(color="#d62728", dash="dash"), name="Perfect"))
        fig.update_layout(margin=dict(t=50), showlegend=False)
        st.plotly_chart(fig, width='stretch')

    with st.expander("Model hyperparameters"):
        p = model.named_steps["classifier"].get_params()
        st.json({k: p[k] for k in ["n_estimators", "learning_rate", "max_depth",
                                   "min_samples_split", "subsample"]})

# ---------- Tab 4: Data explorer ----------
with tab_data:
    st.subheader("Training data")
    st.write(f"{len(df):,} rows · {df.shape[1]} columns")
    show = df.drop(columns=["Address"]) if "Address" in df else df
    st.dataframe(show.describe().T, width='stretch')

    x = st.selectbox("X axis", FEATURES, key="ex_x")
    fig = px.scatter(df.sample(min(2000, len(df)), random_state=1), x=x, y="Price",
                     opacity=0.4, trendline="ols", trendline_color_override="#d62728")
    st.plotly_chart(fig, width='stretch')

    corr = show.corr(numeric_only=True)
    st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                              zmin=-1, zmax=1, title="Correlation matrix"),
                    width='stretch')
    with st.expander("Browse rows"):
        st.dataframe(df, width='stretch')