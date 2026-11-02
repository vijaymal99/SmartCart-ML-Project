import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="SmartCart Analytics",
    page_icon="🛒",
    layout="wide"
)

# ---------------- LOAD DATA ----------------

st.sidebar.title("🛒 SmartCart")

file_path = Path(__file__).resolve().parent / "smartcart_customers.csv"

uploaded_file = st.sidebar.file_uploader(
    "Upload Customer Dataset",
    type=["csv"]
)

try:
    if uploaded_file is not None:
        df = pd.read_csv(
            uploaded_file,
            sep=None,
            engine="python"
        )
    elif file_path.exists():
        df = pd.read_csv(
            file_path,
            sep=None,
            engine="python"
        )
    else:
        st.error(
            "Dataset not found. Keep smartcart_customers.csv "
            "in the same folder as aap.py."
        )
        st.stop()

except Exception as e:
    st.error(f"Could not load dataset: {e}")
    st.stop()

df.columns = df.columns.str.strip()

# ---------------- FEATURE ENGINEERING ----------------

spending_cols = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds"
]

purchase_cols = [
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumDealsPurchases"
]

available_spending = [
    c for c in spending_cols if c in df.columns
]

available_purchases = [
    c for c in purchase_cols if c in df.columns
]

if available_spending:
    df["Total_Spending"] = (
        df[available_spending]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )

if available_purchases:
    df["Total_Purchases"] = (
        df[available_purchases]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0)
        .sum(axis=1)
    )

# ---------------- MACHINE LEARNING ----------------

preferred_features = [
    "Income",
    "Year_Birth",
    "Recency",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
    "NumWebVisitsMonth",
    "Total_Spending",
    "Total_Purchases"
]

features = [
    c for c in preferred_features if c in df.columns
]

if len(features) < 2:
    features = [
        c for c in df.select_dtypes(include="number").columns
        if c.lower() not in ["id", "cluster"]
    ]

if len(features) < 2:
    st.error(
        "At least two numerical features are required "
        "for customer clustering."
    )
    st.stop()

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.dropna(axis=1, how="all")
features = X.columns.tolist()

if len(features) < 2:
    st.error("Not enough valid numerical features.")
    st.stop()

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# ---------------- SIDEBAR NAVIGATION ----------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Customer Analysis",
        "Cluster Analysis",
        "Marketing Insights"
    ]
)

n_clusters = st.sidebar.slider(
    "Number of Clusters",
    min_value=2,
    max_value=min(8, len(df)),
    value=min(4, len(df))
)

model = KMeans(
    n_clusters=n_clusters,
    random_state=42,
    n_init=10
)

df["cluster"] = model.fit_predict(X_scaled)

# ---------------- CLUSTER SUMMARY ----------------

summary = df.groupby("cluster")[features].mean()

if "Total_Spending" in summary.columns:
    ranked = summary["Total_Spending"].rank(
        method="first",
        ascending=False
    )

    segment_names = {}

    for cluster_id in summary.index:
        rank = ranked.loc[cluster_id]

        if rank == 1:
            segment_names[cluster_id] = "High Spenders"
        elif rank == len(summary):
            segment_names[cluster_id] = "Low Spenders"
        else:
            segment_names[cluster_id] = "Mid Spenders"

else:
    segment_names = {
        c: f"Customer Segment {c}"
        for c in summary.index
    }

df["Segment"] = df["cluster"].map(segment_names)

# ---------------- HEADER ----------------

st.title("🛒 SmartCart Customer Analytics")

st.caption(
    "Machine Learning-based Customer Segmentation Dashboard"
)

# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    st.subheader("Business Overview")

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Customers", f"{len(df):,}")
    c2.metric("Total Features", len(df.columns))
    c3.metric("Customer Segments", df["cluster"].nunique())

    st.divider()

    left, right = st.columns(2)

    with left:
        counts = (
            df["Segment"]
            .value_counts()
            .rename_axis("Segment")
            .reset_index(name="Customers")
        )

        fig = px.pie(
            counts,
            names="Segment",
            values="Customers",
            title="Customer Distribution",
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        if "Total_Spending" in df.columns:
            fig = px.box(
                df,
                x="Segment",
                y="Total_Spending",
                title="Spending by Segment"
            )

            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ---------------- CUSTOMER ANALYSIS ----------------

elif page == "Customer Analysis":

    st.subheader("Customer Explorer")

    if "ID" in df.columns:
        customer_id = st.selectbox(
            "Select Customer ID",
            df["ID"].dropna().unique()
        )

        st.dataframe(
            df[df["ID"] == customer_id],
            use_container_width=True
        )
    else:
        st.dataframe(
            df.head(100),
            use_container_width=True
        )

    feature = st.selectbox(
        "Select Feature",
        features
    )

    fig = px.histogram(
        df,
        x=feature,
        color="Segment",
        title=f"{feature} Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------- CLUSTER ANALYSIS ----------------

elif page == "Cluster Analysis":

    st.subheader("K-Means Clustering Results")

    counts = (
        df["cluster"]
        .value_counts()
        .sort_index()
        .rename_axis("Cluster")
        .reset_index(name="Customers")
    )

    fig = px.bar(
        counts,
        x="Cluster",
        y="Customers",
        title="Customers per Cluster",
        text="Customers"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Cluster Summary")

    st.dataframe(
        summary.round(2),
        use_container_width=True
    )

    st.download_button(
        "Download Clustered Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="smartcart_clustered_customers.csv",
        mime="text/csv"
    )

# ---------------- MARKETING INSIGHTS ----------------

elif page == "Marketing Insights":

    st.subheader("Marketing Recommendations")

    for cluster_id, name in segment_names.items():

        with st.expander(name):

            st.write(
                "Customers:",
                int((df["cluster"] == cluster_id).sum())
            )

            st.dataframe(
                summary.loc[[cluster_id]].round(2),
                use_container_width=True
            )

            if name == "High Spenders":
                st.success(
                    "Suggested strategy: Loyalty rewards "
                    "and exclusive offers."
                )

            elif name == "Low Spenders":
                st.info(
                    "Suggested strategy: Test targeted discounts "
                    "and affordable product bundles."
                )

            else:
                st.info(
                    "Suggested strategy: Analyse customer behaviour "
                    "and test personalised marketing campaigns."
                )

