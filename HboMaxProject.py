import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
st.set_page_config(page_title="HBO Max Movies Popularity & Prediction", layout="wide")

st.title("HBO Max Movies – Popularity & Prediction Dashboard")

# =========================
# Load & clean data
# =========================
df = pd.read_excel("HBO_Movies_OMDb_votes_only.xlsx")

# Basic drops
drop_cols = [
    "index",
    "type",
    "imdb_bucket"
]
df = df.drop(columns=drop_cols, errors="ignore")

# Keep selected platforms only
platform_keep = [
    "platforms_netflix",
    "platforms_amazon_prime",
    "platforms_hulu_plus",
    "platforms_showtime",
    "platforms_starz"
]
platform_cols = [c for c in df.columns if c.startswith("platforms_")]
platform_drop = [c for c in platform_cols if c not in platform_keep]
df = df.drop(columns=platform_drop, errors="ignore")

# Missing summary BEFORE cleaning (for display)
df_missing_counts = df.isna().sum().sort_values(ascending=False)
df_missing_pct = (df.isna().mean().sort_values(ascending=False) * 100).round(2)
n_rows = len(df)

# Basic imputations
df["rating"] = df["rating"].fillna("Unrated")
df["rotten_score"] = df["rotten_score"].fillna(df["rotten_score"].mean())
df = df.dropna(subset=["imdb_score"])
df = df.dropna(subset=["imdb_votes"])
df = df.drop(columns=["title"], errors="ignore")

# Separate copy for EDA with titles kept
df_plot = pd.read_excel("HBO_Movies_OMDb_votes_only.xlsx")
df_plot["imdb_votes_millions"] = df_plot["imdb_votes"] / 1_000_000

# =========================
# Sidebar filters (EDA only)
# =========================
st.sidebar.header("Interactive Filters (EDA Only)")

# Decade filter
decades_all = sorted(df_plot["decade"].dropna().unique())
selected_decades = st.sidebar.multiselect(
    "Decades to include",
    decades_all,
    default=decades_all
)

# Votes range filter
min_votes = int(df_plot["imdb_votes"].min())
max_votes = int(df_plot["imdb_votes"].max())
votes_range = st.sidebar.slider(
    "Filter by IMDb votes range",
    min_value=min_votes,
    max_value=max_votes,
    value=(min_votes, max_votes),
    step=1000
)

# Optional genre filter
genre_cols_plot = [c for c in df_plot.columns if c.startswith("genres_")]
genre_labels = [g.replace("genres_", "") for g in genre_cols_plot]
selected_genre_label = st.sidebar.selectbox(
    "Filter by genre (optional)",
    ["All Genres"] + genre_labels
)

# Build filtered EDA frame
df_eda = df_plot.copy()
df_eda = df_eda[
    (df_eda["imdb_votes"] >= votes_range[0]) &
    (df_eda["imdb_votes"] <= votes_range[1])
]

if selected_decades:
    df_eda = df_eda[df_eda["decade"].isin(selected_decades)]
else:
    st.sidebar.warning("No decade selected – all decades hidden in EDA.")

if selected_genre_label != "All Genres":
    genre_col_name = "genres_" + selected_genre_label
    if genre_col_name in df_eda.columns:
        df_eda = df_eda[df_eda[genre_col_name] == 1]
    else:
        st.sidebar.warning(f"Genre column '{genre_col_name}' not found in data.")

# =========================
# Modeling setup (unchanged logic)
# =========================
df["log_votes"] = np.log1p(df["imdb_votes"])
y = df["log_votes"]
X = df.drop(columns=["imdb_votes", "log_votes"])

categorical_cols = ["rating", "decade"]
numeric_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

linear_model = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression())
])

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X, y, test_size=0.2, random_state=42
)

linear_model.fit(X_train_lr, y_train_lr)
y_pred_lr = linear_model.predict(X_test_lr)

mse_lr = mean_squared_error(y_test_lr, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)
mae_lr = mean_absolute_error(y_test_lr, y_pred_lr)
r2_lr = r2_score(y_test_lr, y_pred_lr)

# Decision tree (one-hot)
cat_cols_all = [c for c in X.columns if X[c].dtype == "object"]
X_encoded = pd.get_dummies(X, columns=cat_cols_all, drop_first=True)

X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

dt = DecisionTreeRegressor(random_state=42)
param_grid_dt = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_leaf": [1, 5, 10, 20]
}

grid_dt = GridSearchCV(
    estimator=dt,
    param_grid=param_grid_dt,
    cv=5,
    scoring="r2",
    n_jobs=-1
)
grid_dt.fit(X_train_tree, y_train_tree)
best_dt = grid_dt.best_estimator_

y_pred_dt = best_dt.predict(X_test_tree)

mse_dt = mean_squared_error(y_test_tree, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
mae_dt = mean_absolute_error(y_test_tree, y_pred_dt)
r2_dt = r2_score(y_test_tree, y_pred_dt)

y_test_dt_orig = np.expm1(y_test_tree)
y_pred_dt_orig = np.expm1(y_pred_dt)
mse_dt_orig = mean_squared_error(y_test_dt_orig, y_pred_dt_orig)
rmse_dt_orig = np.sqrt(mse_dt_orig)
mae_dt_orig = mean_absolute_error(y_test_dt_orig, y_pred_dt_orig)

# Random forest
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid_rf = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_leaf": [1, 5]
}

X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

grid_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_rf,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

grid_rf.fit(X_train_rf, y_train_rf)
best_rf = grid_rf.best_estimator_

y_pred_rf = best_rf.predict(X_test_rf)

mse_rf = mean_squared_error(y_test_rf, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)
mae_rf = mean_absolute_error(y_test_rf, y_pred_rf)
r2_rf = r2_score(y_test_rf, y_pred_rf)

y_test_rf_orig = np.expm1(y_test_rf)
y_pred_rf_orig = np.expm1(y_pred_rf)

mse_rf_orig = mean_squared_error(y_test_rf_orig, y_pred_rf_orig)
rmse_rf_orig = np.sqrt(mse_rf_orig)
mae_rf_orig = mean_absolute_error(y_test_rf_orig, y_pred_rf_orig)

# Linear model on original scale (for comparison table)
y_pred_lr_orig = np.expm1(y_pred_lr)
rmse_lr_orig = np.sqrt(mean_squared_error(np.expm1(y_test_lr), y_pred_lr_orig))
mae_lr_orig = mean_absolute_error(np.expm1(y_test_lr), y_pred_lr_orig)

model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Decision Tree", "Random Forest"],
    "R² (log scale)": [r2_lr, r2_dt, r2_rf],
    "RMSE (votes)":   [rmse_lr_orig, rmse_dt_orig, rmse_rf_orig],
    "MAE (votes)":    [mae_lr_orig, mae_dt_orig, mae_rf_orig]
})

# =========================
# Tabs
# =========================
tabs = st.tabs(["Data Overview", "Exploratory Analysis", "Models & Performance", "Model Comparison"])

# ---------- TAB 0: Data Overview ----------
with tabs[0]:
    st.subheader("Cleaned Dataset Overview")
    st.write(df.head())

    st.subheader("Missing Values Summary (Before Cleaning)")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Total missing values per column")
        st.write(df_missing_counts)
    with col2:
        st.write("Percentage missing per column")
        st.write(df_missing_pct)
    st.write(f"Total rows in original dataset: {n_rows}")

# ---------- TAB 1: EDA ----------
with tabs[1]:
    st.subheader("How Popular Are HBO Max Movies? (IMDb Votes)")

    if df_eda.empty:
        st.warning("No movies match the current filters. Adjust filters in the sidebar to see plots.")
    else:
        # Histogram
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.histplot(df_eda["imdb_votes"], bins=50, kde=True, color="#3182bd", ax=ax1)
        ax1.set_title("How Popular Are HBO Max Movies? (Filtered IMDb Votes)", fontsize=16)
        ax1.set_xlabel("IMDb Votes")
        ax1.set_ylabel("Number of Movies")
        fig1.tight_layout()
        st.pyplot(fig1)

        # Top N slider
        top_n = st.slider("Number of top movies to display", min_value=5, max_value=50, value=20, step=5)

        topN = df_eda.nlargest(top_n, "imdb_votes")[["title", "imdb_votes"]].copy()
        topN = topN.sort_values("imdb_votes", ascending=False)
        topN["imdb_votes_millions"] = topN["imdb_votes"] / 1_000_000

        st.subheader(f"Top {top_n} Most Popular HBO Max Movies (Filtered, IMDb Votes in Millions)")
        fig2, ax2 = plt.subplots(figsize=(12, 10))
        sns.barplot(
            data=topN,
            x="imdb_votes_millions",
            y="title",
            palette="viridis",
            ax=ax2
        )
        ax2.set_title(f"Top {top_n} Most Popular HBO Max Movies (Filtered, IMDb Votes in Millions)", fontsize=18)
        ax2.set_xlabel("IMDb Votes (Millions)")
        ax2.set_ylabel("Movie Title")
        fig2.tight_layout()
        st.pyplot(fig2)

        # Decade popularity
        decade_popularity = (
            df_eda.groupby("decade")["imdb_votes_millions"]
            .mean()
            .reset_index()
            .sort_values("imdb_votes_millions")
        )

        st.subheader("Average Movie Popularity by Decade (Filtered, IMDb Votes in Millions)")
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.barplot(
            data=decade_popularity,
            x="decade",
            y="imdb_votes_millions",
            palette="mako",
            ax=ax3
        )
        ax3.set_title("Average Movie Popularity by Decade (Filtered, IMDb Votes in Millions)", fontsize=18)
        ax3.set_xlabel("Decade")
        ax3.set_ylabel("Avg IMDb Votes (Millions)")
        ax3.tick_params(axis="x", rotation=30)
        fig3.tight_layout()
        st.pyplot(fig3)

        # Genre popularity
        genre_cols = [c for c in df_eda.columns if c.startswith("genres_")]
        genre_melt = df_eda[genre_cols + ["imdb_votes"]].melt(
            id_vars=["imdb_votes"],
            var_name="genre",
            value_name="is_genre"
        )
        genre_melt = genre_melt[genre_melt["is_genre"] == 1]

        genre_votes = (
            genre_melt.groupby("genre")["imdb_votes"]
            .mean()
            .div(1_000_000)
            .sort_values()
        )

        st.subheader("Average IMDb Votes by Genre (Filtered, Millions)")
        fig4, ax4 = plt.subplots(figsize=(12, 10))
        sns.barplot(
            x=genre_votes.values,
            y=genre_votes.index,
            palette="rocket",
            ax=ax4
        )
        ax4.set_title("Average IMDb Votes by Genre (Filtered, Millions)", fontsize=20)
        ax4.set_xlabel("Avg IMDb Votes (Millions)")
        ax4.set_ylabel("Genre")
        fig4.tight_layout()
        st.pyplot(fig4)

        # Critics vs audience
        st.subheader("Critics vs Audience Popularity (Filtered)")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(
            data=df_eda,
            x="rotten_score",
            y="imdb_votes_millions",
            alpha=0.4,
            color="#2a9d8f",
            ax=ax5
        )
        sns.regplot(
            data=df_eda,
            x="rotten_score",
            y="imdb_votes_millions",
            scatter=False,
            color="red",
            line_kws={"linewidth": 2},
            ax=ax5
        )
        ax5.set_title("Critics vs Audience Popularity\n(Rotten Tomatoes Score vs IMDb Votes in Millions)", fontsize=18)
        ax5.set_xlabel("Rotten Tomatoes Critic Score")
        ax5.set_ylabel("IMDb Votes (Millions)")
        fig5.tight_layout()
        st.pyplot(fig5)

# ---------- TAB 2: Models & Performance ----------
with tabs[2]:
    st.subheader("Linear Regression: Actual vs Predicted log(IMDb Votes)")
    fig6, ax6 = plt.subplots(figsize=(9, 7))
    sns.scatterplot(x=y_test_lr, y=y_pred_lr, alpha=0.4, ax=ax6)
    min_val = min(y_test_lr.min(), y_pred_lr.min())
    max_val = max(y_test_lr.max(), y_pred_lr.max())
    ax6.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=2)
    ax6.set_title("Linear Regression: Actual vs Predicted log(IMDb Votes)")
    ax6.set_xlabel("Actual log(IMDb Votes)")
    ax6.set_ylabel("Predicted log(IMDb Votes)")
    ax6.grid(True, linestyle="--", alpha=0.4)
    fig6.tight_layout()
    st.pyplot(fig6)

    st.text(
        f"""
=========================================
     LINEAR REGRESSION PERFORMANCE
     (Target: log(IMDb Votes))
=========================================
R² Score:         {r2_lr:.4f}
RMSE (log scale): {rmse_lr:.4f}
MAE  (log scale): {mae_lr:.4f}
"""
    )

    st.subheader("Decision Tree: Top Levels")
    pretty_feature_names = []
    for col in X_encoded.columns:
        if col.startswith("genres_"):
            name = col.replace("genres_", "Genre: ").replace("_", " ")
        elif col.startswith("platforms_"):
            name = col.replace("platforms_", "On: ").replace("_", " ")
        elif col.startswith("rating_"):
            name = col.replace("rating_", "Rating = ").replace("_", " ")
        elif col.startswith("decade_"):
            name = col.replace("decade_", "Decade = ")
        else:
            name = col.replace("_", " ")
        pretty_feature_names.append(name)

    fig7, ax7 = plt.subplots(figsize=(14, 8), dpi=150)
    plot_tree(
        best_dt,
        feature_names=pretty_feature_names,
        filled=True,
        rounded=True,
        max_depth=2,
        fontsize=10,
        proportion=True,
        ax=ax7
    )
    ax7.set_title("Decision Tree (Top Levels) – Predicting log(IMDb Votes)", fontsize=16)
    fig7.tight_layout()
    st.pyplot(fig7)

    st.text(
        f"""
=========================================
     DECISION TREE PERFORMANCE
     (Target: log(IMDb Votes))
=========================================
Best params:      {grid_dt.best_params_}
R² Score:         {r2_dt:.4f}
RMSE (log scale): {rmse_dt:.4f}
MAE  (log scale): {mae_dt:.4f}

On original votes scale:
RMSE:             {rmse_dt_orig:,.0f} votes
MAE:              {mae_dt_orig:,.0f} votes
"""
    )

    st.subheader("Random Forest: Actual vs Predicted IMDb Votes (Log Scales)")
    fig8, ax8 = plt.subplots(figsize=(9, 7))
    sns.scatterplot(x=y_test_rf_orig, y=y_pred_rf_orig, alpha=0.4, ax=ax8)
    mn, mx = min(y_test_rf_orig.min(), y_pred_rf_orig.min()), max(y_test_rf_orig.max(), y_pred_rf_orig.max())
    ax8.plot([mn, mx], [mn, mx], color="red", linewidth=2)
    ax8.set_xscale("log")
    ax8.set_yscale("log")
    ax8.set_title("Random Forest: Actual vs Predicted IMDb Votes", fontsize=16)
    ax8.set_xlabel("Actual IMDb Votes")
    ax8.set_ylabel("Predicted IMDb Votes")
    ax8.grid(True, linestyle="--", alpha=0.4)
    fig8.tight_layout()
    st.pyplot(fig8)

    importances = best_rf.feature_importances_
    feat_importances = pd.Series(importances, index=X_encoded.columns).sort_values(ascending=False).head(15)

    st.subheader("Random Forest – Top 15 Feature Importances")
    fig9, ax9 = plt.subplots(figsize=(10, 7))
    sns.barplot(x=feat_importances.values, y=feat_importances.index, ax=ax9)
    ax9.set_title("Random Forest – Top 15 Feature Importances", fontsize=16)
    ax9.set_xlabel("Importance")
    ax9.set_ylabel("Feature")
    fig9.tight_layout()
    st.pyplot(fig9)

    st.text(
        f"""
=========================================
     RANDOM FOREST PERFORMANCE
=========================================
Best params:      {grid_rf.best_params_}
R² Score:         {r2_rf:.4f}
RMSE (log):       {rmse_rf:.4f}
MAE  (log):       {mae_rf:.4f}

Original scale:
RMSE:             {rmse_rf_orig:,.0f} votes
MAE:              {mae_rf_orig:,.0f} votes
"""
    )

# ---------- TAB 3: Model Comparison ----------
with tabs[3]:
    st.subheader("Final Model Comparison")
    st.dataframe(model_results, use_container_width=True)

    fig10, ax10 = plt.subplots(figsize=(8, 6), dpi=150)
    sns.barplot(
        data=model_results,
        x="Model",
        y="R² (log scale)",
        palette=["#4C72B0", "#55A868", "#C44E52"],
        ax=ax10
    )
    ax10.set_title("Model Performance Comparison (R² on log(IMDb Votes))", fontsize=14)
    ax10.set_ylabel("R² (higher = better)")
    ax10.set_xlabel("")
    ax10.set_ylim(0, 1)
    fig10.tight_layout()
    st.pyplot(fig10)
