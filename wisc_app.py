import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, auc)
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Breast Cancer Diagnostic - ML", layout="wide", initial_sidebar_state="expanded")

def fig_layout(fig, h=400):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e2e8f0"), margin=dict(t=30, b=30))
    fig.update_xaxes(gridcolor="rgba(100,180,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(100,180,255,0.08)")
    return fig

@st.cache_data
def load_data():
    df = pd.read_csv(BASE_DIR / "wisc_bc_data.csv").drop(columns=["id"], errors="ignore").dropna(axis=1, how="all")
    df["diagnosis_encoded"] = LabelEncoder().fit_transform(df["diagnosis"])
    return df

df = load_data()
features = [c for c in df.columns if c not in ["diagnosis", "diagnosis_encoded"]]
X, y = df[features], df["diagnosis_encoded"]

MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=10000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
}

with st.sidebar:
    st.image(str(BASE_DIR / "OIP.webp"), use_container_width=True)
    st.markdown("---")
    page = st.radio("Navigation", ["Accueil", "Exploration des donnees", "Visualisations",
                                    "Modelisation ML", "Prediction interactive"])
    st.markdown("---")
    model_choice = st.selectbox("Algorithme", list(MODELS.keys()))
    test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20, 5)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size / 100, random_state=42, stratify=y)

# ======================= ACCUEIL =======================
if page == "Accueil":
    st.image(str(BASE_DIR / "eye-catching-d-illustration-depicts-cancer-cell-against-vibrant-blue-backdrop-its-glowing-connections-symbolize-316316209.webp"),
             use_container_width=True)
    st.title("Breast Cancer Diagnostic")
    st.caption("Probleme ML : Classification binaire de la variable 'diagnosis' (M = Malin, B = Benin)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Echantillons", df.shape[0])
    c2.metric("Variables", len(features))
    c3.metric("Malins (M)", (df["diagnosis"] == "M").sum())
    c4.metric("Benins (B)", (df["diagnosis"] == "B").sum())

    st.markdown("---")
    col_l, col_r = st.columns([2, 1])
    with col_l:
        st.subheader("Contexte du projet")
        st.info(
            "**Probleme a resoudre :** predire la variable cible `diagnosis` (M = Malin, B = Benin) "
            "a partir de 30 variables numeriques extraites d'images de biopsies FNA.\n\n"
            "Il s'agit d'un probleme de **classification binaire supervisee**. "
            "Les features decrivent les noyaux cellulaires : "
            "rayon, texture, perimetre, aire, lissite, compacite, concavite, symetrie et dimension fractale "
            "(chacune avec mean, se et worst)."
        )
    with col_r:
        st.image(str(BASE_DIR / "R.jpg"), caption="Biopsie par aspiration a l'aiguille fine (FNA)")

    st.subheader("Distribution de la variable cible : diagnosis")
    diag_counts = df["diagnosis"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=["Benin (B)", "Malin (M)"], values=[diag_counts.get("B", 0), diag_counts.get("M", 0)],
        hole=0.55, marker=dict(colors=["#34d399", "#f87171"]), textinfo="percent+label"))
    st.plotly_chart(fig_layout(fig_pie, 350), use_container_width=True)

# ======================= EXPLORATION =======================
elif page == "Exploration des donnees":
    st.title("Exploration des donnees")
    tab1, tab2, tab3 = st.tabs(["Apercu", "Statistiques", "Valeurs manquantes"])

    with tab1:
        st.subheader("Apercu du jeu de donnees")
        st.dataframe(df.head(20), use_container_width=True, height=500)
        st.subheader("Types de donnees")
        st.dataframe(pd.DataFrame({"Variable": df.columns, "Type": df.dtypes.astype(str).values,
                                    "Non-null": df.notnull().sum().values}), use_container_width=True)

    with tab2:
        st.subheader("Statistiques descriptives")
        st.dataframe(df[features].describe().T, use_container_width=True, height=600)
        st.subheader("Statistiques par diagnostic")
        stat = st.selectbox("Statistique", ["mean", "median", "std", "min", "max"])
        grouped = df.groupby("diagnosis")[features].agg(stat).T
        grouped.columns = ["Benin (B)", "Malin (M)"]
        st.dataframe(grouped, use_container_width=True, height=600)

    with tab3:
        st.subheader("Valeurs manquantes")
        missing = df.isnull().sum()
        if missing.sum() == 0:
            st.success(f"Aucune valeur manquante dans le jeu de donnees ({len(df)} observations completes).")
        else:
            st.dataframe(pd.DataFrame({"Variable": missing.index, "Manquantes": missing.values,
                                        "% ": (missing / len(df) * 100).values})[missing.values > 0])

# ======================= VISUALISATIONS =======================
elif page == "Visualisations":
    st.title("Visualisations")
    tab1, tab2, tab3, tab4 = st.tabs(["Distributions", "Correlations", "Comparaison M vs B", "PCA"])

    with tab1:
        st.subheader("Distribution des variables")
        sel = st.multiselect("Variables", features, default=features[:6])
        if sel:
            n_cols = 3
            fig = make_subplots(rows=(len(sel) + 2) // 3, cols=3, subplot_titles=sel)
            for i, feat in enumerate(sel):
                for diag, color, name in [("B", "#34d399", "Benin"), ("M", "#f87171", "Malin")]:
                    fig.add_trace(go.Histogram(x=df[df["diagnosis"] == diag][feat], name=name,
                                               marker_color=color, opacity=0.7, showlegend=(i == 0)),
                                  row=i // 3 + 1, col=i % 3 + 1)
            fig.update_layout(barmode="overlay")
            st.plotly_chart(fig_layout(fig, 300 * ((len(sel) + 2) // 3)), use_container_width=True)

    with tab2:
        st.subheader("Matrice de correlation")
        grp = st.radio("Groupe", ["Mean", "SE", "Worst", "Toutes"], horizontal=True)
        filt = {"Mean": "_mean", "SE": "_se", "Worst": "_worst"}.get(grp, "")
        cols = [c for c in features if filt in c] if filt else features
        corr = df[cols].corr()
        fig_c = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index,
                                      colorscale="RdBu_r", zmid=0, text=np.round(corr.values, 2),
                                      texttemplate="%{text}", textfont=dict(size=9)))
        st.plotly_chart(fig_layout(fig_c, 600 if filt else 900), use_container_width=True)

        st.subheader("Top correlations avec la cible (diagnosis)")
        corr_target = df[features + ["diagnosis_encoded"]].corr()["diagnosis_encoded"].drop("diagnosis_encoded")
        top = corr_target.abs().sort_values(ascending=False).head(15)
        fig_b = go.Figure(go.Bar(x=[corr_target[f] for f in top.index], y=list(top.index), orientation="h",
                                  marker_color=["#f87171" if corr_target[f] > 0 else "#60a5fa" for f in top.index]))
        fig_b.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Correlation (1=Malin)")
        st.plotly_chart(fig_layout(fig_b, 500), use_container_width=True)

    with tab3:
        st.subheader("Comparaison Malin vs Benin")
        comp = st.multiselect("Variables a comparer", features,
                               default=["radius_mean", "texture_mean", "perimeter_mean",
                                         "area_mean", "smoothness_mean", "compactness_mean"])
        chart = st.radio("Type", ["Box Plot", "Violin Plot"], horizontal=True)
        for feat in comp:
            fig_v = go.Figure()
            for diag, color, name in [("B", "#34d399", "Benin"), ("M", "#f87171", "Malin")]:
                if chart == "Box Plot":
                    fig_v.add_trace(go.Box(y=df[df["diagnosis"] == diag][feat], name=name,
                                           marker_color=color, boxmean="sd"))
                else:
                    fig_v.add_trace(go.Violin(y=df[df["diagnosis"] == diag][feat], name=name,
                                              marker_color=color, box_visible=True, meanline_visible=True))
            fig_v.update_layout(title=feat)
            st.plotly_chart(fig_layout(fig_v, 350), use_container_width=True)

    with tab4:
        st.subheader("Analyse en Composantes Principales (PCA)")
        pca2 = PCA(n_components=2).fit_transform(X_scaled)
        pca_df = pd.DataFrame({"PC1": pca2[:, 0], "PC2": pca2[:, 1],
                                "Diagnostic": df["diagnosis"].map({"M": "Malin", "B": "Benin"})})
        fig_p = px.scatter(pca_df, x="PC1", y="PC2", color="Diagnostic",
                           color_discrete_map={"Malin": "#f87171", "Benin": "#34d399"}, opacity=0.7)
        pca_full = PCA().fit(X_scaled)
        fig_p.update_layout(
            xaxis_title=f"PC1 ({pca_full.explained_variance_ratio_[0]*100:.1f}%)",
            yaxis_title=f"PC2 ({pca_full.explained_variance_ratio_[1]*100:.1f}%)")
        st.plotly_chart(fig_layout(fig_p, 500), use_container_width=True)
        st.info(f"Les 2 premieres composantes capturent **{sum(pca_full.explained_variance_ratio_[:2])*100:.1f}%** de la variance totale.")

        cumvar = np.cumsum(pca_full.explained_variance_ratio_) * 100
        fig_ev = go.Figure()
        fig_ev.add_trace(go.Bar(x=list(range(1, len(cumvar)+1)), y=pca_full.explained_variance_ratio_*100,
                                 name="Individuelle", marker_color="#3b82f6", opacity=0.7))
        fig_ev.add_trace(go.Scatter(x=list(range(1, len(cumvar)+1)), y=cumvar, name="Cumulee",
                                     line=dict(color="#f59e0b", width=3), mode="lines+markers"))
        fig_ev.add_hline(y=95, line_dash="dash", line_color="#f87171",
                         annotation_text="95% variance", annotation_font_color="#f87171")
        fig_ev.update_layout(xaxis_title="Composante", yaxis_title="Variance expliquee (%)")
        st.plotly_chart(fig_layout(fig_ev, 400), use_container_width=True)

# ======================= MODELISATION =======================
elif page == "Modelisation ML":
    st.title("Modelisation ML - Prediction de diagnosis")

    model = MODELS[model_choice]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    st.subheader(f"Resultats : {model_choice}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Precision", f"{prec:.2%}")
    c3.metric("Recall", f"{rec:.2%}")
    c4.metric("F1-Score", f"{f1:.2%}")

    st.markdown("---")
    col_cm, col_roc = st.columns(2)

    with col_cm:
        st.subheader("Matrice de confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(go.Heatmap(
            z=cm, x=["Benin (pred)", "Malin (pred)"], y=["Benin (reel)", "Malin (reel)"],
            colorscale=[[0, "#0f172a"], [1, "#3b82f6"]], text=cm, texttemplate="%{text}",
            textfont=dict(size=20, color="white"), showscale=False))
        fig_cm.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_layout(fig_cm, 400), use_container_width=True)

    with col_roc:
        st.subheader("Courbe ROC")
        if y_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"ROC (AUC={roc_auc:.3f})",
                                          line=dict(color="#3b82f6", width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Aleatoire",
                                          line=dict(color="#64748b", dash="dash")))
            fig_roc.update_layout(xaxis_title="Faux positifs", yaxis_title="Vrais positifs")
            st.plotly_chart(fig_layout(fig_roc, 400), use_container_width=True)

    if hasattr(model, "feature_importances_"):
        st.subheader("Importance des variables")
        imp = pd.DataFrame({"Variable": features, "Importance": model.feature_importances_}
                           ).sort_values("Importance", ascending=True).tail(15)
        fig_imp = go.Figure(go.Bar(x=imp["Importance"], y=imp["Variable"], orientation="h",
                                    marker_color="#3b82f6"))
        st.plotly_chart(fig_layout(fig_imp, 500), use_container_width=True)

    st.markdown("---")
    st.subheader("Comparaison de tous les modeles")
    results = []
    for name, mdl in MODELS.items():
        mdl.fit(X_train, y_train)
        pred = mdl.predict(X_test)
        results.append({"Modele": name, "Accuracy": accuracy_score(y_test, pred),
                         "Precision": precision_score(y_test, pred), "Recall": recall_score(y_test, pred),
                         "F1-Score": f1_score(y_test, pred)})
    results_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False)

    fig_comp = go.Figure()
    for metric, color in {"Accuracy": "#3b82f6", "Precision": "#8b5cf6",
                           "Recall": "#f59e0b", "F1-Score": "#34d399"}.items():
        fig_comp.add_trace(go.Bar(x=results_df["Modele"], y=results_df[metric], name=metric, marker_color=color))
    fig_comp.update_layout(barmode="group", yaxis=dict(range=[0.8, 1.02]))
    st.plotly_chart(fig_layout(fig_comp, 450), use_container_width=True)
    st.dataframe(results_df, use_container_width=True)

    st.subheader("Rapport de classification")
    st.code(classification_report(y_test, y_pred, target_names=["Benin (B)", "Malin (M)"]))

# ======================= PREDICTION =======================
elif page == "Prediction interactive":
    st.title("Prediction de diagnosis")
    st.image(str(BASE_DIR / "R.jpg"), width=300)
    st.info("Ajustez les curseurs pour simuler les mesures d'une biopsie FNA. "
            "Le modele predira la valeur de **diagnosis** : Benin (B) ou Malin (M).")

    model_p = MODELS[model_choice]
    model_p.fit(X_train, y_train)

    mean_feats = [f for f in features if "_mean" in f]
    input_vals = {}
    cols_in = st.columns(3)
    for i, feat in enumerate(mean_feats):
        with cols_in[i % 3]:
            input_vals[feat] = st.slider(feat.replace("_mean", "").replace("_", " ").title(),
                                          float(X[feat].min()), float(X[feat].max()),
                                          float(X[feat].mean()), key=feat)
    for feat in features:
        if feat not in input_vals:
            input_vals[feat] = float(X[feat].mean())

    if st.button("Lancer la prediction", type="primary", use_container_width=True):
        inp_scaled = scaler.transform(pd.DataFrame([input_vals])[features])
        pred = model_p.predict(inp_scaled)[0]
        proba = model_p.predict_proba(inp_scaled)[0] if hasattr(model_p, "predict_proba") else None

        st.markdown("---")
        if pred == 1:
            st.error("### MALIN (M)\nLe modele predit une tumeur maligne.")
        else:
            st.success("### BENIN (B)\nLe modele predit une tumeur benigne.")

        if proba is not None:
            c1, c2 = st.columns(2)
            c1.metric("Probabilite Benin", f"{proba[0]:.1%}")
            c2.metric("Probabilite Malin", f"{proba[1]:.1%}")

            fig_pr = go.Figure(go.Bar(x=["Benin (B)", "Malin (M)"], y=[proba[0], proba[1]],
                                       marker_color=["#34d399", "#f87171"],
                                       text=[f"{proba[0]:.1%}", f"{proba[1]:.1%}"], textposition="outside"))
            fig_pr.update_layout(yaxis=dict(range=[0, 1.15]))
            st.plotly_chart(fig_layout(fig_pr, 350), use_container_width=True)

        st.warning("Cette prediction est a titre demonstratif uniquement. "
                    "Elle ne remplace en aucun cas un diagnostic medical professionnel.")
