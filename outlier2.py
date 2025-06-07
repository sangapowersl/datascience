import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Config de la page
st.set_page_config(page_title="Rapport Iris", layout="wide")

# Chargement du dataset Iris
df = sns.load_dataset("iris")

# Titre
st.title("📊 Rapport interactif Iris : Données + Explication")

# Deux colonnes : gauche (résultat) / droite (explication)
col1, col2 = st.columns([2, 1])

# --- 1. Aperçu du dataset ---
with col1:
    st.subheader("🔍 Aperçu du jeu de données")
    st.dataframe(df.head())

with col2:
    st.subheader("📝 Interprétation")
    st.markdown("""
    Le dataset **Iris** contient 150 échantillons répartis en 3 espèces : *setosa*, *versicolor* et *virginica*.
    Chaque échantillon est décrit par 4 mesures : longueur/largeur des sépales et pétales.
    Ci-contre, les 5 premières lignes illustrent la structure du jeu de données.
    """)

# --- 2. Statistiques descriptives ---
with col1:
    st.subheader("📈 Statistiques descriptives")
    stats = df.describe()
    st.dataframe(stats)

with col2:
    st.subheader("📝 Analyse")
    st.markdown(f"""
    Les statistiques descriptives montrent les valeurs minimales, maximales, moyennes et les quartiles pour chaque variable numérique.
    Par exemple, **petal length** a une moyenne de **{stats.loc['mean','petal_length']:.2f}** cm.
    Cela permet de comprendre la répartition et l’échelle des valeurs.
    """)

# --- 3. Filtrage dynamique ---
with col1:
    st.subheader("🔎 Filtrage par espèce")
    species = st.multiselect("Choisir les espèces à afficher", df["species"].unique(), default=df["species"].unique())
    filtered_df = df[df["species"].isin(species)]
    st.dataframe(filtered_df)

with col2:
    st.subheader("📝 Utilité")
    st.markdown("""
    Le filtrage permet de restreindre l’analyse à une ou plusieurs espèces sélectionnées.
    Cela facilite la comparaison ciblée entre les types de fleurs.
    """)

# --- 4. Graphiques dynamiques ---
numeric_cols = filtered_df.select_dtypes(include="number").columns
selected_col = st.selectbox("📊 Choisissez une variable à analyser", numeric_cols)

tab1, tab2 = st.tabs(["Histogramme + KDE", "Boxplot"])

with tab1:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📊 Histogramme")
        fig1, ax1 = plt.subplots()
        sns.histplot(filtered_df[selected_col], kde=True, ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.subheader("📝 Interprétation")
        st.markdown(f"""
        L’histogramme montre la distribution de **{selected_col}**, accompagnée de la courbe de densité (KDE).
        Cela permet de voir la forme générale (asymétrie, pics) et d’évaluer la normalité.
        """)

with tab2:
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📦 Boxplot")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=filtered_df[selected_col], ax=ax2)
        st.pyplot(fig2)
    with col2:
        st.subheader("📝 Interprétation")
        st.markdown(f"""
        Le boxplot met en évidence les quartiles et les valeurs aberrantes (outliers) de **{selected_col}**.
        C’est un outil efficace pour détecter la dispersion et les extrêmes.
        """)

# --- 5. Corrélation ---
with st.container():
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("🔗 Matrice de corrélation")
        fig_corr, ax_corr = plt.subplots()
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax_corr)
        st.pyplot(fig_corr)
    with col2:
        st.subheader("📝 Observation")
        st.markdown("""
        La matrice montre les corrélations linéaires entre variables.
        Une forte corrélation est observée entre **petal length** et **petal width**.
        Ces informations sont utiles pour la sélection de variables dans des modèles prédictifs.
        """)

# --- 6. Détection d'outliers ---
q1 = filtered_df[selected_col].quantile(0.25)
q3 = filtered_df[selected_col].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
outliers = filtered_df[(filtered_df[selected_col] < lower_bound) | (filtered_df[selected_col] > upper_bound)]

col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🚨 Outliers (méthode IQR)")
    st.write(f"**Bornes IQR** : {lower_bound:.2f} à {upper_bound:.2f}")
    st.write(f"**Nombre d’outliers** : {outliers.shape[0]}")
    st.dataframe(outliers)

with col2:
    st.subheader("📝 Explication")
    st.markdown(f"""
    Les outliers sont calculés selon la méthode IQR (écart interquartile).
    Toutes les valeurs en dehors de [{lower_bound:.2f}, {upper_bound:.2f}] sont considérées comme anormales.
    Cela permet d'identifier des mesures extrêmes de **{selected_col}**.
    """)
