import streamlit as st
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration initiale
st.set_page_config(page_title="Rapport Iris", layout="wide")

# Chargement des données
df = sns.load_dataset("iris")

st.title("📊 Rapport interactif - Dataset Iris")

# 1. Aperçu des données
st.subheader("🔍 Aperçu du jeu de données")
st.write(df.head())

# 2. Statistiques descriptives
st.subheader("📈 Statistiques descriptives")
st.write(df.describe())

# 3. Filtrage par espèce
st.subheader("🔎 Filtrage par espèce")
species = st.multiselect("Choisir une ou plusieurs espèces", df["species"].unique(), default=df["species"].unique())
filtered_df = df[df["species"].isin(species)]
st.write(filtered_df)

# 4. Graphique dynamique : histogramme + boxplot
st.subheader("📊 Visualisation interactive")
numeric_cols = filtered_df.select_dtypes(include="number").columns
col = st.selectbox("Choisir une colonne numérique à visualiser", numeric_cols)

tab1, tab2 = st.tabs(["Histogramme + KDE", "Boxplot"])

with tab1:
    fig1, ax1 = plt.subplots()
    sns.histplot(filtered_df[col], kde=True, ax=ax1)
    st.pyplot(fig1)

with tab2:
    fig2, ax2 = plt.subplots()
    sns.boxplot(x=filtered_df[col], ax=ax2)
    st.pyplot(fig2)

# 5. Corrélation
st.subheader("📌 Matrice de corrélation")
fig_corr, ax_corr = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax_corr)
st.pyplot(fig_corr)

# 6. Détection des outliers avec IQR
st.subheader("🚨 Détection des outliers (méthode IQR)")

q1 = filtered_df[col].quantile(0.25)
q3 = filtered_df[col].quantile(0.75)
iqr = q3 - q1

borne_inf = q1 - 1.5 * iqr
borne_sup = q3 + 1.5 * iqr

outliers = filtered_df[(filtered_df[col] < borne_inf) | (filtered_df[col] > borne_sup)]

st.write(f"**Bornes IQR :** {borne_inf:.2f} à {borne_sup:.2f}")
st.write(f"**Nombre d'outliers détectés pour `{col}` :** {outliers.shape[0]}")
st.dataframe(outliers)
