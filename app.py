import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Chargement des données
st.title("Rapport interactif - Ventes")
df = pd.read_csv("ventes.csv")

# Aperçu
st.subheader("Aperçu des données")
st.write(df.head())

# Statistiques
st.subheader("Statistiques descriptives")
st.write(df.describe())

# Graphique dynamique
st.subheader("Graphique des ventes")
colonne = st.selectbox("Choisir une colonne numérique", df.select_dtypes(include="number").columns)

fig, ax = plt.subplots()
sns.histplot(df[colonne], kde=True, ax=ax)
st.pyplot(fig)