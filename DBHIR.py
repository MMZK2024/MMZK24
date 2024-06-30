import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Configuration du journal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ajouter du CSS personnalisé pour centrer le texte et l'image
st.markdown("""
<style>
body {
    background-color: #f5f5f5;
}
h1, h2 {
    color: #333;
    font-family: 'Arial', sans-serif;
    text-align: center;
}
.container {
    display: flex;
    flex-direction: column;
    align-items: center;
}
.footer {
    position: fixed;
    bottom: 0;
    width: 100%;
    text-align: center;
    padding: 10px 0;
    background-color: #f5f5f5;
    font-family: 'Arial', sans-serif;
    color: #666;
}
</style>
""", unsafe_allow_html=True)

# Titre de l'application et image locale
st.markdown('<div class="container">', unsafe_allow_html=True)
image_path = "logo.jpg"  # Chemin vers l'image locale
st.image(image_path, caption='MIAB 2023-2024', use_column_width=True)
st.markdown('<h2>Hello everybody <br> ___Here is our streamlit project___</h2>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Disposition de l'application Streamlit
st.title('DBSCAN and Hierarchical Clustering')

# Ajouter des instructions
st.write("""
## Instructions
1. Upload your CSV file.
2. Adjust settings for DBSCAN and Hierarchical Clustering.
3. Click the corresponding button to apply clustering.
""")

# Fonction pour charger et prétraiter les données
def charger_donnees(fichier_telecharge):
    if fichier_telecharge is not None:
        try:
            data = pd.read_csv(fichier_telecharge)
            return data
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données : {e}")
            st.error("Erreur lors du chargement des données")
    return None

# Fonction pour prétraiter les données
def pretraiter_donnees(data):
    try:
        data = pd.get_dummies(data, drop_first=True)
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        return data, data_scaled
    except Exception as e:
        logger.error(f"Erreur lors du prétraitement des données : {e}")
        st.error("Erreur lors du prétraitement des données")
        return None, None

# Fonction pour appliquer DBSCAN
def appliquer_dbscan(data, eps, min_samples):
    try:
        data_pretraiter, data_scaled = pretraiter_donnees(data)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(data_scaled)
        data_pretraiter['Cluster'] = dbscan_labels
        return data_pretraiter, dbscan_labels, data_scaled
    except Exception as e:
        logger.error(f"Erreur lors de l'application de DBSCAN : {e}")
        st.error("Erreur lors de l'application de DBSCAN")
        return None, None, None

# Fonction pour appliquer le Clustering Hiérarchique
def appliquer_agglomerative(data, n_clusters):
    try:
        data_pretraiter, data_scaled = pretraiter_donnees(data)
        agg = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
        agg_labels = agg.fit_predict(data_scaled)
        data_pretraiter['Cluster'] = agg_labels
        return data_pretraiter, agg_labels, data_scaled
    except Exception as e:
        logger.error(f"Erreur lors de l'application du Clustering Hiérarchique : {e}")
        st.error("Erreur lors de l'application du Clustering Hiérarchique")
        return None, None, None

# Charger le dataset
fichier_telecharge = st.file_uploader("Choose a CSV file", type="csv")

if fichier_telecharge is not None:
    data = charger_donnees(fichier_telecharge)
    st.write("Data overview:")
    st.write(data.head())

    # Paramètres pour DBSCAN
    st.sidebar.header('DBSCAN settings')
    eps = st.sidebar.slider('Value of epsilon', 0.1, 10.0, 0.5)
    min_samples = st.sidebar.slider('Minimum number of samples', 1, 20, 5)
    
    # Paramètres pour Clustering Hiérarchique
    st.sidebar.header('Hierarchical Clustering Settings')
    n_clusters = st.sidebar.slider('Number of Clusters', 2, 10, 3)
    
    if st.sidebar.button('Apply DBSCAN'):
        data_clustered, cluster_labels, data_scaled = appliquer_dbscan(data, eps, min_samples)
        if data_clustered is not None:
            st.write("Results of DBSCAN Clustering:")
            st.write(data_clustered[['Cluster']].value_counts())

            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data_clustered['Cluster'], palette='viridis')
            plt.title('Clusters DBSCAN')
            st.pyplot(plt)

    if st.sidebar.button('Apply Hierarchical Clustering'):
        data_clustered, cluster_labels, data_scaled = appliquer_agglomerative(data, n_clusters)
        if data_clustered is not None:
            st.write("Results of Hierarchical Clustering:")
            st.write(data_clustered[['Cluster']].value_counts())

            plt.figure(figsize=(10, 7))
            sns.scatterplot(x=data_scaled[:, 0], y=data_scaled[:, 1], hue=data_clustered['Cluster'], palette='viridis')
            plt.title('Hierarchical Clusters')
            st.pyplot(plt)

# Ajouter le pied de page
st.markdown('<div class="footer">Developed by [Mohamed idrissi, Mohamed benriala, Zahira ellaouah, Khadija boudalaa]</div>', unsafe_allow_html=True)