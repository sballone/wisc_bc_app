# Breast Cancer Diagnostic - Machine Learning

Application Streamlit d'analyse et de classification des tumeurs mammaires a partir du jeu de donnees **Wisconsin Breast Cancer**.

## Description

Cette application permet de :

- **Explorer** le jeu de donnees (apercu, statistiques descriptives, valeurs manquantes)
- **Visualiser** les distributions, correlations, comparaisons Malin vs Benin et une analyse PCA
- **Entrainer et comparer** 5 modeles de Machine Learning (Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting)
- **Predire** un diagnostic de maniere interactive en ajustant les parametres cellulaires

## Jeu de donnees

Le fichier `wisc_bc_data.csv` contient 569 observations et 30 variables numeriques extraites d'images de biopsies par aspiration a l'aiguille fine (FNA). La variable cible `diagnosis` indique si la tumeur est **Maligne (M)** ou **Benigne (B)**.

## Installation locale

```bash
pip install -r requirements.txt
streamlit run wisc_app.py
```

## Deploiement

L'application est deployee sur **Streamlit Community Cloud**.

## Technologies

- Python
- Streamlit
- Scikit-learn
- Plotly
- Pandas / NumPy
