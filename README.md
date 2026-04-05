# House Price Prediction — Data Engineering & ML avec Snowflake

## Contexte

Ce projet a été réalisé dans le cadre d'un workshop d'évaluation en Data Engineering et Machine Learning. L'objectif était de construire un pipeline complet de prédiction de prix immobiliers en utilisant Snowflake comme plateforme unifiée, sans exporter les données vers un environnement externe.

L'ensemble du workflow — ingestion, exploration, préparation, entraînement, optimisation et déploiement — a été réalisé directement dans Snowflake en combinant Snowpark, Snowflake ML et les Snowflake Notebooks.

## Dataset

Le dataset contient des informations sur les caractéristiques de 1090 maisons ainsi que leurs prix de vente. Il est disponible dans le bucket S3 suivant : `s3://logbrain-datalake/datasets/house_price/`.

| Variable | Description |
|---|---|
| price | Prix de vente de la maison |
| area | Surface totale en mètres carrés |
| bedrooms | Nombre de chambres |
| bathrooms | Nombre de salles de bain |
| stories | Nombre d'étages |
| mainroad | Accès à une route principale (yes/no) |
| guestroom | Présence d'une chambre d'amis (yes/no) |
| basement | Présence d'un sous-sol (yes/no) |
| hotwaterheating | Système de chauffage à eau chaude (yes/no) |
| airconditioning | Présence de la climatisation (yes/no) |
| parking | Nombre de places de stationnement |
| prefarea | Localisation en zone privilégiée (yes/no) |
| furnishingstatus | État d'ameublement (furnished / semi-furnished / unfurnished) |

## Pipeline réalisé

**1. Ingestion des données**
Les données sont chargées depuis S3 via un stage Snowflake. Le fichier étant au format JSON, les données brutes sont stockées dans une table VARIANT puis aplaties via LATERAL FLATTEN pour obtenir un tableau structuré avec les bons types de colonnes.

**2. Exploration des données**
L'exploration couvre les statistiques descriptives, la détection des valeurs manquantes, l'analyse de la distribution des variables numériques et catégorielles, la détection des outliers par boxplot et IQR, la matrice de corrélation entre variables numériques, et l'analyse des associations entre variables catégorielles via le V de Cramer.

**3. Sélection des variables**
Un test ANOVA a été appliqué sur les variables catégorielles pour mesurer leur impact sur le prix. La variable HOTWATERHEATING a été écartée car sa modalité "yes" ne représente que 4.77% des observations, ce qui la rend peu discriminante pour la prédiction du prix.

**4. Préparation des données**
La variable cible PRICE a été transformée en log pour corriger son asymétrie. Les variables binaires ont été encodées avec un Label Encoding et FURNISHINGSTATUS avec un One-Hot Encoding. Le dataset a ensuite été divisé en 70% pour l'entraînement et 30% pour le test, puis normalisé avec un StandardScaler.

**5. Entraînement des modèles**
Trois modèles ont été entraînés en partant du plus simple vers le plus complexe : Régression Linéaire, Random Forest et XGBoost.

**6. Optimisation**
Le XGBoost a été optimisé via RandomizedSearchCV avec early stopping pour identifier les meilleurs hyperparamètres.

**7. Stockage et inférence**
Le meilleur modèle a été enregistré dans le Snowflake Model Registry avec ses métriques et métadonnées. L'inférence est réalisée directement dans Snowflake via SQL.

**8. Application Streamlit**
Une application Streamlit a été développée dans Snowflake pour permettre aux utilisateurs métier d'estimer le prix d'une maison en saisissant ses caractéristiques.

## Analyse des performances

### Résultats comparatifs

| Modèle | R² | MAE | RMSE |
|---|---|---|---|
| Régression Linéaire | 0.68 | 38 692 € | 51 137 € |
| Random Forest | 0.85 | 19 817 € | 32 210 € |
| XGBoost | 0.84 | 15 852 € | 31 880 € |
| XGBoost Optimisé | 0.86 | 14 402 € | 28 474 € |

### Interprétation

La régression linéaire établit notre baseline avec un R² de 0.68 et une erreur moyenne de 38 692 €. Ce résultat attendu s'explique par la nature non linéaire des données immobilières : l'impact de chaque caractéristique sur le prix ne s'additionne pas de façon uniforme.

Le passage aux modèles basés sur les arbres apporte une amélioration immédiate et significative. Le Random Forest réduit l'erreur de moitié, et le XGBoost confirme sa réputation de modèle le plus efficace sur les données tabulaires en atteignant la meilleure MAE avant optimisation.

Le XGBoost optimisé est notre meilleur modèle. Avec un R² de 0.86 et une MAE de 14 402 €, il explique 86% de la variance du prix et réduit l'erreur moyenne de 63% par rapport à la régression linéaire. Sur un prix moyen de 237 663 €, cela représente une erreur relative d'environ 6%, ce qui est un niveau de précision satisfaisant pour un usage métier réel.

Les 14% de variance non expliqués correspondent vraisemblablement à des critères subjectifs difficiles à quantifier, comme l'état de finition intérieure ou l'attractivité du bien.

### Paramètres du meilleur modèle

| Paramètre | Valeur |
|---|---|
| n_estimators | 500 |
| learning_rate | 0.1 |
| max_depth | 6 |
| early_stopping_rounds | 20 |
| Arbres utilisés (early stopping) | 300 |

## Structure du dépôt
.
├── notebook_house_price.md       # Notebook Snowflake complet (pipeline ML)
├── streamlit_house_price.py      # Application Streamlit de prédiction
└── README.md                     # Ce fichier

## Réaliser par: Missael Jean Léger SYLVAIN
