<div align="center">

# ⚡ TelcoPulse AI — Plateforme d'Intelligence Client & Rétention NPS
### *Customer NPS Prediction, Prescriptive Retention & Revenue Risk Intelligence for Telecom Operators*

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33%2B-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/pytest-10%2F10%20passed-55A868?style=flat-square&logo=pytest&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)
![Status](https://img.shields.io/badge/status-production--ready-10B981?style=flat-square)

Une solution complète de Data Science & MLOps pour prédire la catégorie NPS (**Détracteur / Passif / Promoteur**), estimer le chiffre d'affaires mensuel à risque ($ MRR), et simuler l'impact prescriptif des actions de fidélisation sur la majorité silencieuse (85 % du parc).

</div>

---

## 1. Contexte Métier & Problématique

Dans les télécoms, **seuls ~15 % des clients répondent aux enquêtes de satisfaction** (*TM Forum 2021*). De plus, les répondants volontaires souffrent d'un fort **biais de négativité** (*Anderson 1998*) : les clients insatisfaits ou très anciens sont sur-représentés dans les retours d'enquête.

L'équipe Rétention se retrouve aveugle face aux **85 % restants (la base silencieuse)** :
* Qui sont les détracteurs silencieux prêts à résilier ?
* Quel est le volume de revenu mensuel exposé au churn ($ MRR) ?
* Quelles actions opérationnelles (durée de contrat, remise, options de sécurité) réduisent efficacement le risque ?

**TelcoPulse AI** résout ce défi grâce à une modélisation ordinale robuste, un protocole d'évaluation étanche sous dérive de distribution, et un tableau de bord d'aide à la décision prescriptif.

---

## 2. Fondements Scientifiques & Hypothèses Validées

Chaque choix de modélisation est validé empiriquement et étayé par la littérature du secteur :

1. **Mapping Échelle de Likert (1-5) → NPS Ordinal :**
   * *Méthodologie :* 1-3 = **Détracteur** (58.3 %), 4 = **Passif** (25.4 %), 5 = **Promoteur** (16.3 %).
   * *Références :* Aligné sur les travaux de conversion d'échelles de satisfaction en Net Promoter Score (*Rockwell 2008*, *Survey Research Methods*) et les standards de fidélisation de Fred Reichheld (*Bain & Company / Harvard Business Review 2003*).

2. **Biais de Réponse Asymétrique (Scénario de Production Réaliste) :**
   * Modélisation d'une propension de réponse volontaire pondérée par l'insatisfaction (-z) et l'ancienneté (+z).
   * *Références :* Démontré par *Anderson (1998)* ("Customer Satisfaction and Word of Mouth", *Journal of Service Research*) et documenté dans les études de NPS transactionnel télécom (*Kannan et al. 2022*).

3. **Tolérance au Bruit de Label (10 %) :**
   * Test de sensibilité ordinal par réassignation adjacente de 10 % des labels pour évaluer la stabilité des drivers.
   * *Références :* Aligné sur les taux de variance test-retest des enquêtes de satisfaction client (*Keiningham et al. 2007*).

---

## 3. Architecture Logicielle & Rigueur Méthodologique

### Protocole 3-Voies Étanche (Zéro "Test Set Snooping")
Pour garantir l'intégrité scientifique des métriques et éviter toute contamination du jeu de test (fuite par réutilisation itérative) :
* **`respondent_train` (15 % = 1 056 clients) :** Entraînement avec validation croisée stratifiée pré-calculée à 5 plis (**5-Fold Stratified CV**).
* **`silent_val` (42.5 % = 2 993 clients) :** Partition de validation dédiée à l'évaluation sous dérive de distribution (sélection de modèles, tuning LightGBM, rejet de la recalibration, audit d'équité).
* **`silent_test` (42.5 % = 2 994 clients) :** **Jeu de test scellé et intact**, évalué strictement une seule fois à la fin pour les métriques de référence.

### Modélisation Ordinale
* **Modèle Retenu :** `OrdinalLogisticClassifier` (`mord.LogisticAT`, $\alpha=1.0$, pondération équilibrée des classes).
* **Avantage Clé :** Contrairement aux modèles d'arbres (LightGBM) qui s'effondrent sous le biais de réponse en prédisant 91 % de détracteurs, le modèle ordinal conserve sa structure et atteint **QWK = 0.287** et **51.1 % de rappel détracteur** sur la population silencieuse scellée.
* **Sérialisation Saine :** Encapsulation Scikit-Learn sans pollution du namespace `__main__`, garantissant un rechargement fiable en production via `joblib`.

---

## 4. Fonctionnalités de la Plateforme

L'application (`app/streamlit_app.py`) est organisée en 5 modules professionnels :

| Module | Rôle Opérationnel & Valeur Business |
|---|---|
| **📊 Vue d'ensemble** | KPIs exécutifs, détection des 5 987 silencieux, estimation du MRR à risque, donut NPS interactif Plotly. |
| **🔍 Scoring & Diagnostic** | Fiche client individuelle, probabilités de classe avec jauges interactives, top 5 drivers locaux SHAP, recommandations d'action. |
| **🛠️ Simulateur Prescriptif ("What-If")** | Test en direct de leviers de rétention (passage en contrat annuel, remise de 10 $, add-on sécurité/support) et visualisation de la baisse de probabilité de détraction. |
| **🎯 Campagnes & ROI ($ MRR)** | Outil de segmentation multi-critères, calcul en direct du ROI financier ($ MRR & ARR sauvés vs coût de contact), export CSV télévente. |
| **⚖️ Équité & Monitoring MLOps** | Audit d'équité par groupe protégé (parité de rappel détracteur), suivi de dérive PSI (Population Stability Index), règles de ré-entraînement. |

---

## 5. Installation & Démarrage Rapide

### Prérequis
* Python 3.11+ (testé sous Python 3.11, 3.12 et 3.13)
* Git

### Installation Locale
```powershell
# 1. Cloner le dépôt
git clone https://github.com/grimael/NPS_Project.git
cd NPS_Project

# 2. Créer et activer l'environnement virtuel
python -m venv .venv
.venv\Scripts\activate      # Sur Linux/macOS : source .venv/bin/activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

### Exécution du Pipeline de Données & Modélisation
```powershell
# Préparation des données, cible ordinale, split 3-voies et entraînement
python src/make_dataset.py
python src/build_target.py
python src/make_splits.py
python src/train.py
```

### Lancer les Tests Automatisés
```powershell
python -m pytest tests/ -v
```
*(10 tests unitaires validant l'absence de fuite, la cohérence des splits, le feature engineering et l'inférence du modèle).*

### Lancer l'Application Dashboard
```powershell
streamlit run app/streamlit_app.py
```
L'interface s'ouvre automatiquement sur `http://localhost:8501`.

---

## 6. Déploiement Conteneurisé (Docker)

Le projet intègre un `Dockerfile` optimisé pour un déploiement cloud immédiat (AWS ECS, GCP Cloud Run, Render, Hugging Face Spaces) :

```powershell
# Construction de l'image
docker build -t telcopulse-ai .

# Lancement du conteneur
docker run -p 8501:8501 telcopulse-ai
```

---

## 7. Structure du Dépôt

```
├── app/
│   ├── streamlit_app.py     # Point d'entrée & routeur de l'application
│   ├── styles.py            # Design system, CSS moderne & tokens
│   ├── utils.py             # Services de chargement, inférence & explicabilité
│   └── views/
│       ├── overview.py      # Module Vue d'ensemble & KPIs
│       ├── prediction.py    # Module Diagnostic individuel & SHAP
│       ├── simulator.py     # Module Simulateur prescriptif What-If
│       ├── campaigns.py     # Module Gestionnaire de campagnes & ROI ($ MRR)
│       └── monitoring.py    # Module Audit d'équité & MLOps (PSI)
├── data/
│   ├── raw/                 # 6 tables sources IBM Telco Customer Churn
│   ├── processed/           # Dataset analytique, labels & splits Parquet
│   └── verbatims/           # Verbatims synthétiques FR (étude NLP)
├── src/
│   ├── __init__.py          # Package TelcoPulse
│   ├── config.py            # Chemins, seed=42, blacklist anti-fuite
│   ├── make_dataset.py      # Traitement brut -> analytique
│   ├── build_target.py      # Construction de la cible ordinale
│   ├── make_splits.py       # Split réaliste 3-voies & CV stratifiée
│   ├── features.py          # Feature engineering (35 variables, 7 groupes)
│   ├── evaluate.py          # Métriques QWK, Macro-F1, Lift, Calibration
│   ├── train.py             # Pipeline d'entraînement et validation
│   └── models/
│       ├── __init__.py
│       └── ordinal.py       # OrdinalLogisticClassifier (mord scikit-learn wrapper)
├── tests/
│   ├── conftest.py          # Fixtures Pytest
│   ├── test_leakage.py      # Tests d'absence de fuite
│   ├── test_splits.py       # Tests d'étanchéité du protocole 3-voies
│   ├── test_features.py     # Tests du feature engineering
│   └── test_model.py        # Tests d'inférence et probabilités
├── Dockerfile               # Conteneurisation de production
├── pyproject.toml           # Configuration de packaging standard
├── requirements.txt         # Dépendances versionnées
└── DOCUMENTATION.md         # Documentation technique et méthodologique détaillée
```

---

<div align="center">

*Conçu et développé dans le respect des meilleures pratiques en Machine Learning, Software Engineering & MLOps.*

</div>
