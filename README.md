# NPS Challenge — Prédiction NPS client télécom

Prédire la catégorie NPS (Detractor / Passive / Promoter) de clients télécom
à partir du dataset IBM Telco Customer Churn v11.1.3+, pour une équipe
rétention qui priorise les détracteurs.

**Documentation complète** (architecture, chaque phase, résultats, limites,
captures d'écran) : [`DOCUMENTATION.md`](DOCUMENTATION.md).

**Journal de décisions** (38 entrées, format Décision/Alternatives/
Justification/Impact) : [`docs/decisions.md`](docs/decisions.md).
**Write-up final** : [`docs/writeup_fr.docx`](docs/writeup_fr.docx)
(Word, mis en forme, captures d'écran incluses).

## Prérequis

- Python 3.11+ (développé et testé avec 3.13.14).
- ~2 Go d'espace disque libre (dépendances ML incluses).
- Les 6 fichiers IBM Telco Customer Churn v11.1.3+ dans `data/raw/`
  (déjà présents dans ce dépôt).
- Commandes ci-dessous en PowerShell (Windows) ; équivalents bash en
  commentaire quand ils diffèrent.

## Installation

```powershell
cd Challenge_Artefact
python -m venv .venv
.venv\Scripts\activate          # bash/macOS/Linux : source .venv/bin/activate
pip install -r requirements.txt
Copy-Item .env.example .env     # bash/macOS/Linux : cp .env.example .env
```

Aucune clé API n'est nécessaire pour le pipeline principal (raw →
prédiction) : `data/verbatims/verbatims.csv` est déjà fourni. Une clé
n'est requise que pour **régénérer** les verbatims soi-même — voir
[`DOCUMENTATION.md` §3](DOCUMENTATION.md#3-installation-dépendances-et-exécution).

## Exécution du pipeline complet (raw → prédiction)

```powershell
python src/make_dataset.py     # -> data/processed/analytical_dataset.parquet
python src/build_target.py     # -> data/processed/labeled_dataset.parquet
python src/make_splits.py      # -> split_realistic.parquet, split_random.parquet
python -c "import sys; sys.path.insert(0, 'src'); from train import main; main()"
                                # -> models/final_model.joblib (NE PAS lancer train.py directement, voir DOCUMENTATION.md)
streamlit run app/streamlit_app.py
```

La dernière commande ouvre le dashboard (http://localhost:8501, 4 pages
en barre latérale). Détail de chaque étape, de chaque page, des audits
(`fairness_audit.py`), des notebooks, et des commandes optionnelles
(régénération des verbatims, régénération des write-ups Word) :
[`DOCUMENTATION.md` §3](DOCUMENTATION.md#3-installation-dépendances-et-exécution).

## Reproductibilité

Seed globale = 42 partout (`src/config.py:SEED`). Chaque script est
déterministe et peut être ré-exécuté sans changer les artefacts produits
— vérifié dans un environnement vierge (voir
[`DOCUMENTATION.md` §12](DOCUMENTATION.md#12-reproductibilité)).

## Structure du projet

Arborescence complète et commentée : [`DOCUMENTATION.md` §2](DOCUMENTATION.md#2-architecture-du-projet).

```
README.md, DOCUMENTATION.md, requirements.txt, .env.example
data/{raw,processed,verbatims}/
src/*.py            pipeline (dataset -> target -> splits -> features -> train -> evaluate)
notebooks/*.ipynb   EDA, modélisation, interprétation
models/*.joblib     modèle entraîné (non versionné)
app/streamlit_app.py + app/assets/logo.png
docs/{decisions.md, writeup_fr.docx, monitoring_proposal.md, screenshots/}
reports/figures/    SHAP, lift, calibration
```
