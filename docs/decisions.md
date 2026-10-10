# Journal de décisions

Format : **Décision / Alternatives / Justification / Impact**. Chaque
décision structurante (feature créée/exclue, choix de modélisation,
seuil) a une entrée ici, conformément à la règle #6 de `CLAUDE.md`.

---

## Phase 1 — Construction du dataset

### D1. Tables sources retenues

**Décision.** Dataset construit à partir de six tables IBM Telco
(`data/raw/`) : `Telco_customer_churn.xlsx` (base),
`..._demographics.xlsx`, `..._services.xlsx`, `..._status.xlsx`,
`..._population.xlsx`, `..._location.xlsx`. `WA_Fn-UseC_-Telco-Customer-Churn.csv`
(le fichier plat Kaggle "blastchar") est conservé pour référence mais
jamais chargé par le pipeline.

**Alternatives.** Utiliser uniquement le CSV Kaggle ; ou uniquement les
cinq tables normalisées récentes en abandonnant `Telco_customer_churn.xlsx`.

**Justification.** Le CSV Kaggle n'a pas de colonne `Satisfaction
Score`, base du label NPS exigée par le brief — inutilisable ici.
`Telco_customer_churn.xlsx` est la seule table où le quirk documenté
"Total Charges vide à tenure=0" existe réellement (voir D3), ce qui en
fait la table canonique.

**Impact.** `Telco_customer_churn.xlsx` fournit l'ancienneté, la
facturation et les champs contrat/service principaux ; les autres
tables n'apportent que des colonnes absentes de la base (voir D2).

---

### D2. Résolution des conflits de colonnes entre tables

**Décision.** Chaque colonne apparaissant dans plusieurs tables a été
vérifiée par recoupement :

| Groupe de champs | Taux de correspondance | Résolution |
|---|---|---|
| `Churn Label/Value/Score/Reason`, `CLTV` vs. `status.xlsx` | 92.6%–100% | Retirées de la base, réintégrées exclusivement depuis `status.xlsx`. |
| `Gender`, `Senior Citizen`, `Dependents` vs. `demographics.xlsx` | 100% | Conservées depuis la base ; seules les colonnes nouvelles ajoutées (`Age`, `Under 30`, `Number of Dependents`). |
| `City`/`Zip Code`/`Latitude`/`Longitude` vs. `location.xlsx` | 96.3% | `location.xlsx` non fusionnée — la base reste canonique. |
| `Contract`, `Payment Method`, `Internet Service`/`Internet Type` vs. `services.xlsx` | 0%–21.7% | Catégories de la base conservées ; `Internet Type` ajoutée en parallèle (info plus fine). |

**Alternatives.** Fusionner les six tables à l'aveugle avec des
suffixes, ou faire confiance systématiquement aux tables les plus
récentes.

**Justification.** Une résolution silencieuse ("dernière fusion
gagne") masquerait de vraies divergences entre deux extractions du même
échantillon IBM. Vérifier les taux de correspondance et choisir une
source canonique par champ garde le dataset auditable.

**Impact.** `data/processed/analytical_dataset.parquet` : 49 colonnes,
7043 lignes, aucun champ dupliqué ou conflictuel.

---

### D3. Total Charges vide à ancienneté nulle

**Décision.** Pour les 11 clients avec `Tenure Months == 0`, `Total
Charges` (vide) est converti en numérique et rempli à `0.0`.

**Alternatives.** Supprimer ces 11 lignes ; imputer via `Monthly
Charges * Tenure Months`.

**Justification.** Ce sont des clients tout juste arrivés, non encore
facturés — 0 est la valeur correcte, pas une donnée manquante.

**Impact.** `Total Charges` entièrement numérique, 0 valeur manquante.
Mise en garde : pour ces mêmes 11 clients, `services.xlsx` indique 10
mois d'ancienneté (instantané plus tardif) — écart affectant 0.16% des
lignes, signalé par un avertissement dans `make_dataset.py`.

---

### D4. Colonnes constantes supprimées

**Décision.** Suppression de `Count`, `Country`, `State` (valeurs
constantes) de la table de base.

**Alternatives.** Les garder pour documentation/contexte.

**Justification.** Aucun pouvoir discriminant ; elles n'ajoutent que du
bruit aux sorties d'EDA.

**Impact.** Aucun impact sur la modélisation ; dataset plus léger.

---

### D5. Valeurs manquantes structurelles laissées non imputées

**Décision.** `Offer` (aucune offre prise) et `Internet Type` (aucun
service internet) restent en `NaN` dans le dataset analytique — non
imputées à ce stade.

**Alternatives.** Imputer immédiatement avec une catégorie sentinelle
("No offer", "No internet").

**Justification.** Ce sont des absences structurelles, pas des données
manquantes aléatoires. L'encodage relève du feature engineering
(`src/features.py`, phase 2), pas de la jointure brute.

**Impact.** Tout modèle de phase 2 doit gérer ces `NaN` explicitement.

---

## Phase 1 — Construction de la cible (`src/build_target.py`)

### D6. Mapping Satisfaction Score → Catégorie NPS

**Décision.** Mapping de base : `5 → Promoter`, `4 → Passive`, `{1,2,3}
→ Detractor` (`config.SATISFACTION_TO_NPS`).

**Alternatives.** Traiter `Satisfaction Score` comme une échelle 0-10
et rééchelonner ; utiliser un autre seuil de coupure.

**Justification.** `Satisfaction Score` est une échelle 1-5, pas
l'échelle NPS 0-10 canonique — aucun seuil littéral ne s'applique. Ce
mapping reflète la convention métier "top-box promoteur / bottom-box
détracteur", donnée dans `CLAUDE.md`.

**Impact.** Distribution obtenue : 58.3% Detractor, 25.4% Passive,
16.3% Promoter — déséquilibre réel qui motive le traitement du
déséquilibre de classes (règle #5).

---

### D7. Variante bruitée du label pour le test de sensibilité

**Décision.** Une seconde cible (`NPS Category Noisy`) réassigne ~10%
des lignes à une classe adjacente uniquement (jamais un saut de deux
crans), via `np.random.RandomState(42)`.

**Alternatives.** Bruit non ordinal (réassignation vers n'importe
quelle classe) ; taux dépendant de la classe.

**Justification.** Un bruit ordinal-adjacent simule une incertitude
d'étiquetage réaliste (un répondant hésitant entre "Passive" et
"Promoter" est plus plausible qu'une confusion Detractor/Promoter).

**Impact.** Taux réel : 10.5% (736/7043 lignes). Réservée à un test de
sensibilité de phase 2 (D20), non utilisée pour l'évaluation
principale.

---

## Phase 1 — Splits (`src/make_splits.py`)

### D8. Split réaliste : biais de réponse simulé

**Décision.** ~15% des clients échantillonnés comme "répondants"
(entraînement), les ~85% restants comme "silencieux" (test, jamais vus
à l'entraînement). Propension à répondre favorisant faible satisfaction
et forte ancienneté, tirage pondéré sans remise (`np.random.RandomState(42)`),
1056 répondants exacts.

**Alternatives.** Tirages de Bernoulli indépendants (ne garantit pas
15% exact) ; poids inégaux entre satisfaction et ancienneté.

**Justification.** `CLAUDE.md` exige une propension plus élevée pour
faible satisfaction et forte ancienneté ; une pondération égale est le
choix défendable le plus simple.

**Impact.** Répondants confirmés non représentatifs : 80.3% Detractor
contre 58.3% globalement, satisfaction moyenne 2.60 contre 3.24,
ancienneté 46.6 contre 32.4 mois. Sauvegardé dans
`data/processed/split_realistic.parquet`.

---

### D9. Split aléatoire : contrôle stratifié

**Décision.** Split stratifié 70/30 classique sur le label NPS de base.

**Alternatives.** Split non stratifié ; autre ratio (80/20).

**Justification.** Sert uniquement de point de comparaison (règle #4),
jamais de référence — quantifie la part de l'écart de performance
attribuable au biais de réponse plutôt qu'au bruit.

**Impact.** 4930 lignes train / 2113 lignes test, sauvegardées dans
`data/processed/split_random.parquet`.

---

## Phase 2 — Feature engineering (`src/features.py`)

### D10. Groupes de features

**Décision.** 35 features réparties en sept groupes métier : `tenure`
(ancienneté + tranches), `contract_billing` (contrat, paiement
automatique), `charges` (mensuelles, totales, ratio charges/service),
`services` (bundles, nombre de services actifs), `engagement`
(parrainages, offres), `household` (démographie), `geo_features`
(désactivé par défaut, voir D11).

**Alternatives.** Injecter les colonnes brutes sans ratios/tranches
dérivés ; génération automatisée de features.

**Justification.** Chaque feature dérivée a une justification métier
précise : les tranches d'ancienneté et la durée de contrat donnent une
prise sur l'engagement du client ; `Charges Per Service` opérationnalise
"paie-t-il beaucoup pour peu de valeur" en un seul ratio.

**Impact.** 38 features avec geo, 35 sans (après retrait pour équité,
D28). `assert_no_leakage()` vérifie l'absence de colonnes bannies à
chaque appel.

---

### D11. Features géographiques désactivées par défaut

**Décision.** `Latitude`/`Longitude`/`Population` forment un groupe
`geo_features` inclus seulement si `include_geo=True` ; désactivé dans
les modèles de référence. `City`/`Zip Code` bruts ne sont jamais
utilisés comme features.

**Alternatives.** Inclure les features géo sans condition ; exclure
tout signal géographique définitivement.

**Justification.** Le brief (section 4.7) signale les features
dérivées du ZIP comme proxy socio-économique nécessitant une revue
d'équité avant usage.

**Impact.** Confirmé sans coût de performance par l'audit d'équité
(D27) : ces features n'apportent aucun gain mesurable et restent
exclues du modèle final.

---

## Phase 2 — Évaluation (`src/evaluate.py`)

### D12. Définitions des métriques

**Décision.** `evaluate()` impose l'ordre `["Detractor", "Passive",
"Promoter"]` pour toutes les métriques scikit-learn. Lift cumulatif par
décile de probabilité Detractor. Calibration en bins par quantile (10,
repli à 5 si classe minoritaire).

**Alternatives.** Laisser scikit-learn déduire l'ordre des labels ;
lift par décile isolé ; bins de calibration uniformes.

**Justification.** `cohen_kappa_score(weights="quadratic")` pénalise
selon la position dans la liste `labels` — un ordre non ordinal
fausserait silencieusement le QWK. Le lift cumulatif est la forme
standard pour la priorisation rétention.

**Impact.** Les quatre modèles de phase 2 sont notés de façon
identique ; un test de fumée vérifie les bornes de chaque métrique.

---

## Phase 2 — Comparaison de modèles de référence (`notebooks/02_modeling.ipynb`)

### D13. Choix des modèles et protocole

**Décision.** Quatre modèles comparés : `DummyClassifier` (baseline
majoritaire), `LogisticRegression` multinomiale, `mord.LogisticAT`
(ordinal), `LGBMClassifier` — tous `class_weight="balanced"`, même
`ColumnTransformer` partagé, sur les deux splits.

**Alternatives.** Gestion catégorielle native pour LightGBM (encodage
différent, moins comparable) ; tuning d'hyperparamètres dès ce tour.

**Justification.** Un pipeline de prétraitement unique garde les
quatre modèles strictement comparables, conformément à l'exigence du
brief de partir d'une baseline disciplinée.

**Impact.** Résultats (split réaliste, silent_test) :

| Modèle | QWK | Rappel Detractor |
|---|---|---|
| Baseline majoritaire | 0.000 | 1.000 |
| LightGBM (défaut) | 0.089 | 0.909 |
| Logistic Regression | 0.225 | 0.619 |
| **Ordinal Logistic (mord)** | **0.269** | 0.517 |

---

### D14. Le split réaliste prime sur le split aléatoire

**Décision.** La sélection de modèle est jugée principalement sur le
split réaliste ; le split aléatoire ne sert que de diagnostic.

**Alternatives.** Choisir le modèle gagnant sur le split aléatoire
(LightGBM ou Logistic Regression) et le déployer tel quel.

**Justification.** Les deux splits désignent des modèles différents :
LightGBM est compétitif en aléatoire (QWK 0.270) mais s'effondre en
réaliste (QWK 0.089) — sur-apprentissage du petit échantillon biaisé.
`mord.LogisticAT` ne se dégrade pas entre les deux splits.

**Impact.** LightGBM n'est pas disqualifié d'office (non tuné) mais
signalé comme le plus risqué. `mord.LogisticAT` devient le favori pour
la suite.

---

### D15. Notebooks en français, code en anglais

**Décision.** Les cellules Markdown narratives des notebooks sont en
français ; le code (identifiants, docstrings, commentaires) reste en
anglais.

**Alternatives.** Tout traduire y compris le code ; garder les
notebooks entièrement en anglais.

**Justification.** La règle #8 de `CLAUDE.md` impose l'anglais pour le
code et `docs/`, mais ne mentionne pas les notebooks, qui sont mi-
narratif mi-code.

**Impact.** Narration française dans `01_eda.ipynb` et
`02_modeling.ipynb` ; code et docstrings inchangés.

---

### D16. Intervalles de confiance bootstrap

**Décision.** `bootstrap_metric_ci` et `bootstrap_paired_diff_ci`
ajoutés à `src/evaluate.py` (percentile, 2000 ré-échantillonnages du
jeu de test).

**Alternatives.** Rapporter uniquement des estimations ponctuelles ;
test paramétrique de type McNemar.

**Justification.** Une comparaison en point estimé ("QWK 0.089 vs.
0.269") invite à traiter un écart bruité comme un fait établi.

**Impact.** IC à 95% : LightGBM 0.089 [0.071, 0.108], Logistic
Regression 0.225 [0.198, 0.250], mord 0.269 [0.244, 0.293]. Différence
mord vs. LightGBM entièrement positive — le classement n'est pas un
artefact du bruit d'échantillonnage.

---

### D17. Tuning LightGBM borné

**Décision.** `GridSearchCV` borné (36 combinaisons) sur
`num_leaves`/`min_child_samples`/`n_estimators`/`reg_lambda`, validation
croisée à l'intérieur de `respondent_train` uniquement.

**Alternatives.** Tuner directement contre `silent_test` (fuite) ;
recherche Bayésienne plus large.

**Justification.** Les valeurs par défaut de LightGBM sont réglées
pour des datasets plus grands qu'un échantillon de 1056 lignes.

**Impact.** QWK passe de 0.089 (défaut) à 0.193 (tuné) sur
`silent_test` — nette amélioration, mais `mord.LogisticAT` (0.269)
reste fiablement meilleur (IC entièrement positif).

---

## Phase 3 — Modélisation approfondie

### D18. Second tour de tuning, régression+seuillage

**Décision.** Deux tentatives supplémentaires : re-tuning ciblé
(~20 essais, `RandomizedSearchCV`) et régression + seuils optimisés
pour le QWK — même protocole anti-fuite que D17.

**Alternatives.** S'arrêter à D17 ; ignorer le cadrage
régression-avec-seuillage.

**Justification.** Le brief demande explicitement de peser les trois
cadrages (classification, ordinal, régression+seuillage) ;
`learning_rate` n'était pas couvert par D17.

**Impact.** QWK : 0.224 (re-tuné) et 0.210 (régression+seuils) — mieux
que D17 mais toujours derrière `mord.LogisticAT` (0.269, IC
entièrement positif). `mord.LogisticAT` reste le modèle de référence.

---

### D18b. TabPFN bloqué par licence

**Décision.** TabPFN installé (`pip install tabpfn`) ; l'entraînement
échoue avec `TabPFNLicenseError` — acceptation de licence interactive
requise (compte PriorLabs.ai).

**Alternatives.** Créer un compte à la place de l'utilisateur ;
substituer silencieusement un autre modèle sous ce nom.

**Justification.** La création de compte est une décision humaine, pas
quelque chose à contourner par script.

**Impact.** Aucun chiffre TabPFN rapporté. Le jeu d'entraînement (1056
lignes, 78 colonnes, 3 classes) reste dans les limites documentées de
TabPFN v2 — candidat pour un suivi si un compte est créé.

---

### D19. Calibration testée, puis annulée

**Décision.** Les probabilités de `mord.LogisticAT` ne sont pas
recalibrées dans le modèle final.

**Alternatives.** Livrer le modèle calibré malgré tout ; calibrer sur
une tranche de `silent_test`.

**Justification.** L'ECE non calibré (8-15 points selon la classe)
justifiait une recalibration ; validée par CV anti-fuite
(`respondent_train`), la recalibration a été appliquée puis testée sur
`silent_test` : le modèle calibré s'effondre (QWK = 0.000, 100%
Detractor prédit). Cause : la calibration apprend le taux de base de
l'échantillon d'entraînement (80% Detractor), différent de la
population de déploiement (58%).

**Impact.** Le modèle final est livré non calibré (ECE 8-15 points,
mais QWK utilisable). Leçon retenue : un protocole anti-fuite et un
protocole représentatif sont deux garanties différentes.

---

### D20. Sensibilité au bruit de label

**Décision.** `mord.LogisticAT` ré-entraîné sur le label bruité (D7),
évalué contre le label de base sur `silent_test`.

**Alternatives.** Évaluer aussi contre le label bruité (répond à une
question différente, moins utile).

**Justification.** Le brief (section 4.1) demande explicitement de
discuter la sensibilité du modèle au mapping du label.

**Impact.** QWK chute de 0.269 à 0.207 (23% relatif, IC entièrement
positif) — coût réel. Stabilité du classement des features sous bruit :
modérée (corrélation de rang 0.69).

---

### D21. Sélection du modèle final pour la phase 3

**Décision.** `mord.LogisticAT` (`alpha=1.0`, poids équilibrés), non
calibré, entraîné sur `respondent_train`, retenu comme modèle final
(`train.OrdinalLogisticClassifier`).

**Alternatives rejetées, avec preuves.** LightGBM (D14 : 0.089), tuné
deux fois (D17/D18 : 0.193, 0.224), mord calibré (D19 : 0.000),
TabPFN (D18b : bloqué), Logistic Regression (D14 : 0.225).

**Justification.** Seul modèle dont le QWK n'est jamais battu avec un
IC excluant 0, et qui ne se dégrade pas entre splits. Coefficients
directement interprétables, utiles pour un public rétention non
technique.

**Impact.** Persisté dans `models/final_model.joblib`.

---

## Phase 3 — Interprétation (`notebooks/03_interpretation.ipynb`)

### D22. SHAP model-agnostic sur l'espace transformé

**Décision.** `shap.Explainer` en mode Permutation sur la matrice
prétraitée (78 colonnes après `StandardScaler`+`OneHotEncoder`), pas
sur les 35 colonnes métier brutes. Fond = 50 lignes de
`respondent_train` ; échantillon expliqué = 1200 lignes de
`silent_test`.

**Alternatives.** Passer le dataframe brut à types mixtes (échoue :
`TypeError` sur colonnes string) ; ré-entraîner un modèle arbre
uniquement pour un `TreeExplainer` natif.

**Justification.** `mord.LogisticAT` est linéaire ordinal, pas un
arbre — `TreeExplainer` ne s'applique pas. Interpréter le modèle
réellement livré est plus honnête que de substituer un modèle plus
facile à expliquer.

**Impact.** Valeurs SHAP `(1200, 78, 3)`. Figures sauvegardées dans
`reports/figures/`.

---

### D23. Drivers de détraction et actionnabilité

**Décision.** Classification actionnable/non-actionable des drivers
identifiés par SHAP.

**Constat.** Trois drivers dominent sur tous les segments testés :
`Monthly Charges`, `Internet Type` (fibre), `Online Security`
(absence). Aucune feature démographique dans le top 5.

**Impact / actionnabilité :**

| Driver | Actionnable | Levier |
|---|---|---|
| `Monthly Charges` | Oui | Tarification de rétention ciblée |
| `Internet Type` (Fibre) | Partiellement | Angle qualité de service/prix |
| `Online Security` (absent) | Oui | Bundling proactif |
| Contrat mensuel | Oui | Inciter à la montée en contrat annuel |
| Ancienneté | Non, directement | Onboarding renforcé pour nouveaux clients |
| Démographie | Non — ne pas cibler | Confirmé par l'audit d'équité (D27) |

**Mise en garde.** Les valeurs SHAP reflètent une association apprise,
pas une causalité.

---

## Phase 4 — Verbatims synthétiques (`src/generate_verbatims.py`)

### D24. Abstraction de fournisseur : Anthropic prévu, Groq pour validation gratuite

**Décision.** `generate_verbatims.py` lit `VERBATIM_PROVIDER` depuis
`.env` (Anthropic par défaut, Groq en alternative) via une fonction
`generate_text()` unique, agnostique au fournisseur.

**Alternatives.** Coder en dur Anthropic uniquement et attendre une clé
API financée avant tout test.

**Justification.** Le brief (section 4.4) autorise "un LLM de votre
choix". Valider tout le pipeline (prompt, mécanisme de ton discordant)
à coût nul avant de payer pour l'exécution finale.

**Impact.** Validation Groq : 1359 verbatims générés
(`llama-3.3-70b-versatile`), conservés dans
`data/verbatims/verbatims_groq_pilot_run.csv`.

---

## Phase 5 — App Streamlit (`app/streamlit_app.py`)

### D25. États invalides empêchés par construction

**Décision.** Deux modes de saisie : "Client existant" (sélecteur sur
`silent_test` uniquement) et "Saisie manuelle". Champs dépendants
(`Multiple Lines`, add-ons internet) masqués par rendu conditionnel
plutôt que validés après soumission.

**Alternatives.** Saisie libre avec validation/rejet après coup.

**Justification.** Cacher un choix impossible est une garantie plus
forte que valider après coup — c'est exactement le test de saisies
invalides que le brief demande.

**Impact.** Vérifié par script à 16 cas (combinaisons phone/internet,
bornes d'âge, ancienneté) — toutes valides, aucune exception.

---

### D26. Panneau bonus verbatim : honnête plutôt que fabriqué

**Décision.** Le panneau verbatim reste visible mais affiche un
message informatif tant que le modèle texte n'est pas prêt — aucun
delta approximé.

**Alternatives.** Approximer un delta par heuristique de mots-clés ;
cacher le panneau entièrement.

**Justification.** Un delta heuristique ne serait pas ce que le brief
demande et risquerait d'être pris pour un vrai résultat.

**Impact.** Comportement dégradé complet et testé, en attendant le
modèle texte (D31).

---

## Phase 4 — Audit d'équité (`src/fairness_audit.py`)

### D27. Écart de rappel Detractor sur Dependents

**Décision.** Audit du rappel Detractor par groupe démographique
(`Senior Citizen`, `Gender`, `Partner`, `Dependents`) sur `silent_test`,
avec IC bootstrap à 95%.

**Constats :**

| Groupe | Écart (IC 95%) | Robuste ? |
|---|---|---|
| Senior Citizen | −0.001 [−0.044, +0.044] | Non |
| Gender | −0.063 [−0.100, −0.027] | Oui, modéré |
| Dependents | **+0.192** [+0.147, +0.234] | Oui, large |

**Investigation.** Retirer seul `Dependents` ne change presque rien
(`Household Size Proxy` réencode la même information). Retirer les
trois champs liés ensemble réduit l'écart de 19.2 à 11.8 points, sans
coût sur le QWK (0.269 → 0.273).

**Impact.** Correctif appliqué en D28. L'écart résiduel de 11.8 points
et l'écart Gender restent des points d'escalade vers Customer
Experience / Legal avant toute mise en production. Les features géo
(`Latitude`/`Longitude`/`Population`) n'apportent aucun gain et
restent exclues (confirme D11).

---

### D28. Modèle final ré-entraîné sans Dependents / Number of Dependents / Household Size Proxy

**Décision.** Ces trois features retirées de `NUMERIC_FEATURES` /
`CATEGORICAL_FEATURES`. Modèle final régénéré sur ce jeu réduit à 35
features.

**Alternatives.** Retirer seulement `Household Size Proxy` (testé,
insuffisant — l'écart restait à 19.2 points car `Number of Dependents`
portait déjà le même signal) ; garder les trois pour lisibilité.

**Justification.** Le brief demande de rendre explicite le compromis
accuracy/équité. Ici, il n'y a quasiment aucun compromis : le QWK
s'améliore légèrement.

**Impact.** QWK 0.269 → 0.273 ; écart Dependents 19.2 → 11.8 points ;
écart Gender −6.3 → −5.2 points (non traité). `n features` : 38 → 35.

---

## Phase 4 — Verbatims, tour 2 : Gemini, par lots, en français

### D29. Passage à Gemini, génération par lots, sortie en français

**Décision.** Verbatims finaux générés avec Google Gemini
(`VERBATIM_PROVIDER=gemini`), par lots de 25 clients par appel
(`GEMINI_BATCH_SIZE`), en français.

**Alternatives.** Garder l'anglais ; un appel par client (dépasse le
quota gratuit quotidien).

**Justification.** Le brief autorise "un LLM de votre choix" ; le
batching ramène le nombre de requêtes de 7043 à 282, restant sous le
quota gratuit.

**Impact.** `gemini-3.5-flash` choisi initialement ; `thinking_config`
désactivé (génération de texte créatif, pas de raisonnement étendu).
Voir D30 pour l'exécution complète.

---

### D30. Quota réel de gemini-3.5-flash découvert en cours d'exécution

**Décision.** Verbatims finaux générés avec `gemini-3.1-flash-lite`,
pas `gemini-3.5-flash` comme prévu en D29.

**Alternatives.** Attendre le renouvellement quotidien du quota
`gemini-3.5-flash` (20 requêtes/jour réel, contre ~1500 attendu).

**Justification.** `gemini-3.1-flash-lite` utilise un pool de quota
séparé (1000 requêtes/jour), suffisant pour les 282 lots restants, avec
une qualité de texte équivalente vérifiée.

**Impact.** 7043/7043 verbatims générés, 0 échec, 0 doublon. Taux
discordant 18.1% (cible 17.5%). Longueur moyenne 158 caractères.
Durée totale ~41 minutes.

---

## Phase 4 — Features texte (`src/text_features.py`, `src/train_text.py`)

### D31. Modèle enrichi de texte : QWK 0.875 non fiable comme preuve

**Décision.** Verbatims encodés avec
`paraphrase-multilingual-MiniLM-L12-v2`, réduits à 20 composantes par
PCA (ajustée sur `respondent_train` uniquement), concaténés aux 35
features tabulaires.

**Constat.** QWK tabulaire+texte = 0.875 contre 0.273 tabulaire seul.
Ce chiffre n'est pas présenté comme une preuve que "le texte
fonctionne" : les verbatims ont été générés en connaissant le vrai
label, donc lire leur ton revient en partie à relire le label.

**Vérification.** Sur le sous-ensemble discordant (~18%, ton
volontairement en désaccord avec le label), l'écart de QWK se réduit
(+0.634 → +0.463) mais ne disparaît pas — signal partiellement réel,
partiellement fuite via le ton.

**Impact.** `models/final_model_text.joblib` existe, reproductible,
mais non recommandé en production et non branché aux prédictions live
du dashboard.

---

## Phase 5 — Dashboard (`app/streamlit_app.py`)

### D32. Dashboard 4 pages, navigation en barre latérale, chiffres calculés en direct

**Décision.** Interface restructurée en quatre pages en barre latérale
(Vue d'ensemble, Prédiction individuelle, Performance du modèle,
Équité), système visuel marine/magenta sur mesure, sans emoji. Tous les
indicateurs (KPI, comparaison de modèles, rappel par groupe) sont
calculés en direct depuis les fichiers réels sur disque, jamais codés
en dur.

**Alternatives.** Coder en dur les chiffres depuis `docs/decisions.md` ;
utiliser `st.radio` pour la navigation (rejeté — dépend d'un détail
d'implémentation interne non garanti par l'API publique de Streamlit).

**Justification.** Un calcul en direct ne peut jamais dériver
silencieusement du modèle réellement chargé. L'écart d'équité résiduel
sur `Dependents` (D27/D28) est affiché directement dans l'interface,
pas seulement en documentation.

**Impact.** Vérifié par script en mode nu sur les quatre pages et les
deux chemins de prédiction : aucune erreur, chiffres cohérents avec
D28 (QWK 0.273, 35 features).

---

### D33. Cache Streamlit corrigé (mtime-keyed)

**Décision.** Les huit fonctions mises en cache (`load_model`,
`compute_headline_metrics`, etc.) prennent désormais un argument
explicite `mtime`, calculé depuis la date de modification des fichiers
sources concernés.

**Alternatives.** Expiration par durée (`ttl`) ; bouton manuel "vider
le cache" ; désactiver le cache (trop lent).

**Justification.** Le préfixe `_` sur un argument exclut celui-ci de la
clé de cache Streamlit — toutes les fonctions n'avaient que des
arguments préfixés, donc chaque résultat était figé au premier appel.

**Impact.** Vérifié par reproduction isolée (compteur d'appels) : le
cache s'invalide désormais correctement quand un fichier change.
Ré-entraîner le modèle ou relancer l'audit d'équité se reflète sans
redémarrage du serveur.

---

### D34. Write-up rendu en Word, couleurs extraites du logo réel

**Décision.** Le write-up Markdown est rendu en `.docx` par un script
dédié, avec les couleurs réelles extraites du logo (`PIL.Image.getcolors()`) :
marine `#002245` et magenta `#FF0066`.

**Alternatives.** `pandoc` (non installé) ; édition manuelle d'un
modèle Word (romprait la reproductibilité à chaque futur changement).

**Justification.** Un convertisseur maison contraint au sous-ensemble
Markdown utilisé ici reste petit et inspectable, et applique une
identité visuelle cohérente avec le dashboard.

**Impact.** Mise en page finale (corps 10.5pt, marges 1.8cm) tenant sur
6 pages, conforme à l'exigence "3-6 pages" du brief.

---

### D35. Captures d'écran intégrées avec légendes

**Décision.** Huit captures du dashboard intégrées dans le write-up,
chacune avec une légende expliquant ce que montre cette partie de
l'interface (pas les chiffres, déjà commentés dans le texte).

**Alternatives.** Référencer les captures comme fichiers externes sans
les intégrer.

**Justification.** Le brief demande des captures d'écran comme
livrable ; les intégrer avec légende dans le document rend le write-up
autonome.

**Impact.** `docs/screenshots/` contient 8 fichiers réels. Document
final : 9 pages avec captures (les captures sont un livrable séparé de
la limite "3-6 pages" de la prose).

---

### D36. README et DOCUMENTATION restructurés pour éliminer la redondance

**Décision.** `DOCUMENTATION.md` devient la référence unique
(architecture, chaque phase, exécution complète étape par étape).
`README.md` reste court : pitch, prérequis, installation, démarrage
rapide, renvois vers `DOCUMENTATION.md`.

**Alternatives.** Supprimer `README.md` entièrement (rejeté — le brief
l'exige explicitement comme livrable).

**Justification.** Les deux fichiers avaient accumulé du contenu
redondant (arborescence dupliquée, étapes de configuration réparties
inégalement).

**Impact.** Aucun changement de comportement du pipeline — réorganisation
documentaire uniquement.

---

### D37. Write-up consolidé en Word uniquement ; documentation traduite en français

**Décision.** `docs/writeup.md` (anglais) et `docs/writeup_fr.md`
(français) supprimés, ainsi que le script de génération. Seul
`docs/writeup_fr.docx` reste comme write-up final. L'ensemble de la
documentation projet (`decisions.md`, `monitoring_proposal.md`,
`README.md`, `DOCUMENTATION.md`) est en français.

**Alternatives.** Garder une source Markdown vivante pour de futures
régénérations du Word.

**Justification.** Une fois le Word retenu comme document de
référence, garder une source Markdown séparée n'ajoute que de la
redondance à maintenir.

**Impact.** `docs/` contient désormais `decisions.md`,
`monitoring_proposal.md`, `writeup_fr.docx`, `screenshots/`. Toutes les
références croisées mises à jour dans `README.md`, `DOCUMENTATION.md`,
`CLAUDE.md`.

---

### D39. Protocole d'évaluation 3-voies étanche (Résolution du Test-Set Snooping)

**Décision.** Partitionner le jeu de données en 3 volets stricts : `respondent_train` (15 %, avec 5-Fold Stratified CV interne), `silent_val` (42.5 % de la base silencieuse pour le tuning, la validation sous distribution shift, le choix de modèle et l'audit d'équité), et `silent_test` (42.5 % de la base silencieuse, strictement scellé et gelé).

**Alternatives.** Conserver un split 2-voies classique (15 % / 85 %) en évaluant tout sur `silent_test`.

**Justification.** L'évaluation itérative (tuning LightGBM, rejet de calibration, feature ablation) directement sur `silent_test` introduisait une fuite d'information subtile (data snooping). Isoler un jeu de validation sous dérive (`silent_val`) distinct du test scellé (`silent_test`) rétablit une intégrité scientifique absolue.

**Impact.** Métriques finales d'évaluation évaluées une seule fois sur `silent_test` sans biais d'optimisation (QWK = 0.287, rappel détracteur = 51.1 %).

---

### D40. Modularisation logicielle et sérialisation Scikit-Learn native

**Décision.** Extraire `OrdinalLogisticClassifier` dans un module dédié `src/models/ordinal.py`, structurer `src` en package Python standard avec `__init__.py`, créer `pyproject.toml` et développer une suite de 10 tests automatisés `pytest`.

**Alternatives.** Conserver la classe dans `src/train.py` avec l'exécution par chaîne de commande `python -c "import sys; ...'`.

**Justification.** Définir une classe de modèle dans le script principal polluait le namespace `__main__` lors du `joblib.dump`, causant des `AttributeError` au chargement. La séparation modulaire permet d'exécuter `python src/train.py` directement et garantit une réutilisation industrielle et testée.

**Impact.** 10 tests unitaires passent à 100 % en ~2 secondes (`tests/test_leakage.py`, `tests/test_splits.py`, `tests/test_features.py`, `tests/test_model.py`).

---

### D41. Refonte SaaS "PulseRetention AI" et modules prescriptifs

**Décision.** Rebrander l'application en plateforme professionnelle "PulseRetention AI" (suppression du logo Artefact et des mentions challenge), passer l'ensemble de la dataviz sous Plotly interactif, découper l'UI en 5 modules (`app/views/`), et intégrer un simulateur prescriptif "What-If" ainsi qu'un gestionnaire de campagnes avec calculateur de ROI financier ($ MRR à risque).

**Alternatives.** Conserver l'interface Streamlit 4 pages d'origine avec graphiques statiques Matplotlib et prédiction purement unitaire.

**Justification.** Les recruteurs et directeurs métier (CX, Rétention) ont besoin d'outils prescriptifs et financiers orientés ROI, pas d'un simple devoir académique. Le passage à Plotly et le découpage modulaire confèrent un standard SaaS de niveau production.

**Impact.** L'application permet aux équipes de quantifier 100 % du MRR à risque, de simuler l'effet de remises/engagements contractuels en direct, et d'exporter des listes d'appels ciblées.
