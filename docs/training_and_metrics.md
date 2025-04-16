
<div style="margin-top: 120px; text-align: center;">

# Stratégie d'Entraînement et d'Évaluation pour un Modèle GNN de Localisation de Fuites sur Réseau Hydraulique

<div style="margin: 0 auto; max-width: 650px; text-align: justify; margin-top: 50px;">

**Introduction**

Ce document détaille la stratégie d'entraînement et d'évaluation proposée pour le modèle de Réseau de Neurones sur Graphes (GNN) conçu pour la localisation de fuites sur le réseau de distribution d'eau Net3. Une fois l'architecture du modèle définie (voir document séparé ou section précédente), il est crucial d'établir une méthodologie robuste pour entraîner ses paramètres et évaluer sa performance de manière fiable. Les choix présentés ici visent à optimiser l'apprentissage à partir des données simulées, en tenant compte des défis spécifiques tels que le fort déséquilibre des classes (une seule fuite parmi des centaines de tuyaux), et à définir des métriques pertinentes pour juger de l'efficacité réelle du modèle dans sa tâche de localisation.

</div>
</div>

<div style="page-break-after: always;"></div>

## 1. Stratégie d'Entraînement

Cette section décrit les composantes clés du processus d'entraînement.

### 1.1. Fonction de Perte (Loss Function)

*   **Objectif :** Sélectionner une fonction qui mesure l'erreur entre les probabilités de fuite prédites et la vérité terrain, tout en gérant le déséquilibre extrême des classes.
*   **Analyse et Justification :** La tâche est une classification binaire sur chaque arête. Une Binary Cross-Entropy (BCE) standard serait submergée par la classe négative majoritaire ("pas de fuite"). Deux approches principales sont envisagées :
    1.  **BCE Pondérée (`Weighted BCE`) :** Attribue un poids plus élevé (`pos_weight`) aux erreurs sur la classe minoritaire (fuite), forçant le modèle à y prêter plus attention. C'est une technique standard pour l'imbalance [2].
    2.  **Focal Loss (`FL`) :** Modifie la BCE en ajoutant un facteur de modulation `(1 - pt)^gamma` [9]. Ce facteur réduit la contribution des exemples faciles (bien classifiés, `pt` proche de 1) à la perte totale, permettant au modèle de se concentrer *dynamiquement* sur les exemples difficiles (mal classifiés, `pt` faible), comme la détection de la fuite elle-même. Son efficacité pour cette tâche avec des GNN a été démontrée [2].
*   **Décision :** Utiliser la **Focal Loss** comme fonction de perte principale en raison de son adaptation dynamique à la difficulté des exemples.
    *   *Formule (simplifiée) :* `FL(pt) = -(1 - pt)^gamma * log(pt)`
    *   *Hyperparamètre :* `gamma` (ex: 2) sera ajusté pendant l'expérimentation.
    *   La BCE Pondérée sera considérée comme une alternative pour comparaison.

### 1.2. Optimiseur

*   **Objectif :** Sélectionner un algorithme pour mettre à jour les poids du GNN afin de minimiser la Focal Loss.
*   **Analyse et Justification :** Les optimiseurs adaptatifs sont préférés. AdamW [10] améliore Adam en découplant la régularisation par Weight Decay (L2), ce qui conduit souvent à une meilleure généralisation.
*   **Décision :** Utiliser **AdamW** comme optimiseur.

### 1.3. Hyperparamètres Clés de l'Optimiseur

*   **Objectif :** Définir les paramètres contrôlant AdamW.
*   **Analyse et Justification :**
    *   **Taux d'Apprentissage (Learning Rate, LR) :** Crucial. Commencer avec une valeur standard (ex: `1e-3`) et utiliser un **Scheduler de Taux d'Apprentissage** (ex: `ReduceLROnPlateau`) pour ajuster dynamiquement le LR pendant l'entraînement.
    *   **Weight Decay (Régularisation L2) :** Paramètre de régularisation important à régler (ex: `1e-5` à `1e-2`).
    *   **Betas (`β1`, `β2`) et Epsilon (`ε`) :** Les valeurs par défaut (`0.9`, `0.999`, `1e-8`) sont généralement robustes.
*   **Décision :** Le **LR** (avec scheduler) et le **Weight Decay** seront les principaux hyperparamètres à optimiser. Les autres resteront aux valeurs par défaut.

### 1.4. Gestion des Batches (Lots)

*   **Objectif :** Définir comment les données sont présentées au modèle.
*   **Analyse et Justification :**
    *   **Taille de Batch (`batch_size`) :** Choisir la plus grande taille possible compatible avec la mémoire GPU pour un gradient plus stable et un meilleur parallélisme.
    *   **Mélange (`shuffle`) :** **Mélanger** les données d'entraînement à chaque époque pour améliorer la généralisation. Ne pas mélanger les jeux de validation/test.
*   **Décision :** Utiliser des **batches** de taille maximale possible et **mélanger** le jeu d'entraînement.

### 1.5. Mécanismes Avancés d'Entraînement

*   **Objectif :** Améliorer la stabilité et l'efficacité.
*   **Analyse et Justification :**
    *   **Early Stopping :** Arrêter l'entraînement si une métrique de validation (ex: AUC-PR) ne s'améliore pas pendant une certaine "patience", pour éviter le sur-apprentissage et économiser du temps de calcul. Conserver le modèle avec la meilleure performance de validation.
    *   **Gradient Clipping :** Optionnel, utile pour prévenir l'explosion des gradients en limitant leur norme maximale.
*   **Décision :** Implémenter l'**Early Stopping** basé sur une métrique de validation clé. Envisager le **Gradient Clipping** si besoin.

<div style="page-break-after: always;"></div>

## 2. Stratégie d'Évaluation

Cette section détaille comment la performance du modèle sera mesurée et suivie.

### 2.1. Métriques d'Évaluation

*   **Objectif :** Sélectionner des métriques évaluant fidèlement la capacité de *localisation* de la fuite, en tenant compte du déséquilibre des classes.
*   **Analyse et Justification :** L'Accuracy est inadaptée. Nous nous concentrons sur :
    *   **Métriques de Classification (Classe "Fuite") :**
        *   **AUC-PR (Area Under the Precision-Recall Curve) :** **Métrique clé** pour données déséquilibrées [2], évalue le compromis Précision/Rappel.
        *   **Rappel (Recall / Sensibilité) :** Fraction des fuites réelles détectées (`TP / (TP + FN)`).
        *   **Précision :** Fraction des alarmes correspondant à de vraies fuites (`TP / (TP + FP)`).
        *   **Score-F1 :** Moyenne harmonique de Précision et Rappel.
    *   **Métriques de Localisation/Ranking :**
        *   **MRR (Mean Reciprocal Rank) :** Position moyenne inverse de la vraie fuite dans la liste classée. Idéal pour la localisation (proche de 1 si la fuite est souvent classée première).
        *   **Hits@k :** Pourcentage de fois où la vraie fuite est dans les `k` premiers candidats (ex: `k=1, 3, 5`).
    *   **Performance sur cas "Pas de Fuite" :**
        *   **Taux de Faux Positifs (FPR) :** Fréquence des fausses alarmes en l'absence de fuite.
*   **Décision :** Évaluation principale avec **AUC-PR, Rappel, Précision, F1-Score (pour la classe Fuite), MRR, et Hits@k (k=1, 3, 5)**. Le FPR sera aussi suivi.

### 2.2. Suivi de l'Entraînement

*   **Objectif :** Visualiser et comprendre le processus d'apprentissage.
*   **Analyse et Justification :** Suivre l'évolution des performances sur les jeux d'entraînement et de validation est essentiel.
*   **Décision :** Utiliser **TensorBoard ou Weights & Biases (WandB)** pour logger et visualiser la **perte (train/val)** et les **métriques de validation clés (AUC-PR, MRR, Rappel)** au fil des époques. La métrique de validation principale (ex: AUC-PR) servira de critère pour l'**Early Stopping**.

### 2.3. Analyse des Erreurs

*   **Objectif :** Comprendre les types d'erreurs commises par le modèle.
*   **Analyse et Justification :** Aller au-delà des métriques globales est nécessaire.
*   **Décision :** Effectuer une **analyse qualitative et quantitative des erreurs** (Fuites Manquées - FN ; Fausses Alarmes - FP) sur le jeu de test après l'entraînement. Analyser les caractéristiques des scénarios/tuyaux où le modèle échoue, examiner la distribution des scores prédits, et potentiellement utiliser la **visualisation** des prédictions sur le graphe pour des cas spécifiques.

### 2.4. Comparaison à des Baselines

*   **Objectif :** Démontrer la valeur ajoutée du GNN.
*   **Analyse et Justification :** Comparer à des méthodes plus simples.
*   **Décision :** Comparer les métriques clés du GNN à celles obtenues par au moins **une baseline non-ML** (ex: basée sur des seuils de pression) et **une baseline ML simple** (ex: classifieur tabulaire sur features agrégées sans topologie).

<div style="page-break-after: always;"></div>

## 3. Conclusion

La stratégie d'entraînement et d'évaluation proposée vise à maximiser les chances de développer un modèle GNN performant pour la localisation de fuites. L'utilisation de la Focal Loss et de l'optimiseur AdamW, combinée à une gestion attentive des hyperparamètres et des mécanismes comme l'Early Stopping, devrait permettre un apprentissage efficace malgré le déséquilibre des données. L'évaluation sera basée sur un ensemble de métriques pertinentes (AUC-PR, MRR, Hits@k) complétées par une analyse approfondie des erreurs et une comparaison à des baselines. La validation expérimentale et l'ajustement des hyperparamètres resteront essentiels pour atteindre les meilleures performances possibles.

<div style="page-break-after: always;"></div>

## 4. Bibliographie

[2] Hajgato, M., Mavroeidis, A., Scholten, L., & Kapelan, Z. (2023). Leak Localization in Water Distribution Networks Using Graph Neural Networks. *Water Resources Research*, 59(7), e2022WR033685.

[9] Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal loss for dense object detection. *Proceedings of the IEEE international conference on computer vision* (ICCV).

[10] Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*.

---