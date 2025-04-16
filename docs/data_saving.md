
# Stratégie de Stockage et de Chargement des Données Simulées pour l'Entraînement d'un GNN de Localisation de Fuites

**1. Introduction**

Une gestion efficace des données est cruciale pour le succès d'un projet d'apprentissage profond basé sur des simulations, tel que la localisation de fuites dans un réseau de distribution d'eau (WDN) à l'aide de Réseaux de Neurones sur Graphes (GNN). Ce document détaille une stratégie complète pour stocker les résultats des simulations hydrauliques (via WNTR) et pour charger efficacement ces données lors de l'entraînement et de l'évaluation du modèle GNN avec PyTorch Geometric (PyG). L'approche proposée vise la modularité, la reproductibilité, l'efficacité du stockage et la flexibilité du chargement, notamment en permettant des divisions programmatiques des ensembles de données.

**2. Principes Directeurs**

*   **Séparation Brut/Traité :** Maintenir une distinction claire entre les sorties directes des simulations (brutes) et les données formatées pour le GNN (traitées).
*   **Stockage Efficace :** Utiliser des formats de fichiers adaptés au type et au volume des données (ex: HDF5 pour les séries temporelles brutes, `.pt` pour les objets PyG).
*   **Flexibilité des Partitions :** Ne pas figer la répartition train/validation/test au niveau du stockage ; permettre une définition programmatique pour faciliter l'expérimentation et la validation croisée.
*   **Chargement Optimisé :** Utiliser les outils de l'écosystème PyTorch/PyG (`Dataset`, `DataLoader`, `num_workers`) pour minimiser les goulots d'étranglement I/O.
*   **Reproductibilité :** Assurer que le processus de génération, de traitement et de division des données puisse être reproduit de manière fiable.

**3. Stockage des Données Brutes (Sorties de Simulation WNTR)**

*   **Objectif :** Conserver les résultats essentiels de chaque simulation pour la reproductibilité, le débogage et le traitement ultérieur, sans stocker l'intégralité des résultats souvent volumineux de WNTR.
*   **Contenu par Scénario :** Pour chaque simulation `i` :
    *   Résultats Hydrauliques Clés (Séries Temporelles) : Pressions aux nœuds capteurs virtuels (`P_abs_leak`), et les pressions correspondantes de la simulation **baseline** associée (`P_abs_baseline`). Débits/statuts optionnels.
    *   Métadonnées : ID Scénario, Type (Fuite/Pas de Fuite), infos sur la fuite (`leak_pipe_id`, sévérité, timing), ID baseline associée.
    *   Label : `leak_pipe_id` (ou None si pas de fuite).
*   **Format de Stockage : HDF5 par Scénario.**
    *   Chaque scénario (simulation `i` + sa baseline) est stocké dans un unique fichier `scenario_i.h5`.
    *   Structure interne HDF5 organisée (ex: groupes `/metadata`, `/leak_results`, `/baseline_results`) pour stocker les métadonnées et les séries temporelles (pressions, etc.) sous forme de datasets numériques.
    *   Avantages : Compact (compression), I/O rapide, bien adapté aux tableaux numériques, maintient les données liées ensemble.
*   **Localisation :** Tous les fichiers `.h5` sont stockés dans un **répertoire unique** `data/raw/`.

**4. Stockage des Données Traitées (Format GNN / PyG)**

*   **Objectif :** Transformer les données brutes en objets `Data` PyTorch Geometric, prêts à être directement consommés par le modèle GNN, et les sauvegarder pour éviter de répéter le traitement à chaque exécution.
*   **Processus de Traitement (Implémenté dans `Dataset.process()`) :**
    1.  Itérer sur chaque `scenario_i.h5` dans `data/raw/`.
    2.  Charger les données brutes nécessaires depuis HDF5.
    3.  **Calculer les caractéristiques GNN :**
        *   Différence de pression (`x = P_abs_leak - P_abs_baseline`).
        *   Préparer les séquences temporelles.
        *   Assembler les caractéristiques statiques/dynamiques pour les nœuds (`x`) et les arêtes (`edge_attr`).
        *   Construire le vecteur cible binaire `y` (taille `num_edges`).
        *   Appliquer la normalisation/standardisation.
    4.  Créer l'objet `torch_geometric.data.Data` contenant `x`, `edge_index`, `edge_attr`, `y`, etc.
    5.  Sauvegarder l'objet `Data` dans `data/processed/scenario_{i}.pt` via `torch.save()`.
*   **Format de Stockage : Fichiers `.pt` PyTorch par Scénario.**
    *   Chaque scénario traité correspond à un fichier `.pt` contenant un objet `Data` PyG.
    *   Avantages : Format natif PyG, intégration directe.
*   **Localisation :** Tous les fichiers `.pt` sont stockés dans un **répertoire unique** `data/processed/`.

**5. Chargement des Données pour l'Entraînement / Évaluation**

*   **Objectif :** Charger efficacement les données traitées (`.pt`) par lots pour l'entraînement, la validation et le test, tout en permettant une définition flexible des partitions.
*   **Composants Clés :**
    1.  **Classe `Dataset` PyG (`WDNLeakDataset`) :**
        *   Hérite de `torch_geometric.data.Dataset`.
        *   Initialisée avec `root='./data'`.
        *   `raw_dir` pointe vers `data/raw/`, `processed_dir` vers `data/processed/`.
        *   `process()`: Contient la logique de transformation `raw/` (.h5) -> `processed/` (.pt), exécutée seulement si `processed/` est vide ou un flag manque.
        *   `len()`: Retourne le nombre total de fichiers `.pt` dans `processed/`.
        *   `get(idx)`: Charge et retourne `data/processed/scenario_{idx}.pt`.
    2.  **Division Programmatique des Indices (dans le script `train.py`) :**
        *   Obtenir le nombre total d'échantillons `N = len(dataset)`.
        *   Générer les indices `0` à `N-1`.
        *   **Mélanger (Shuffle)** les indices (avec `random.seed()` fixe pour la reproductibilité).
        *   **Répartir** les indices mélangés en listes `train_indices`, `val_indices`, `test_indices` selon les pourcentages souhaités.
    3.  **`torch.utils.data.Subset` :**
        *   Créer des instances `Subset(dataset, indices)` pour chaque partition (train, val, test) à partir du `Dataset` complet et des listes d'indices correspondantes.
    4.  **`torch_geometric.loader.DataLoader` :**
        *   Créer un `DataLoader` pour chaque `Subset`.
        *   Utiliser `shuffle=True` pour le `DataLoader` d'entraînement (mélange les indices de ce subset à chaque époque).
        *   **Crucial : Utiliser `num_workers > 0`** (ex: 4, 8) pour charger les fichiers `.pt` en parallèle et masquer la latence I/O.
        *   Définir `batch_size`.
*   **(Alternative) `InMemoryDataset` :** Si la taille totale de tous les fichiers `.pt` est gérable en RAM, hériter de `InMemoryDataset` au lieu de `Dataset`. Les données sont chargées une seule fois au début, offrant le chargement le plus rapide possible pendant l'entraînement. La logique de `process()` reste similaire, mais les objets `Data` sont stockés dans une liste en mémoire. La division par indices et `Subset` s'appliquent toujours.

**6. Structure de Répertoire Recommandée (Unifiée)**

```
projet/
│
├── data/                     # Racine unique pour les données
│   ├── raw/                  # Contient TOUS les scenario_i.h5 bruts
│   │   ├── scenario_0.h5
│   │   └── ...
│   └── processed/            # Contient TOUS les scenario_i.pt traités
│       ├── scenario_0.pt
│       └── ...
│
├── src/                      # Code source
│   ├── datasets.py           # Définition de WDNLeakDataset
│   ├── models.py             # Définition de l'architecture GNN
│   ├── train.py              # Script d'entraînement (inclut le split des indices)
│   ├── generate_data.py      # Script pour lancer les simulations et créer les .h5 bruts
│   └── ...
│
└── ... (autres fichiers: config, scripts, etc.)
```

**7. Conclusion**

Cette stratégie de gestion des données propose un pipeline clair et flexible :
1.  **Génération Brute :** Simulations WNTR -> Fichiers **HDF5** par scénario (incluant baseline) dans `data/raw/`.
2.  **Traitement :** `Dataset.process()` lit les HDF5 -> Calcule features GNN -> Sauvegarde objets `Data` PyG en fichiers **`.pt`** dans `data/processed/`.
3.  **Chargement :** Le script principal définit les splits **train/val/test via indices**, utilise `Subset` sur le `Dataset` complet, et charge les données via `DataLoader` avec **`num_workers`** (ou via `InMemoryDataset` si possible).

