# Plan détaillé pour la génération et la sauvegarde des données (Stage 1)

## Objectif
Simuler des scénarios hydrauliques (avec et sans fuite) sur le réseau Net3 à l’aide de WNTR/EPANET, et sauvegarder les résultats au format HDF5 pour l’entraînement du GNN.

---

## Recommandations et Justifications

1. **Nombre de scénarios**  
   Générer des milliers de scénarios de fuite et de non-fuite (par exemple, 1000+ avec fuite, 200+ sans fuite) est fortement recommandé. Cela assure une diversité suffisante pour un apprentissage robuste du GNN, limite le surapprentissage et favorise la généralisation, notamment pour le transfert d’apprentissage.

2. **Emplacement des capteurs virtuels**  
   Dans un réseau réel, les capteurs sont fixes. Pour garantir la transférabilité et la pertinence pratique, il est préférable de garder les emplacements des capteurs virtuels fixes pour tous les scénarios. Sélectionner 10–20% des jonctions, en privilégiant la couverture et la sensibilité hydraulique.

3. **Durée de simulation**  
   Une durée de 24 heures par scénario est un standard efficace. Cela permet de capturer les variations journalières de la demande et des opérations, et d’apprendre à la fois les comportements stationnaires et transitoires.

4. **Paramètres de fuite**  
   - Localisation, instant de début, durée et sévérité de la fuite doivent être tirés aléatoirement dans des bornes réalistes pour chaque scénario.
   - Pour chaque scénario avec fuite, générer un scénario baseline (sans fuite) sous conditions identiques.

---
## Ce qu’il faut sauvegarder pour chaque scénario (fuite ou baseline)

Chaque fichier HDF5 (un par scénario) doit contenir :

### 1. **Résultats hydrauliques (groupes/datasets numériques)**
- `/leak_results/pressures` : Tableau [n_capteurs, n_timesteps] des pressions absolues aux capteurs virtuels pour le scénario avec fuite.
- `/baseline_results/pressures` : Tableau [n_capteurs, n_timesteps] des pressions absolues aux capteurs virtuels pour le scénario baseline (sans fuite, mêmes conditions).
- `/leak_results/flows` (optionnel) : Tableau [n_pipes, n_timesteps] des débits dans chaque tuyau pour le scénario avec fuite.
- `/baseline_results/flows` (optionnel) : Tableau [n_pipes, n_timesteps] des débits dans chaque tuyau pour le scénario baseline.

### 2. **Attributs statiques du réseau**
- `/static/node_ids` : Liste des IDs des nœuds capteurs.
- `/static/pipe_ids` : Liste des IDs des tuyaux.
- `/static/node_elevations` : Tableau [n_capteurs] des élévations des nœuds capteurs.
- `/static/pipe_lengths` : Tableau [n_pipes] des longueurs des tuyaux.
- `/static/pipe_diameters` : Tableau [n_pipes] des diamètres des tuyaux.
- `/static/pipe_roughness` : Tableau [n_pipes] des rugosités des tuyaux.

### 3. **Label cible**
- `/label/leak_pipe_id` : ID du tuyau en fuite (ou None si pas de fuite).
- `/label/leak_vector` : Vecteur binaire [n_pipes], 1 à l’indice du tuyau en fuite, 0 ailleurs.

### 4. **Métadonnées du scénario**
- `/metadata/scenario_id` : Identifiant unique du scénario.
- `/metadata/type` : "leak" ou "baseline".
- `/metadata/leak_start_time` : Temps de début de la fuite (si applicable).
- `/metadata/leak_duration` : Durée de la fuite (si applicable).
- `/metadata/leak_severity` : Sévérité/aire de la fuite (si applicable).
- `/metadata/sensor_node_ids` : Liste des IDs des nœuds capteurs utilisés.
- `/metadata/baseline_scenario_id` : ID du scénario baseline associé.

---
## Résumé du plan

- Générer des milliers de scénarios (ex : 1000+ fuite, 200+ non-fuite).
- Utiliser un ensemble fixe d’emplacements de capteurs virtuels (10–20% des jonctions).
- Simuler chaque scénario sur 24 heures.
- Paramètres de fuite aléatoires pour la diversité.
- Stocker les résultats dans des fichiers HDF5 dans `data/raw/`.

---

## Structure HDF5 recommandée

```
scenario_X.h5
├── /leak_results/
│     ├── pressures
│     └── flows (optionnel)
├── /baseline_results/
│     ├── pressures
│     └── flows (optionnel)
├── /static/
│     ├── node_ids
│     ├── node_elevations
│     ├── pipe_ids
│     ├── pipe_lengths
│     ├── pipe_diameters
│     └── pipe_roughness
├── /label/
│     ├── leak_pipe_id
│     └── leak_vector
├── /metadata/
│     ├── scenario_id
│     ├── type
│     ├── leak_start_time
│     ├── leak_duration
│     ├── leak_severity
│     ├── sensor_node_ids
│     └── baseline_scenario_id
```

---

## Bonnes pratiques

- Fixer la seed aléatoire pour la reproductibilité.
- Permettre la configuration (nombre de scénarios, pourcentage de capteurs, durée, etc.) via arguments ou fichier de config.
- Vérifier la cohérence des fichiers HDF5 générés (features, labels, métadonnées).
- **Vérification** : Après chaque sauvegarde, vérifier l’intégrité et la cohérence des données.
- **Documentation** : Documenter chaque champ dans le code pour garantir la reproductibilité.

---

Ce plan explicite précisément les données à sauvegarder pour chaque scénario, en cohérence avec les exigences de Hypotheses_de_donnes.md et data_saving.md.