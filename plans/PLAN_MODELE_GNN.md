# Plan détaillé pour l’architecture du modèle GNN (Stage 4)

## Objectif
Développer une architecture de Réseau de Neurones sur Graphes (GNN) pour la localisation de fuites sur les arêtes (tuyaux) d’un réseau de distribution d’eau, en exploitant au mieux les caractéristiques physiques et dynamiques du réseau Net3.

---

## 1. Structure générale du modèle

- **Entrée :**
  - Graphe avec :
    - Caractéristiques des nœuds : séquences temporelles de différence de pression, élévation, etc.
    - Caractéristiques des arêtes : longueur, diamètre, rugosité, statut, etc.
- **Sortie :**
  - Pour chaque arête (tuyau), probabilité d’être le siège d’une fuite (classification binaire par arête).

---

## 2. Choix des couches GNN

- **Justification :**
  - Les couches GNN doivent intégrer explicitement les `edge_attr` (caractéristiques physiques des tuyaux) pour modéliser la propagation hydraulique.
  - **GINEConv** (Graph Isomorphism Network with Edge features) ou **NNConv** (Neural Network Convolution) sont recommandées :
    - GINEConv : extension puissante de GIN, intègre les `edge_attr` dans la mise à jour des messages, validée dans la littérature (Hu et al., 2020 ; Hajgato et al., 2023).
    - NNConv : très flexible, utilise un MLP pour moduler les messages selon les `edge_attr`.
  - **Alternative** : GATConv (avec injection des `edge_attr`) pour comparaison.

---

## 3. Profondeur et largeur du réseau

- **Profondeur** : 2 à 3 couches GNN (compromis entre champ récepteur suffisant et sur-lissage).
- **Largeur** : 64 à 128 canaux cachés (capacité adaptée à la taille de Net3).

---

## 4. Mécanisme de prédiction par arête

- Pour chaque arête (u, v) :
  1. Concaténer les embeddings finaux des nœuds (h_u, h_v) et les `edge_attr` :  
     `combined_e = [h_u || h_v || edge_attr_e]`
  2. Passer ce vecteur dans un MLP (1–2 couches, avec ReLU/LeakyReLU, dropout, normalisation).
  3. Sortie : logit → sigmoid → probabilité de fuite pour chaque arête.

---

## 5. Composants auxiliaires

- **Activation** : ReLU ou LeakyReLU après chaque couche.
- **Normalisation** : LayerNorm ou BatchNorm après les couches GNN/MLP.
- **Dropout** : 0.2–0.5 entre les couches pour la régularisation.
- **Weight Decay** : utilisé dans l’optimiseur (AdamW).

---



## 6. Schéma Mermaid de l’architecture

```mermaid
flowchart TD
    A[Entrée : Graphe (x, edge_attr, edge_index)] --> B[Couches GNN (GINEConv/NNConv) x2-3]
    B --> C[Embeddings de nœuds]
    C --> D[Pour chaque arête : concat(h_u, h_v, edge_attr_e)]
    D --> E[MLP + Sigmoid]
    E --> F[Proba fuite par arête]
    F --> G[Calcul Focal Loss vs. labels]
    G --> H[Optimiseur AdamW + Scheduler]
    F --> I[Métriques : AUC-PR, MRR, Hits@k, etc.]
```

---

## 7. Justification synthétique

- **Intégration des edge_attr** : essentielle pour modéliser la physique du réseau.
- **Profondeur/largeur modérées** : équilibre entre expressivité et stabilité.

- **Composants auxiliaires** : favorisent la généralisation et la stabilité.

---

Ce plan est aligné avec l’état de l’art et les exigences du projet, et servira de référence pour l’implémentation du modèle dans `src/models.py`.