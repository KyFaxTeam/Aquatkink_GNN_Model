

<div style="margin-top: 200px; text-align: center;">

# Conception Détaillée de l'Architecture GNN pour la Localisation de Fuites dans les Réseaux de Distribution d'Eau

<div style="margin: 0 auto; max-width: 650px; text-align: justify; margin-top: 50px;">

**Introduction**

Ce document présente une analyse approfondie et une proposition d'architecture de Réseau de Neurones sur Graphes (GNN) spécifiquement conçue pour la tâche de localisation de fuites sur les arêtes (tuyaux) d'un réseau de distribution d'eau (WDN), en utilisant l'exemple du réseau Net3. L'objectif est de traiter des données simulées représentant la structure du réseau, ses caractéristiques physiques statiques, et l'évolution temporelle de son état hydraulique (principalement la pression et potentiellement le débit) pour prédire la probabilité qu'une fuite soit présente sur chaque tuyau. La conception de l'architecture est une étape cruciale qui conditionne la capacité du modèle à apprendre les motifs spatio-temporels complexes caractéristiques des fuites. Nous examinerons les choix clés concernant les couches GNN, la gestion des caractéristiques d'arêtes, la profondeur, la largeur, le mécanisme de prédiction, et les composantes auxiliaires d'entraînement, en justifiant chaque décision par rapport aux spécificités du problème et à l'état de l'art.

</div>
</div>

<div style="page-break-after: always;"></div>

## 1. Rappel des Données d'Entrée et de la Tâche de Prédiction

Pour contextualiser les choix architecturaux, rappelons la nature des données et l'objectif :

*   **Entrée :** Un graphe représentant le WDN où :
    *   Les **nœuds** (jonctions, réservoirs, bassins) ont des caractéristiques statiques (élévation, type) et dynamiques (séquence temporelle de la *différence de pression* par rapport à une baseline simulée sans fuite).
    *   Les **arêtes** (tuyaux, pompes) ont des caractéristiques statiques (longueur, diamètre, rugosité, type) et dynamiques (statut ouvert/fermé, séquence temporelle optionnelle du débit). Ces caractéristiques d'arêtes (`edge_attr`) sont cruciales car elles gouvernent la propagation hydraulique.
*   **Sortie :** Pour chaque arête (tuyau) du graphe, une **probabilité** indiquant si cette arête est le siège de la fuite simulée (problème de classification binaire sur les arêtes).

<div style="page-break-after: always;"></div>

## 2. Choix Fondamental : Type de Couche GNN

La couche GNN est le cœur du modèle, définissant comment l'information des nœuds et des arêtes est combinée et propagée.

*   **Approches Spatiales vs Spectrales :** Les méthodes spatiales opèrent directement sur le voisinage des nœuds en agrégeant les messages des voisins. Elles sont généralement plus flexibles pour gérer des graphes hétérogènes et intégrer diverses caractéristiques, ce qui est adapté à notre cas. Les méthodes spectrales, basées sur la décomposition du Laplacien du graphe, sont mathématiquement élégantes mais peuvent être moins intuitives pour intégrer des caractéristiques dynamiques et d'arêtes complexes, et peuvent mal gérer les changements de structure du graphe (non pertinent ici, mais une considération générale). **Nous nous concentrerons sur les approches spatiales.**

*   **Candidats Principaux (Approches Spatiales) :**
    *   **GCN (Graph Convolutional Network)** [12] : Agrège les caractéristiques des voisins de manière isotrope (même poids pour tous les voisins, après normalisation par degré). Simple, efficace pour les graphes homophiles (nœuds connectés similaires). **Limitation majeure :** La formulation standard n'intègre pas directement les `edge_attr`.
    *   **GAT (Graph Attention Network)** [13] : Utilise des mécanismes d'auto-attention pour apprendre des poids différents pour chaque voisin lors de l'agrégation. Plus expressif que GCN, potentiellement capable de mieux cibler les informations pertinentes. **Limitation :** L'intégration directe des `edge_attr` dans le calcul d'attention nécessite des adaptations ou des variantes spécifiques [cf. 8 pour une application avec attention].
    *   **GraphSAGE** [20] : Échantillonne un voisinage de taille fixe et utilise diverses fonctions d'agrégation (moyenne, max, LSTM). Scalable, mais l'agrégation des voisins peut être moins précise que l'attention.
    *   **GIN (Graph Isomorphism Network)** [21] : Théoriquement l'une des architectures spatiales les plus puissantes, capable de distinguer différentes structures de graphes. Utilise un MLP pour transformer la somme des voisins. **Limitation :** Comme GCN, n'intègre pas nativement les `edge_attr`.

*   **Nécessité Cruciale : Intégration des `edge_attr` :** Les propriétés physiques des tuyaux (diamètre, rugosité) et l'état des pompes déterminent comment les changements de pression/débit se propagent. Ignorer ces `edge_attr` revient à ignorer une partie essentielle de la physique du système. Des couches GNN capables de les exploiter sont donc fortement préférables [5, 22].
    *   **NNConv (Neural Network Convolution)** [23] : Une approche très flexible où un petit réseau de neurones (MLP) est appliqué aux `edge_attr` pour générer une matrice de poids spécifique à l'arête, qui est ensuite utilisée pour transformer le message du nœud voisin. Permet au modèle d'apprendre comment les caractéristiques de l'arête modulent l'influence du voisin.
    *   **GINEConv (Graph Isomorphism Network with Edge features)** [24, utilisé dans 5] : Extension de GIN qui intègre explicitement les `edge_attr` dans sa fonction de mise à jour. Le message du voisin est transformé, puis combiné (souvent par addition) avec les `edge_attr` traités, avant d'être agrégé. Combine la puissance de GIN avec la prise en compte des arêtes.
    *   D'autres variantes existent (ex: GatedGraphConv avec edge gates, adaptations de GAT).

*   **Décision Architecturale (Couche GNN) :**
    *   **Choix Principal : `GINEConv` ou `NNConv`.** Ces couches permettent une intégration directe et flexible des caractéristiques physiques et dynamiques cruciales des arêtes (tuyaux, pompes), ce qui est essentiel pour modéliser correctement la propagation hydraulique et la signature d'une fuite. `GINEConv` est particulièrement attrayant en raison de sa base théorique solide et de son succès démontré dans des tâches similaires [5].
    *   **Alternative/Baseline : `GATConv`.** À considérer si l'attention sur les nœuds s'avère primordiale, mais nécessitera une méthode spécifique pour injecter l'information des `edge_attr` (par exemple, en les concaténant aux caractéristiques des nœuds voisins avant le calcul de l'attention, ou en utilisant des variantes de GAT conçues pour les `edge_attr`).

<div style="page-break-after: always;"></div>

## 3. Profondeur du Réseau (Nombre de Couches GNN)

La profondeur (nombre de couches `L`) détermine le rayon du voisinage effectif (champ récepteur) d'un nœud, c'est-à-dire jusqu'où l'information peut se propager en `L` étapes de message passing.

*   **Trade-off :**
    *   **Profondeur insuffisante :** Le modèle ne peut pas capturer les influences à longue distance. L'information d'un capteur lointain pourrait ne pas atteindre la zone de la fuite (et vice-versa) dans le calcul de l'embedding final.
    *   **Profondeur excessive :** Risque de **sur-lissage (over-smoothing)** [25], où les embeddings de tous les nœuds convergent vers une valeur similaire, perdant ainsi l'information locale distinctive nécessaire pour la localisation précise. Peut aussi entraîner une dilution de l'information initiale (vanishing gradient).
*   **Contexte WDN :** L'impact d'une fuite est généralement plus fort localement et s'atténue avec la distance. Il faut une profondeur suffisante pour connecter les capteurs pertinents à la zone de fuite, mais pas au point de rendre tous les embeddings indifférenciés. La structure souvent "linéaire" ou "arborescente" de certaines parties des WDN peut nécessiter plusieurs sauts pour relier des points distants.
*   **Pratique Courante :** Des profondeurs de 2 à 5 couches sont souvent utilisées pour les GNN sur des graphes de taille moyenne [5, 26].

*   **Décision Architecturale (Profondeur) :** Nous commencerons l'expérimentation avec une profondeur de **L = 2 ou 3 couches GNN**. Cela semble un bon compromis initial pour capturer une propagation suffisante de l'information hydraulique sans risquer un sur-lissage excessif immédiat. Ce sera un hyperparamètre clé à optimiser lors de la validation.

<div style="page-break-after: always;"></div>

## 4. Largeur du Réseau (Dimension des Embeddings Cachés)

La largeur (`hidden_channels`) est la dimension des vecteurs de caractéristiques (embeddings) que chaque couche GNN calcule.

*   **Trade-off :**
    *   **Largeur insuffisante :** Le modèle pourrait manquer de capacité pour représenter les relations complexes entre les caractéristiques hydrauliques, topologiques et dynamiques.
    *   **Largeur excessive :** Augmente le nombre de paramètres, le coût de calcul, le risque de sur-apprentissage (overfitting) et peut ne pas apporter d'amélioration significative si la complexité intrinsèque de l'information n'est pas si élevée.
*   **Valeurs Typiques :** Souvent des puissances de 2 (ex: 16, 32, 64, 128, 256), en fonction de la complexité de la tâche et de la taille du graphe/des données.

*   **Décision Architecturale (Largeur) :** Nous explorerons des dimensions cachées de **`hidden_channels` = 64 ou 128**. Ces valeurs offrent une capacité de représentation substantielle sans être excessives pour un réseau de la taille de Net3. Ce sera également un hyperparamètre important à régler.

<div style="page-break-after: always;"></div>

## 5. Mécanisme de Prédiction par Arête

Étant donné que la tâche est de localiser la fuite sur une *arête* (tuyau), le modèle doit produire une sortie pour chaque arête pertinente.

*   **Approche :** Après `L` couches GNN, nous obtenons des embeddings finaux pour chaque nœud (`h_u` pour le nœud `u`). Pour faire une prédiction sur l'arête `e` connectant les nœuds `u` et `v` :
    1.  **Rassembler les informations pertinentes :** Obtenir les embeddings finaux des nœuds connectés (`h_u`, `h_v`) et les caractéristiques statiques et/ou dynamiques propres à cette arête (`edge_attr_e`).
    2.  **Combiner ces informations :** Une méthode robuste consiste à **concaténer** ces vecteurs : `combined_e = [h_u || h_v || edge_attr_e]`. La concaténation préserve toute l'information individuelle des nœuds et de l'arête. D'autres opérateurs (somme, moyenne, produit scalaire) sont possibles mais peuvent entraîner une perte d'information.
    3.  **Classifier l'arête :** Passer ce vecteur combiné `combined_e` à travers un **classifieur**, typiquement un Multi-Layer Perceptron (MLP) final (ex: 1 ou 2 couches linéaires avec activation non-linéaire). Ce MLP apprend à mapper la représentation combinée de l'arête vers une prédiction de fuite.
    4.  **Obtenir la Probabilité :** La sortie du MLP (un logit) est passée à travers une **fonction d'activation Sigmoid** pour obtenir une probabilité de fuite P(fuite|`e`) comprise entre 0 et 1.

*   **Décision Architecturale (Sortie) :** Nous adopterons le mécanisme de prédiction par arête basé sur la **concaténation des embeddings des nœuds adjacents et des caractéristiques de l'arête (`[h_u || h_v || edge_attr_e]`), suivie d'un MLP de classification avec une sortie Sigmoid.**

<div style="page-break-after: always;"></div>

## 6. Composants Auxiliaires Essentiels

Pour améliorer l'apprentissage et la généralisation :

*   **Fonctions d'Activation :** Introduisent la non-linéarité nécessaire après les transformations linéaires dans les couches GNN et le MLP final.
    *   **Choix : ReLU (Rectified Linear Unit)** ou ses variantes comme **LeakyReLU** sont recommandées pour les couches cachées. Elles sont efficaces computationnellement et atténuent le problème de disparition du gradient [14].
*   **Normalisation :** Stabilise l'entraînement en normalisant les activations ou les embeddings, permettant potentiellement des taux d'apprentissage plus élevés et une convergence plus rapide.
    *   **Choix : Layer Normalization (LayerNorm)** ou **Batch Normalization (BatchNorm)** [14]. LayerNorm normalise sur les caractéristiques d'un seul échantillon et est souvent préférée quand la taille des batches est petite ou variable. BatchNorm normalise sur le batch. Leur position (avant/après activation) peut être expérimentée.
*   **Régularisation :** Techniques pour prévenir le sur-apprentissage sur les données d'entraînement.
    *   **Choix :**
        *   **Dropout** [27] : Met aléatoirement à zéro une fraction des unités (neurones ou caractéristiques) pendant l'entraînement. Peut être appliqué aux caractéristiques d'entrée, entre les couches GNN, ou dans le MLP final (taux p typique entre 0.2 et 0.5).
        *   **Weight Decay (Régularisation L2)** : Ajoute une pénalité proportionnelle au carré des poids du modèle à la fonction de coût. Encourage des poids plus petits. Souvent intégré directement dans les optimiseurs modernes (ex: AdamW [28]).

*   **Décision Architecturale (Auxiliaires) :** L'architecture intégrera **ReLU/LeakyReLU**, **LayerNorm/BatchNorm**, **Dropout**, et **Weight Decay** pour favoriser un entraînement stable et une meilleure généralisation.



<div style="page-break-after: always;"></div>

## 7. Résumé Conceptuel du Flux de l'Architecture

1.  **Entrée :** Graphe avec caractéristiques de nœuds (`x`, séquences de diff. de pression) et d'arêtes (`edge_attr`, statiques et dynamiques).
2.  **Traitement Initial (Optionnel) :** MLP pour projeter les caractéristiques initiales dans la dimension cachée.
3.  **Couches GNN (L = 2-3) :** Séquence de couches `GINEConv` ou `NNConv` avec ReLU/LeakyReLU, Normalisation, Dropout. Chaque couche met à jour les embeddings des nœuds en utilisant les messages des voisins pondérés/modulés par les `edge_attr`.
4.  **Embeddings Finaux :** Obtention des embeddings `h_u` pour chaque nœud `u` après L couches.
5.  **Préparation de la Prédiction par Arête :** Pour chaque arête `e = (u, v)`, former le vecteur `combined_e = [h_u || h_v || edge_attr_e]`.
6.  **MLP de Classification :** Appliquer un MLP (avec ReLU/LeakyReLU, Dropout) à `combined_e`.
7.  **Sortie :** Appliquer une fonction Sigmoid pour obtenir P(fuite|`e`) pour chaque arête.
8.  **Calcul de la Perte :** Utiliser la BCE Loss (potentiellement pondérée) entre les probabilités prédites et les labels réels (0 ou 1).
9.  **Optimisation :** Ajuster les poids du modèle via rétropropagation et un optimiseur (ex: AdamW [28] qui intègre Weight Decay).

<div style="page-break-after: always;"></div>

## 8. Conclusion

L'architecture GNN proposée ici est spécifiquement conçue pour la localisation de fuites dans les WDN, en mettant l'accent sur l'intégration cruciale des caractéristiques d'arêtes (`edge_attr`) via des couches comme `GINEConv` ou `NNConv`. Avec une profondeur et une largeur modérées (2-3 couches, 64-128 dimensions cachées), un mécanisme de prédiction par arête basé sur la concaténation et un MLP final, ainsi que l'utilisation de composants d'entraînement standards (ReLU, Normalisation, Dropout, BCE Loss), cette architecture constitue un point de départ solide et bien justifié. Il est impératif de souligner que les choix finaux et les hyperparamètres optimaux devront être déterminés par une **validation expérimentale rigoureuse** sur les données simulées de Net3.

<div style="page-break-after: always;"></div>

## 9. Bibliographie

[1] Klise, K. A., Murray, R., & Haxton, T. (2018). An overview of the Water Network Tool for Resilience (WNTR). *EPANET Technology Exchange Workshop*.

[2] Zhou, S., Zhou, Y., Wang, J., & Liu, Y. (2021). Graph Neural Network-Based Pipe Leak Localization in Water Distribution Systems. *Journal of Water Resources Planning and Management*, 147(10), 04021063.

[3] Sanz, G., Pérez, R., Kapelan, Z., Savic, D., &cones, J. (2016). Leak detection and localization through demand and pressure data analysis. *Procedia Engineering*, 154, 1258-1265.

[4] Wu, Z., Wang, X., & Jiang, R. (2020). Graph neural network for pipeline leak detection. *arXiv preprint arXiv:2008.08902*.

[5] Hajgato, M., Mavroeidis, A., Scholten, L., & Kapelan, Z. (2023). Leak Localization in Water Distribution Networks Using Graph Neural Networks. *Water Resources Research*, 59(7), e2022WR033685.

[6] Zhang, Q., Liu, H., Wu, Y., & Wang, W. (2022). Leak detection in water distribution systems based on graph neural networks. *Journal of Hydroinformatics*, 24(4), 901-916.

[7] Bakker, M., Vreeburg, J. H. G., & van Schagen, K. M. (2013). A framework for modeling the impact of hydraulic events in water distribution networks. *Drinking Water Engineering and Science*, 6(1), 25-32.

[8] Tom, L., Yoon, S., & Choi, J. (2021). Leak Detection in Water Distribution Networks Using Graph Neural Networks with Attention Mechanism. *Water*, 13(18), 2509.

[9] Colombo, A. F., Karney, B. W., & Brunone, B. (2009). A review of the methods for detecting leaks in water distribution systems. *Urban Water Journal*, 6(5), 391-408.

[10] Araujo, L. S., Ramos, H. M., & Coelho, S. T. (2021). Pressure sensors location for leak detection in water distribution systems: A literature review. *Water*, 13(8), 1067.

[11] Lambert, A. O., Fantozzi, M., & Thornton, J. (2014). Practical Experience in Using the IWA Methodology for Calculating the Water Balance and Water Losses in Different Countries. *Water Practice & Technology*, 9(1), 79-90.

[12] Kipf, T. N., & Welling, M. (2017). Semi-Supervised Classification with Graph Convolutional Networks. *International Conference on Learning Representations (ICLR)*.

[13] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). Graph Attention Networks. *International Conference on Learning Representations (ICLR)*.

[14] Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.

[15] Rossman, L. A. (2000). *EPANET 2 users manual*. US Environmental Protection Agency, National Risk Management Research Laboratory.

[16] Hastie, T., Tibshirani, J., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Series in Statistics.

[17] Brentan, B. M., Meirelles, G., Luvizotto Jr, E., & Izquierdo, J. (2018). Hybrid calibration approach for water distribution network models. *Journal of Water Resources Planning and Management*, 144(8), 04018045.

[18] Ye, G., & Yao, L. (2021). A review on data-driven anomaly detection for water distribution systems. *Water Supply*, 21(7), 3175-3190.

[19] Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. *IEEE Transactions on knowledge and data engineering*, 22(10), 1345-1359.

[20] Hamilton, W. L., Ying, R., & Leskovec, J. (2017). Inductive representation learning on large graphs. *Advances in neural information processing systems (NIPS)*, 30.

[21] Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). How Powerful are Graph Neural Networks? *International Conference on Learning Representations (ICLR)*.

[22] Battaglia, P. W., Hamrick, J. B., Bapst, V., Sanchez-Gonzalez, A., Zambaldi, V., Malinowski, M., ... & Pascanu, R. (2018). Relational inductive biases, deep learning, and graph networks. *arXiv preprint arXiv:1806.01261*. (Article général sur l'importance des relations/arêtes).

[23] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). Neural message passing for quantum chemistry. *International conference on machine learning (ICML)*. (Introduit le framework Message Passing Neural Networks, incluant des concepts similaires à NNConv).

[24] Hu, W., Fey, M., Zitnik, M., Dong, Y., Ren, H., Liu, B., ... & Leskovec, J. (2020). Open Graph Benchmark: Datasets for Machine Learning on Graphs. *Advances in Neural Information Processing Systems (NeurIPS)*, 33, 22118-22133. (Décrit OGB et mentionne GINE comme une baseline performante intégrant edge features).

[25] Li, Q., Han, Z., & Wu, X. M. (2018). Deeper insights into graph convolutional networks for semi-supervised learning. *Thirty-Second AAAI conference on artificial intelligence*. (Discute du problème de over-smoothing).

[26] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Philip, S. Y. (2020). A comprehensive survey on graph neural networks. *IEEE transactions on neural networks and learning systems*, 32(1), 4-24. (Revue générale des GNN).

[27] Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. *The journal of machine learning research*, 15(1), 1929-1958.

[28] Loshchilov, I., & Hutter, F. (2017). Decoupled weight decay regularization. *arXiv preprint arXiv:1711.05101*. (Introduit AdamW).

---

