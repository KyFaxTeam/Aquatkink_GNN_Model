

<div style="margin-top: 240px; text-align: center;">

# Conception d'un Modèle GNN pour la Détection de Fuites sur le Réseau Net3

<div style="margin: 0 auto; max-width: 650px; text-align: justify; margin-top: 50px;">

**Introduction**

Ce document décrit les étapes initiales de la conception d'un modèle basé sur les Réseaux de Neurones sur Graphes (GNN) pour la détection et la localisation de fuites dans le réseau de distribution d'eau potable Net3. Nous définirons d'abord les composants clés d'un tel réseau, puis nous préciserons l'objectif du modèle, les hypothèses fondamentales, comment structurer les données nécessaires à son entraînement, et enfin, discuterons des implications pour une application réelle.

</div>
</div>

<div style="page-break-after: always;"></div>

## 1. Composants d'un Réseau de Distribution d'Eau Potable (WDN)

Comprendre les éléments d'un WDN comme Net3 est essentiel pour modéliser correctement le système :

*   **Jonctions (Junctions) :** Points du réseau où plusieurs tuyaux se connectent. En réalité, cela représente des intersections de conduites, des points de livraison majeurs, ou des **points où la demande en eau est supposée concentrée**. Chaque jonction a une **élévation** et une **demande en eau** (qui peut varier dans le temps, définie par une "demande de base" et un "profil de demande"). **C'est à ce niveau que la consommation des utilisateurs (maisons, industries, etc.) est modélisée de manière agrégée.**

*   **Tuyaux (Pipes) :** Conduites qui transportent l'eau entre les jonctions, les réservoirs, les bassins, etc. Ils sont caractérisés par des **propriétés physiques (longueur, diamètre, matériau, rugosité)** qui déterminent les pertes de charge (perte de pression due à la friction) en utilisant des formules comme Hazen-Williams ou Darcy-Weisbach.

*   **Réservoirs (Reservoirs) :** Représentent des sources d'eau externes au réseau (comme un lac, une rivière traitée, ou une connexion à un réseau plus large) considérées comme ayant une **capacité illimitée** et imposant une **charge hydraulique (pression équivalente) constante ou variable** dans le temps au point de connexion. Net3 utilise "River" et "Lake" comme réservoirs.

*   **Bassins (Tanks) :** Structures de stockage d'eau *à l'intérieur* du réseau (châteaux d'eau, réservoirs surélevés ou au sol). Leur **niveau d'eau varie** en fonction des apports et des demandes, ce qui influence la pression dans les zones avoisinantes. Ils sont caractérisés par leur géométrie (diamètre, niveaux min/max/initial) et servent à équilibrer les pressions et à répondre aux pics de demande. Net3 possède 3 bassins (Tank 1, 2, 3).

*   **Pompes (Pumps) :** Dispositifs qui ajoutent de l'énergie (charge) au système pour augmenter la pression et/ou transporter l'eau vers des zones plus élevées ou éloignées. Elles sont définies par leur **courbe caractéristique** (charge ajoutée en fonction du débit) ou une puissance constante. Net3 a deux pompes (ID 10 depuis Lake, ID 335 depuis River).

*   **Vannes (Valves) :** Dispositifs utilisés pour contrôler le débit ou la pression dans un tuyau. Il en existe plusieurs types (vannes d'isolement pour ouvrir/fermer un tronçon, réducteurs de pression (PRV), vannes de maintien de pression (PSV), vannes de régulation de débit (FCV), etc.). Chaque type a un comportement spécifique et des paramètres associés. Net3 n'a pas de vannes explicites définies dans la section `[VALVES]`, mais utilise des contrôles sur les pompes et un tuyau pour simuler des actions de contrôle.

*   **Sources (Sources) :** Points où l'eau entre dans le réseau. Dans EPANET/Net3, cela correspond souvent aux **Réservoirs** (Lake, River) ou à des jonctions spécifiques où une "source de qualité" est définie (pour tracer l'origine ou la qualité de l'eau).

*   **Capteurs (Sensors) :** Dispositifs *réels* installés sur un réseau physique pour mesurer des grandeurs comme la pression, le débit, le niveau d'eau, ou la qualité de l'eau. **Point important :** EPANET/WNTR *simule* le comportement hydraulique *complet* du réseau (pression à chaque nœud, débit dans chaque tuyau) basé sur les lois de la physique. Pour entraîner un modèle GNN, nous n'avons pas besoin de simuler les capteurs eux-mêmes. Nous *utilisons* la simulation pour générer les valeurs (ex: pression, débit) aux *endroits où nous supposerions que des capteurs seraient placés dans la réalité*. Le GNN apprendra ensuite à détecter les fuites à partir de ces données "virtuellement captées". Le choix de l'emplacement et du type de ces capteurs virtuels est une décision de conception cruciale [3, 5]. Cette approche, utilisant des données simulées faute de données réelles labellisées suffisantes, est couramment employée dans la recherche sur la détection de fuites basée sur l'IA [2, 5, 8].

*   **Contrôles (Controls) :** Règles logiques (simples ou basées sur des conditions) qui modifient l'état du réseau (ex: statut d'une pompe, réglage d'une vanne, fermeture d'un tuyau) en fonction du temps ou de conditions hydrauliques spécifiques (niveau d'un bassin, pression à un nœud). Ils rendent le comportement du réseau dynamique et permettent de simuler les opérations réelles. Dans Net3, des contrôles gèrent les pompes en fonction du temps ou du niveau d'un bassin.

**Clarification sur la Modélisation de la Demande (Consommation) :**

Il est important de noter que les modèles hydrauliques comme EPANET/WNTR, pour des raisons de complexité et d'échelle, ne modélisent généralement pas chaque point de consommation individuel (chaque maison, chaque robinet). À la place, ils utilisent une approche de **demande agrégée** (ou "lumped demand"). La consommation d'eau d'une zone géographique (quartier, groupe de bâtiments) est **concentrée et attribuée à la jonction la plus proche ou la plus représentative**. Cette demande agrégée à la jonction est définie par :
1.  Une **Demande de Base :** Le taux moyen de consommation à cette jonction.
2.  Un **Profil de Demande (Pattern) :** Un multiplicateur qui varie au cours du temps (ex: sur 24 heures) pour simuler les fluctuations typiques de la consommation (pics matin et soir, baisse la nuit).

Donc, les "points de sortie" de l'eau pour la consommation sont implicitement représentés par l'attribut de **demande aux nœuds de type Jonction**.

<div style="page-break-after: always;"></div>

## 2. Définition du Problème et Hypothèses pour Net3

### Objectif Principal

L'objectif principal de ce projet est de développer un modèle GNN capable de **localiser une fuite unique sur un tuyau spécifique** dans le réseau Net3, en se basant sur des données simulées issues de capteurs virtuels.

### Sortie Attendue du Modèle

Le modèle devra produire, pour un instant ou une période donnée, un **vecteur de probabilités (ou scores)**, où chaque élément correspond à un tuyau du réseau et représente la probabilité estimée que ce tuyau soit le siège de la fuite.

### Hypothèses Clés

Nous posons les hypothèses suivantes pour cadrer le développement initial :

1.  **Fuite Unique :** Nous supposons qu'**une seule fuite significative** peut se produire à la fois dans le réseau [4, 8]. Bien que des fuites multiples puissent survenir, l'hypothèse de fuite unique est une simplification courante et nécessaire pour établir une base de référence et évaluer initialement les modèles, le cas multi-fuites présentant des défis supplémentaires significatifs en termes de superposition des signaux et de complexité du modèle, comme souligné dans diverses revues de méthodes de détection [9].

2.  **Disponibilité des Capteurs (Virtuels) :** Nous simulons la présence de **capteurs de pression** à un sous-ensemble de **jonctions** du réseau (par exemple, 10-20% des nœuds, choisis stratégiquement ou aléatoirement) [3, 5]. Le placement optimal des capteurs est d'ailleurs un domaine de recherche actif visant à maximiser la détectabilité des fuites avec un nombre limité de capteurs, souvent abordé par des techniques d'optimisation [10]. Optionnellement, nous pourrions ajouter des capteurs de débit virtuels sur quelques tuyaux clés [5, 6]. La performance dépendra fortement de ce choix.

3.  **Nature des Données :** Les données d'entraînement et de test seront entièrement **générées par simulation** en utilisant WNTR/EPANET [1, 2, 7, 8]. Cela permet de créer un grand nombre de scénarios contrôlés (avec et sans fuite, fuites de tailles et localisations variées).

4.  **Données Temporelles :** Nous utiliserons des **séquences temporelles** des lectures des capteurs (principalement la pression) comme caractéristiques d'entrée pour les nœuds [2, 5, 8]. Une séquence temporelle capture mieux la dynamique d'une fuite que des mesures instantanées.

5.  **Dynamique du Réseau :** Bien que la topologie (connexions) soit fixe, le réseau Net3 est **hydrauliquement dynamique**. Le GNN devra gérer ces variations normales.

6.  **Coût des Erreurs :** Nous considérons que **manquer une fuite réelle (Faux Négatif) est plus coûteux** que déclencher une fausse alarme (Faux Positif), notamment en termes de pertes d'eau non comptabilisées (composante majeure de l'Eau Non Facturée), de coûts énergétiques accrus, et de risques potentiels [11]. Ce déséquilibre justifie le choix de métriques d'évaluation privilégiant la sensibilité (ex: Rappel, AUC-PR [cf. 5]) et pourrait influencer la conception d'une fonction de perte pondérée.

<div style="page-break-after: always;"></div>

## 3. Ingénierie des Données pour le GNN

**Définition de la Baseline (Fonctionnement Normal) :**
Avant de détailler les caractéristiques, définissons la **"baseline"** ou **"état de référence normal"**. Elle représente le comportement hydraulique simulé du réseau **en l'absence de toute fuite**, mais sous les **mêmes conditions opérationnelles externes et de demande** que celles d'un scénario de fuite donné. Pour chaque simulation avec fuite, une simulation de baseline correspondante (même période, mêmes demandes/contrôles, sans la fuite) est générée. Ceci est crucial pour que la différence `Pression_Scénario(t) - Pression_Baseline(t)` isole l'impact de la fuite.

Basé sur les hypothèses précédentes, voici comment nous allons structurer les données :

*   **Représentation du Graphe :**
    *   **Nœuds :** Jonctions, Réservoirs, Bassins.
    *   **Arêtes :** Tuyaux, Pompes. L'état des pompes/tuyau contrôlé sera un attribut dynamique.
    *   **Directionnalité :** Le graphe sera traité comme **non-dirigé** pour la propagation GNN, standard pour GCN [12] ou GAT [13].

*   **Caractéristiques des Nœuds (`x`) :**
    *   *Dynamiques :*
        *   **Option Principale Recommandée:** Séquence temporelle de la **différence de pression par rapport à la baseline sans fuite** (`P_abs(t) - P_baseline(t)`) simulée aux nœuds capteurs.
        *   *(Alternative/Complémentaire):* Pression absolue, différence temporelle (`P(t)-P(t-1)`). Le choix final reposera sur l'expérimentation.
    *   *Statiques :* **Élévation**, **Type de nœud**, Demande de base.

*   **Caractéristiques des Arêtes (`edge_attr`) :**
    *   *Dynamiques :* Séquence temporelle du **débit** (optionnel), **Statut** (pompes/tuyau contrôlé).
    *   *Statiques :* **Longueur**, **Diamètre**, **Rugosité**, Type d'arête.

*   **Encodage de la Cible (`y`) :**
    *   Pour chaque scénario : **vecteur binaire** de taille `nombre_total_tuyaux`. `y[i] = 1` si fuite sur tuyau `i`, 0 sinon. Vecteur nul si pas de fuite.

*   **Normalisation/Standardisation :** Toutes les caractéristiques numériques continues devront être **normalisées ou standardisées**, étape cruciale pour l'apprentissage profond [14, Chapitre 8].

<div style="page-break-after: always;"></div>

## 4. Conséquences : Construction des Données d'Entraînement et de Test

La construction des jeux de données suivra ce processus :

1.  **Chargement du Réseau :** Charger `Net3.inp` avec `wntr`.

2.  **Définition des Capteurs Virtuels :** Sélectionner les nœuds capteurs.

3.  **Simulation Baseline (sans fuite) :** Simuler le fonctionnement normal sur une période étendue. Extraire les séquences temporelles de pression (servira pour les exemples "Pas de Fuite" et comme référence `P_baseline` pour calculer les différences).

4.  **Simulation des Scénarios de Fuite :**
    *   Pour chaque tuyau `p` (ou un sous-ensemble) :
        *   Choisir aléatoirement `t_start`, durée `d`, sévérité `s`.
        *   Ajouter la fuite via `wn.add_leak(..., area=..., ...)` (modèle d'émetteur standard EPANET [15, Appendix D]).
        *   Simuler le scénario avec fuite (`P_abs`).
        *   **Calculer la caractéristique dynamique** (ex: `P_abs(t) - P_baseline(t)` en utilisant la simulation baseline correspondante).
        *   Extraire les séquences de features et créer le label `y`.
        *   Stocker features + label.
        *   Retirer la fuite.

5.  **Structuration & Division :** Organiser en format compatible GNN. Diviser en jeux d'entraînement, validation, test avec une **stratégie de séparation rigoureuse** pour éviter la fuite d'information et évaluer la généralisation [16, Chapitre 7]. **Le calcul des caractéristiques pour le jeu de test suivra exactement la même procédure que pour l'entraînement, y compris la génération et l'utilisation de la baseline spécifique à chaque cas de test.**

<div style="page-break-after: always;"></div>

## 5. Transition vers l'Application Réelle : Gestion de la Baseline et Adaptation du Modèle

L'entraînement sur simulation avec une **baseline parfaite (sans fuite)** est idéal pour le développement. Cependant, l'application sur un réseau réel impose de surmonter un défi majeur : l'absence de cette baseline idéale, car les réseaux opérationnels comportent des fuites préexistantes [11], des incertitudes et du bruit de mesure.

Appliquer directement un modèle entraîné sur des différences simulées idéales à des données réelles est donc risqué. Pour un déploiement efficace, des stratégies alternatives sont nécessaires pour estimer ou gérer la baseline opérationnelle :

1.  **Baseline via Modèle Hydraulique Calibré (Digital Twin) :**
    *   Construire et calibrer en continu un modèle numérique du réseau réel [17]. Ce modèle simule l'état hydraulique *attendu* en se basant sur une calibration effectuée sur une période d'observation passée.
    *   **Limitation Importante :** Le processus de calibration ajuste les paramètres du modèle pour reproduire au mieux les mesures réelles. Si des **fuites stables préexistaient pendant la calibration, leurs effets sont implicitement intégrés dans le modèle calibré**. Le modèle apprend à considérer cet état (potentiellement déjà fuyard) comme la "normale".
    *   **Détection :** En conséquence, le GNN utilisant la différence (`P_reel - P_sim_calibré`) comme entrée sera principalement apte à détecter les **nouvelles fuites** ou les **changements significatifs** survenant *après* la calibration. Il est **peu susceptible de signaler une fuite ancienne et stable** déjà "absorbée" par la calibration.
    *   **Cohérence :** L'exécution du modèle calibré pour générer `P_sim_calibré` doit être effectuée de manière cohérente en entraînement/fine-tuning et en test/application.

2.  **Baseline Apprise sur les Données Opérationnelles (Détection d'Anomalies/Changements) :**
    *   Apprendre le comportement "normal" **directement à partir du flux de données des capteurs réels** [18].
    *   Cette "normalité" apprise inclut **toutes les caractéristiques stables et récurrentes observées (y compris l'impact des fuites préexistantes stables)**.
    *   Cette approche excelle à détecter les **changements significatifs par rapport à cette norme apprise** (nouvelles fuites, aggravations, changements opérationnels).
    *   **Limitation Clé :** Comme l'approche du modèle calibré, elle est **peu susceptible de détecter une fuite ancienne, stable**, déjà intégrée à la baseline "normale" apprise.
    *   **Cohérence :** Lors du test/application, le système continue de comparer les nouvelles données à cette "normalité" apprise dynamiquement.

3.  **Adaptation par Apprentissage par Transfert (Transfer Learning) :**
    *   **Principe :** Utiliser les connaissances acquises sur simulation pour améliorer l'apprentissage sur le réseau réel [19].
    *   **Application :** Pré-entraîner le GNN sur simulation. Ensuite, **fine-tuner** ce modèle en utilisant les données réelles et la méthode de baseline choisie (Stratégie 1 ou 2).

**Implications Clés pour le Déploiement :**

*   **Nature des Détections :** Les deux principales approches de baseline opérationnelle sont **principalement orientées vers la détection de *changements* et de *nouvelles* anomalies**. L'identification des fuites stables préexistantes nécessite souvent des méthodes complémentaires.
*   **Cohérence Entraînement/Test :** La méthode de calcul des caractéristiques d'entrée (basée sur une baseline, qu'elle soit simulée, issue d'un modèle calibré ou apprise) **doit être rigoureusement la même** lors de l'entraînement/fine-tuning et lors du test/application réelle.
*   **Stratégie de Baseline Cruciale :** Le choix dépendra des objectifs, des ressources et des données.
*   **Adaptation Indispensable :** Le Transfer Learning et le fine-tuning [19] sont essentiels pour adapter le GNN aux spécificités du réel.

En conclusion, la transition vers le réel exige une stratégie claire pour la baseline opérationnelle, appliquée de façon cohérente, en reconnaissant ses implications sur la détection des fuites préexistantes vs. nouvelles, et une adaptation robuste du modèle.

<div style="page-break-after: always;"></div>

## 6. Bibliographie

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

[16] Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. Springer Series in Statistics.

[17] Brentan, B. M., Meirelles, G., Luvizotto Jr, E., & Izquierdo, J. (2018). Hybrid calibration approach for water distribution network models. *Journal of Water Resources Planning and Management*, 144(8), 04018045.

[18] Ye, G., & Yao, L. (2021). A review on data-driven anomaly detection for water distribution systems. *Water Supply*, 21(7), 3175-3190.

[19] Pan, S. J., & Yang, Q. (2009). A survey on transfer learning. *IEEE Transactions on knowledge and data engineering*, 22(10), 1345-1359.

---

