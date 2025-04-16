import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, NNConv, LayerNorm, BatchNorm

class WDNLeakGNN(nn.Module):
    """
    Réseau de Neurones sur Graphes (GNN) pour la localisation de fuites sur les arêtes (tuyaux) d'un réseau de distribution d'eau.
    - Utilise des couches GINEConv ou NNConv pour intégrer les caractéristiques des arêtes.
    - Prédit la probabilité de fuite pour chaque arête via un MLP appliqué à la concaténation des embeddings des nœuds et des features d'arête.
    """

    def __init__(
        self,
        node_in_dim,
        edge_in_dim,
        hidden_dim=128,
        num_layers=3,
        gnn_type='gine',  # 'gine' ou 'nnconv'
        mlp_hidden_dim=128,
        dropout=0.3,
        norm_type='layer',  # 'layer' ou 'batch'
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.norm_type = norm_type

        # Encodeurs linéaires pour projeter les features des nœuds et arêtes dans l'espace caché
        self.node_encoder = nn.Linear(node_in_dim, hidden_dim)
        self.edge_encoder = nn.Linear(edge_in_dim, hidden_dim)

        # Liste des couches GNN et des couches de normalisation
        self.gnn_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        for i in range(num_layers):
            if gnn_type == 'gine':
                # GINEConv : intègre les edge_attr via un petit MLP
                nn_edge = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
                conv = GINEConv(nn_edge)
            elif gnn_type == 'nnconv':
                # NNConv : génère une matrice de poids à partir des edge_attr
                nn_edge = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * hidden_dim)
                )
                conv = NNConv(hidden_dim, hidden_dim, nn_edge, aggr='add')
            else:
                raise ValueError(f"Unknown gnn_type: {gnn_type}")
            self.gnn_layers.append(conv)
            if norm_type == 'layer':
                self.norm_layers.append(LayerNorm(hidden_dim))
            else:
                self.norm_layers.append(BatchNorm(hidden_dim))

        # Classifieur MLP pour la prédiction au niveau des arêtes
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        """
        x: [num_nodes, node_in_dim]
        edge_index: [2, num_edges]
        edge_attr: [num_edges, edge_in_dim]
        Retourne : [num_edges] probabilités de fuite (sigmoïde)
        """
        # Encodage initial des features des nœuds et arêtes
        x = self.node_encoder(x)
        if torch.isnan(x).any(): print("[DEBUG Model] NaN after node_encoder")
        edge_attr_encoded = self.edge_encoder(edge_attr) # Use new variable name
        if torch.isnan(edge_attr_encoded).any(): print("[DEBUG Model] NaN after edge_encoder")

        # Passage à travers les couches GNN avec normalisation, activation et dropout
        for i, (conv, norm) in enumerate(zip(self.gnn_layers, self.norm_layers)):
            x_before_conv = x
            # Pass encoded edge_attr to GINEConv/NNConv
            x = conv(x, edge_index, edge_attr_encoded)
            if torch.isnan(x).any(): print(f"[DEBUG Model] NaN after GNN layer {i} conv")
            x = norm(x)
            if torch.isnan(x).any(): print(f"[DEBUG Model] NaN after GNN layer {i} norm")
            x = F.relu(x)
            if torch.isnan(x).any(): print(f"[DEBUG Model] NaN after GNN layer {i} relu")
            x = F.dropout(x, p=self.dropout, training=self.training)
            if torch.isnan(x).any(): print(f"[DEBUG Model] NaN after GNN layer {i} dropout")

        # Pour chaque arête, concatène les embeddings des deux nœuds et les features d'arête
        src, dst = edge_index
        h_src = x[src]
        h_dst = x[dst]
        # Use encoded edge_attr for MLP input
        edge_inputs = torch.cat([h_src, h_dst, edge_attr_encoded], dim=1)
        if torch.isnan(edge_inputs).any(): print("[DEBUG Model] NaN in edge_inputs before MLP")
        logits = self.edge_mlp(edge_inputs).squeeze(-1)
        if torch.isnan(logits).any(): print("[DEBUG Model] NaN after final edge_mlp")
        return logits # Return raw logits for numerical stability with BCEWithLogitsLoss