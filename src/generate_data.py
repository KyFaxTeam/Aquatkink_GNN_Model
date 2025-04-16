import argparse
import os
import numpy as np
import h5py
import wntr
from tqdm import tqdm

# =========================
# Configuration & Arguments
# =========================

def parse_args():
    parser = argparse.ArgumentParser(description="Génération de scénarios hydrauliques Net3 pour GNN (fuite et baseline)")
    parser.add_argument('--inp', type=str, default='data/Net3.inp', help='Chemin du fichier INP EPANET')
    parser.add_argument('--output_dir', type=str, default='data/raw/', help='Répertoire de sortie pour les fichiers HDF5')
    parser.add_argument('--n_scenarios', type=int, default=1000, help='Nombre de scénarios de fuite à générer')
    parser.add_argument('--n_no_leak', type=int, default=200, help='Nombre de scénarios sans fuite (baseline)')
    parser.add_argument('--sensor_pct', type=float, default=0.15, help='Pourcentage de jonctions équipées de capteurs virtuels')
    parser.add_argument('--sim_duration', type=float, default=24.0, help='Durée de simulation en heures')
    parser.add_argument('--seed', type=int, default=42, help='Seed aléatoire pour la reproductibilité')
    parser.add_argument('--save_flows', action='store_true', help='Sauvegarder aussi les débits dans les fichiers HDF5')
    return parser.parse_args()

# =========================
# Utilitaires de sélection
# =========================

def select_sensor_nodes(wn, pct, seed=42):
    np.random.seed(seed)
    junctions = [j for j in wn.junction_name_list]
    n_sensors = max(1, int(len(junctions) * pct))
    return sorted(np.random.choice(junctions, n_sensors, replace=False))

# =========================
# Simulation de scénarios
# =========================

def simulate_scenario(wn, leak_pipe=None, leak_start=None, leak_duration=None, leak_severity=None, sensor_nodes=None, sim_duration=24.0, save_flows=False):
    # Ajoute une fuite si spécifié
    # Ajoute une fuite si spécifié
    if leak_pipe is not None:
        pipe = wn.get_link(leak_pipe)
        # =========================
        # Ajout de la fuite :
        # On tente d'ajouter la fuite sur le noeud de début du tuyau si c'est une jonction.
        # Sinon, on essaie le noeud de fin. Si aucun n'est une jonction, on ignore ce tuyau.
        # Ceci évite les erreurs avec les réservoirs ou bassins qui ne supportent pas add_leak.
        # =========================
        start_node = wn.get_node(pipe.start_node_name)
        end_node = wn.get_node(pipe.end_node_name)
        if start_node.node_type == 'Junction':
            start_node.add_leak(wn, area=leak_severity, start_time=leak_start, end_time=leak_start+leak_duration)
        elif end_node.node_type == 'Junction':
            end_node.add_leak(wn, area=leak_severity, start_time=leak_start, end_time=leak_start+leak_duration)
        else:
            # Aucun noeud valide pour la fuite sur ce tuyau
            pass

    # Simulation hydraulique
    sim = wntr.sim.EpanetSimulator(wn)
    results = sim.run_sim()
    pressures = results.node['pressure'].loc[:, sensor_nodes].values.T  # [n_sensors, n_timesteps]
    flows = results.link['flowrate'].values.T if save_flows else None

    # Pas de nettoyage nécessaire : le modèle est jeté après chaque simulation

    return pressures, flows

# =========================
# Sauvegarde HDF5
# =========================

def save_hdf5(filepath, data_dict):
    with h5py.File(filepath, 'w') as f:
        for group, group_data in data_dict.items():
            grp = f.create_group(group)
            for key, value in group_data.items():
                if value is not None:
                    grp.create_dataset(key, data=value)

# =========================
# Main
# =========================

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Charger le réseau
    wn = wntr.network.WaterNetworkModel(args.inp)

    # Sélection des capteurs virtuels (fixe pour tous les scénarios)
    sensor_nodes = select_sensor_nodes(wn, args.sensor_pct, args.seed)

    # Récupération des attributs statiques
    pipe_ids = wn.pipe_name_list
    node_elevations = np.array([wn.get_node(n).elevation for n in sensor_nodes])
    pipe_lengths = np.array([wn.get_link(pid).length for pid in pipe_ids])
    pipe_diameters = np.array([wn.get_link(pid).diameter for pid in pipe_ids])
    pipe_roughness = np.array([wn.get_link(pid).roughness for pid in pipe_ids])

    # Génération des scénarios
    for i in tqdm(range(args.n_scenarios), desc="Scénarios de fuite"):
        # Définir aléatoirement les paramètres de fuite
        leak_pipe = np.random.choice(pipe_ids)
        leak_start = np.random.uniform(0, args.sim_duration * 3600 * 0.8)  # en secondes
        leak_duration = np.random.uniform(1*3600, 4*3600)  # 1h à 4h
        leak_severity = np.random.uniform(0.001, 0.01)  # m2, à ajuster selon le réseau

        # Baseline
        wn_base = wntr.network.WaterNetworkModel(args.inp)
        baseline_pressures, baseline_flows = simulate_scenario(
            wn_base, leak_pipe=None, sensor_nodes=sensor_nodes, sim_duration=args.sim_duration, save_flows=args.save_flows
        )

        # Scénario fuite
        wn_leak = wntr.network.WaterNetworkModel(args.inp)
        leak_pressures, leak_flows = simulate_scenario(
            wn_leak, leak_pipe=leak_pipe, leak_start=leak_start, leak_duration=leak_duration, leak_severity=leak_severity,
            sensor_nodes=sensor_nodes, sim_duration=args.sim_duration, save_flows=args.save_flows
        )

        # Label
        leak_vector = np.zeros(len(pipe_ids), dtype=np.int8)
        leak_vector[pipe_ids.index(leak_pipe)] = 1

        # Sauvegarde
        data_dict = {
            "leak_results": {
                "pressures": leak_pressures,
                "flows": leak_flows if args.save_flows else None
            },
            "baseline_results": {
                "pressures": baseline_pressures,
                "flows": baseline_flows if args.save_flows else None
            },
            "static": {
                "node_ids": np.array(sensor_nodes, dtype='S'),
                "node_elevations": node_elevations,
                "pipe_ids": np.array(pipe_ids, dtype='S'),
                "pipe_lengths": pipe_lengths,
                "pipe_diameters": pipe_diameters,
                "pipe_roughness": pipe_roughness
            },
            "label": {
                "leak_pipe_id": np.string_(leak_pipe),
                "leak_vector": leak_vector
            },
            "metadata": {
                "scenario_id": i,
                "type": "leak",
                "leak_start_time": leak_start,
                "leak_duration": leak_duration,
                "leak_severity": leak_severity,
                "sensor_node_ids": np.array(sensor_nodes, dtype='S'),
                "baseline_scenario_id": i  # même index pour baseline associée
            }
        }
        save_hdf5(os.path.join(args.output_dir, f"scenario_{i}.h5"), data_dict)

    # Génération des scénarios sans fuite (baselines purs)
    for i in tqdm(range(args.n_no_leak), desc="Scénarios sans fuite"):
        wn_base = wntr.network.WaterNetworkModel(args.inp)
        baseline_pressures, baseline_flows = simulate_scenario(
            wn_base, leak_pipe=None, sensor_nodes=sensor_nodes, sim_duration=args.sim_duration, save_flows=args.save_flows
        )
        leak_vector = np.zeros(len(pipe_ids), dtype=np.int8)
        data_dict = {
            "leak_results": {
                "pressures": baseline_pressures,
                "flows": baseline_flows if args.save_flows else None
            },
            "baseline_results": {
                "pressures": baseline_pressures,
                "flows": baseline_flows if args.save_flows else None
            },
            "static": {
                "node_ids": np.array(sensor_nodes, dtype='S'),
                "node_elevations": node_elevations,
                "pipe_ids": np.array(pipe_ids, dtype='S'),
                "pipe_lengths": pipe_lengths,
                "pipe_diameters": pipe_diameters,
                "pipe_roughness": pipe_roughness
            },
            "label": {
                "leak_pipe_id": np.string_("None"),
                "leak_vector": leak_vector
            },
            "metadata": {
                "scenario_id": i + args.n_scenarios,
                "type": "baseline",
                "leak_start_time": None,
                "leak_duration": None,
                "leak_severity": None,
                "sensor_node_ids": np.array(sensor_nodes, dtype='S'),
                "baseline_scenario_id": i + args.n_scenarios
            }
        }
        save_hdf5(os.path.join(args.output_dir, f"scenario_{i + args.n_scenarios}.h5"), data_dict)

if __name__ == "__main__":
    main()