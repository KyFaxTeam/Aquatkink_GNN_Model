import wntr
import pandas as pd
import numpy as np
import os
import random
import glob
import re

def sanitize_network_name(inp_filename: str) -> str:
    """
    Convertit le nom du fichier .inp en nom de dossier valide
    Exemple: 'Net3_EPANET-EXAMPLE.inp' -> 'Net3_EPANET-EXAMPLE'
    """
    # Enlever l'extension .inp
    base_name = os.path.splitext(os.path.basename(inp_filename))[0]
    # Remplacer les caractères non valides par des underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', base_name)
    return sanitized

def create_network_subdirectory(base_dir: str, network_name: str) -> str:
    """
    Crée un sous-dossier pour le réseau spécifié
    Retourne le chemin complet du sous-dossier
    """
    network_dir = os.path.join(base_dir, network_name)
    os.makedirs(network_dir, exist_ok=True)
    return network_dir

def get_network_output_path(base_dir: str, network_name: str, filename: str) -> str:
    """
    Construit le chemin complet pour un fichier de sortie
    """
    return os.path.join(base_dir, network_name, filename)

def simulate_network(wn, duration_hours, hydraulic_timestep_min, leak_pipe_name=None, leak_area=0.05, leak_start_hour=12):
    """
    Simulates the water network, optionally adding a leak.

    Args:
        wn: wntr WaterNetworkModel object.
        duration_hours (int): Total simulation duration in hours.
        hydraulic_timestep_min (int): Hydraulic timestep in minutes.
        leak_pipe_name (str, optional): Name of the pipe where the leak occurs. Defaults to None (no leak).
        leak_area (float, optional): Area of the leak in m^2. Defaults to 0.05.
        leak_start_hour (int, optional): Hour when the leak starts. Defaults to 12.

    Returns:
        wntr Results object containing simulation results, or None if simulation fails.
    """
    # Reset any previous modifications
    wn.reset_initial_values()
    for pipe_name, pipe in wn.pipes():
         if hasattr(pipe, 'leak_node'):
             wn.remove_node(pipe.leak_node.name) # Clean up potential previous leak nodes

    # Set simulation options
    wn.options.time.duration = duration_hours * 3600  # seconds
    wn.options.time.hydraulic_timestep = hydraulic_timestep_min * 60  # seconds
    wn.options.time.report_timestep = hydraulic_timestep_min * 60 # seconds

    sim = None # Initialize sim to None

    if leak_pipe_name:
        try:
            # Add leak
            pipe = wn.get_link(leak_pipe_name)
            leak_start_time_sec = leak_start_hour * 3600
            leak_end_time_sec = duration_hours * 3600 # Leak continues till end
            # --- Workaround: Add leak to the start node of the pipe ---
            # Original approach wn.split_pipe(...) failed as method may not exist
            leak_node = pipe.start_node # Get the start node of the pipe
            leak_node_name = leak_node.name
            # Add the leak properties to the start node
            leak_node.add_leak(wn, area=leak_area, start_time=leak_start_time_sec, end_time=leak_end_time_sec)
            print(f"Added leak to START NODE {leak_node_name} (for pipe {leak_pipe_name}) starting at hour {leak_start_hour}")
            # --- End Workaround ---

        except Exception as e:
            print(f"Error adding leak to pipe {leak_pipe_name}: {e}")
            return None # Return None if leak addition fails

    try:
        # Run simulation
        sim = wntr.sim.EpanetSimulator(wn)
        results = sim.run_sim()
        print(f"Simulation successful for scenario: {'Leak at ' + leak_pipe_name if leak_pipe_name else 'Baseline'}")
        return results
    except Exception as e:
        print(f"Error running simulation for scenario {'Leak at ' + leak_pipe_name if leak_pipe_name else 'Baseline'}: {e}")
        # Attempt to clean up the added leak node if simulation failed after adding leak
        # Cleanup might be less critical now as we modify an existing node, not add/split.
        # However, the added leak property persists on the node if simulation fails mid-way.
        # Reloading wn_base before each scenario (already done) is the safer approach.
        if leak_pipe_name and 'leak_node' in locals():
             print(f"Note: Simulation failed after adding leak to node {leak_node.name}. Network state might be altered for this wn copy.")
        return None # Return None if simulation fails

def generate_leak_data(inp_file_path, output_dir='simulated_data', network_name=None, num_leak_scenarios=10, duration_hours=72, hydraulic_timestep_min=60):
    """
    Generates simulated leak data for a water network.

    Args:
        inp_file_path (str): Path to the EPANET .inp file.
        output_dir (str): Base directory to save the generated data. Defaults to 'simulated_data'.
        network_name (str, optional): Name of the network subdirectory. If None, extracted from inp_file_path.
        num_leak_scenarios (int): Number of leak scenarios to simulate (random pipes). Defaults to 10.
        duration_hours (int): Simulation duration in hours. Defaults to 72.
        hydraulic_timestep_min (int): Hydraulic timestep in minutes. Defaults to 60.
    """
    if network_name is None:
        network_name = sanitize_network_name(inp_file_path)
    
    print(f"Loading network model from: {inp_file_path}")
    try:
        wn_base = wntr.network.WaterNetworkModel(inp_file_path)
    except Exception as e:
        print(f"Error loading network model: {e}")
        return

    # Create network-specific output directory
    network_dir = create_network_subdirectory(output_dir, network_name)
    print(f"Using output directory: {network_dir}")

    # --- 1. Run Baseline Simulation (No Leak) ---
    print("\n--- Running Baseline Simulation ---")
    # Need a fresh copy for baseline as simulate_network modifies wn
    wn_baseline_copy = wntr.network.WaterNetworkModel(inp_file_path)
    baseline_results = simulate_network(wn_baseline_copy, duration_hours, hydraulic_timestep_min)
    if baseline_results is None:
        print("Baseline simulation failed. Cannot proceed.")
        return

    baseline_pressures = baseline_results.node['pressure']
    baseline_flows = baseline_results.link['flowrate']

    # Save baseline data
    baseline_pressures.to_csv(get_network_output_path(output_dir, network_name, 'baseline_pressures.csv'))
    baseline_flows.to_csv(get_network_output_path(output_dir, network_name, 'baseline_flows.csv'))

    # Extract node features (pressure at hour 13) for baseline
    time_index_for_features = 13 * 3600 # Time in seconds for hour 13
    try:
        baseline_node_features = baseline_pressures.loc[time_index_for_features].values.reshape(-1, 1) # Shape [num_nodes, 1]
        baseline_node_features_filename = get_network_output_path(output_dir, network_name, 'baseline_node_features.npy')
        np.save(baseline_node_features_filename, baseline_node_features)
        print(f"Saved baseline node features (pressure at hour 13) to {baseline_node_features_filename}")
    except KeyError:
         print(f"Error: Time index {time_index_for_features}s not found in baseline results. Check simulation duration/timestep.")
         print("Skipping feature extraction.")
         baseline_node_features_filename = None # Indicate failure

    print(f"Saved baseline data (pressures, flows, features) to {output_dir}")


    # --- Extract Static Graph Info ---
    print("\n--- Extracting Static Graph Information ---")
    node_names = wn_base.node_name_list
    pipe_names = wn_base.pipe_name_list
    num_nodes = wn_base.num_nodes
    num_pipes = wn_base.num_pipes

    # Create mapping from node names to indices (0 to num_nodes-1)
    
    node_name_to_index = {name: i for i, name in enumerate(node_names)}

    # Create edge_index [2, num_pipes]
    start_nodes = []
    end_nodes = []
    edge_features_list = [] # To store static features like length, diameter
    pipe_name_to_index = {} # Map pipe name to its index in the edge list

    for i, pipe_name in enumerate(pipe_names):
        pipe = wn_base.get_link(pipe_name)
        start_node_idx = node_name_to_index[pipe.start_node_name]
        end_node_idx = node_name_to_index[pipe.end_node_name]
        start_nodes.append(start_node_idx)
        end_nodes.append(end_node_idx)
        edge_features_list.append([pipe.length, pipe.diameter])
        pipe_name_to_index[pipe_name] = i

    edge_index = np.array([start_nodes, end_nodes], dtype=np.int64)
    static_edge_features = np.array(edge_features_list, dtype=np.float32)

    # Save static graph info
    np.save(get_network_output_path(output_dir, network_name, 'edge_index.npy'), edge_index)
    np.save(get_network_output_path(output_dir, network_name, 'static_edge_features.npy'), static_edge_features)
    # Save mappings (optional but helpful)
    pd.Series(node_name_to_index).to_csv(get_network_output_path(output_dir, network_name, 'node_name_to_index.csv'))
    pd.Series(pipe_name_to_index).to_csv(get_network_output_path(output_dir, network_name, 'pipe_name_to_index.csv'))
    print(f"Saved edge_index, static_edge_features, and mappings to {output_dir}")


    # --- 2. Run Leak Scenarios ---
    print(f"\n--- Running {num_leak_scenarios} Leak Scenarios ---")
    if num_leak_scenarios > num_pipes:
        print(f"Warning: Requested {num_leak_scenarios} scenarios, but only {num_pipes} pipes available. Simulating leaks on all pipes.")
        num_leak_scenarios = num_pipes
        leak_pipes_to_simulate = pipe_names
    else:
        leak_pipes_to_simulate = random.sample(pipe_names, num_leak_scenarios) # Sample random pipes

    scenario_metadata = [] # To store metadata including label file paths

    for i, leak_pipe in enumerate(leak_pipes_to_simulate):
        print(f"\n--- Scenario {i+1}/{num_leak_scenarios}: Leak on Pipe {leak_pipe} ---")
        # Use a fresh copy of the network model for each scenario to avoid interference
        wn_scenario_copy = wntr.network.WaterNetworkModel(inp_file_path)
        leak_results = simulate_network(wn_scenario_copy, duration_hours, hydraulic_timestep_min, leak_pipe_name=leak_pipe)

        if leak_results:
            leak_pressures = leak_results.node['pressure']
            leak_flows = leak_results.link['flowrate']

            # Example: Store pressure difference as a feature
            # More sophisticated feature engineering would be needed for a real model
            pressure_diff = leak_pressures - baseline_pressures
            # Flatten or aggregate data per scenario (e.g., mean pressure diff)
            # For now, just save the full data for the scenario
            scenario_pressure_filename = get_network_output_path(output_dir, network_name, f'leak_pipe_{leak_pipe}_pressures.csv')
            scenario_flow_filename = get_network_output_path(output_dir, network_name, f'leak_pipe_{leak_pipe}_flows.csv')
            leak_pressures.to_csv(scenario_pressure_filename)
            leak_flows.to_csv(scenario_flow_filename)
            print(f"Saved leak scenario pressure/flow data for pipe {leak_pipe}")

            # Create edge labels for this scenario
            edge_labels = np.zeros(num_pipes, dtype=np.int64)
            leaked_pipe_index = pipe_name_to_index[leak_pipe]
            edge_labels[leaked_pipe_index] = 1 # Mark the leaking pipe

            # Save edge labels
            label_filename = get_network_output_path(output_dir, network_name, f'leak_pipe_{leak_pipe}_labels.npy')
            np.save(label_filename, edge_labels)
            print(f"Saved edge labels to {label_filename}")

            # Store metadata
            # Extract node features (pressure at hour 13) for this leak scenario
            try:
                leak_node_features = leak_pressures.loc[time_index_for_features].values.reshape(-1, 1) # Shape [num_nodes, 1]
                node_features_filename = get_network_output_path(output_dir, network_name, f'leak_pipe_{leak_pipe}_node_features.npy')
                np.save(node_features_filename, leak_node_features)
                print(f"Saved node features to {node_features_filename}")
            except KeyError:
                 print(f"Error: Time index {time_index_for_features}s not found in leak results for pipe {leak_pipe}.")
                 print("Skipping feature extraction for this scenario.")
                 node_features_filename = None # Indicate failure


            # Store metadata
            scenario_metadata.append({'scenario_id': f'leak_{leak_pipe}',
                                      'leaked_pipe_name': leak_pipe,
                                      'leaked_pipe_index': leaked_pipe_index,
                                      'pressure_file': scenario_pressure_filename, # Raw pressures
                                      'flow_file': scenario_flow_filename,       # Raw flows
                                      'node_features_file': node_features_filename, # Features at hour 13
                                      'label_file': label_filename})             # Leak labels
        else:
            print(f"Skipping scenario for pipe {leak_pipe} due to simulation error.")

    # Save metadata about all generated leak scenarios
    scenario_metadata_df = pd.DataFrame(scenario_metadata)
    metadata_path = get_network_output_path(output_dir, network_name, 'scenarios_metadata.csv')
    scenario_metadata_df.to_csv(metadata_path, index=False)
    print(f"\nSaved scenario metadata to {metadata_path}")

    print("\n--- Data Generation Complete ---")


if __name__ == "__main__":
    try:
        # Define parameters communs
        OUTPUT_DIRECTORY = 'simulated_data'
        NUM_SCENARIOS = 200  # Number of random pipes to simulate leaks on
        SIM_DURATION_HOURS = 24 * 3  # Simulate for 3 days
        HYDRAULIC_TIMESTEP_MIN = 60  # Simulate every hour

        # Liste tous les fichiers .inp dans le dossier data
        inp_files = glob.glob(os.path.join('data', '*.inp'))
        
        if not inp_files:
            raise FileNotFoundError("Aucun fichier .inp trouvé dans le dossier 'data'")
        
        print(f"Fichiers .inp trouvés: {len(inp_files)}")
        for inp_file in inp_files:
            print(f"\n{'='*80}")
            print(f"Traitement du fichier: {inp_file}")
            print(f"{'='*80}")
            
            try:
                # Extraire le nom du réseau à partir du fichier
                network_name = sanitize_network_name(inp_file)
                
                # Générer les données pour ce réseau
                generate_leak_data(
                    inp_file_path=inp_file,
                    output_dir=OUTPUT_DIRECTORY,
                    network_name=network_name,
                    num_leak_scenarios=NUM_SCENARIOS,
                    duration_hours=SIM_DURATION_HOURS,
                    hydraulic_timestep_min=HYDRAULIC_TIMESTEP_MIN
                )
                print(f"Traitement terminé pour {network_name}")
                
            except Exception as e:
                print(f"Erreur lors du traitement de {inp_file}: {e}")
                print("Passage au fichier suivant...\n")
                continue

        print("\nTraitement de tous les fichiers terminé!")

    except FileNotFoundError as fnf_error:
        print(f"Erreur: {fnf_error}")
    except Exception as e:
        print(f"Une erreur inattendue s'est produite: {e}")