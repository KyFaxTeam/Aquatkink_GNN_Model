import os
import glob
from src import paths
import h5py
import wntr
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

# Use centralized paths

def normalize_features(x, mean=None, std=None, eps=1e-8):
    if mean is None:
        mean = x.mean(axis=0, keepdims=True)
    if std is None:
        std = x.std(axis=0, keepdims=True)
    return (x - mean) / (std + eps), mean, std

def process_scenario(h5_path, norm_stats=None):
    """
    Processes a single HDF5 scenario file to create a PyG Data object
    representing the full WDN topology with sensor data as node features.

    Args:
        h5_path (str): Path to the HDF5 scenario file.
        norm_stats (dict, optional): Dictionary containing 'mean' and 'std'
                                     for global node feature normalization.
                                     Defaults to None (per-scenario normalization).

    Returns:
        torch_geometric.data.Data: PyG Data object for the scenario, or None if an error occurs.
    """
    try:
        with h5py.File(h5_path, "r") as f:
            # --- Load Sensor Data from HDF5 ---
            try:
                leak_pressures = f["leak_results/pressures"][()]      # [n_sensors, n_timesteps]
                baseline_pressures = f["baseline_results/pressures"][()] # [n_sensors, n_timesteps]
                pressure_diff = leak_pressures - baseline_pressures     # [n_sensors, n_timesteps]

                node_elevations_h5 = f["static/node_elevations"][()]  # [n_sensors]
                node_elevations_h5 = node_elevations_h5[:, None]      # [n_sensors, 1]

                # Combine dynamic pressure diff and static elevation from HDF5 for sensors
                sensor_features = np.concatenate([pressure_diff, node_elevations_h5], axis=1) # [n_sensors, n_timesteps+1]

                sensor_node_ids_bytes = f["static/node_ids"][()]
                sensor_node_ids = [nid.decode('utf-8') for nid in sensor_node_ids_bytes]

                # Load leak information (assuming binary vector aligned with HDF5 pipes)
                y_original_h5 = f["label/leak_vector"][()]
                pipe_ids_h5_bytes = f["static/pipe_ids"][()]
                pipe_ids_h5 = [pid.decode('utf-8') for pid in pipe_ids_h5_bytes]

                if len(y_original_h5) != len(pipe_ids_h5):
                    raise ValueError(f"HDF5 mismatch: leak_vector ({len(y_original_h5)}) vs pipe_ids ({len(pipe_ids_h5)})")

            except KeyError as e:
                print(f"Error: Missing key {e} in HDF5 file {h5_path}.")
                return None
            except ValueError as e:
                 print(f"Error processing HDF5 data in {h5_path}: {e}")
                 return None

            # --- Load Full Network Topology from INP File ---
            # Use central path from src/paths.py
            inp_file_path = os.path.join(os.path.dirname(paths.RAW_DATA_DIR), 'Net3.inp')
            try:
                wn = wntr.network.WaterNetworkModel(inp_file_path)
            except FileNotFoundError:
                print(f"Error: INP file not found at {inp_file_path}. Cannot process scenario {h5_path}.")
                return None
            except Exception as e:
                print(f"Error loading INP file {inp_file_path}: {e}")
                return None

            # --- Build Graph based on Full WNTR Topology ---

            # 1. Map *All* Relevant Nodes from WNTR
            # Include junctions, reservoirs, and tanks
            all_node_names = list(wn.junction_name_list) + list(wn.reservoir_name_list) + list(wn.tank_name_list)
            if not all_node_names:
                print(f"Error: No nodes found in WNTR model {inp_file_path}.")
                return None
            node_name_to_index = {name: i for i, name in enumerate(all_node_names)}
            num_all_nodes = len(all_node_names)

            # 2. Build *Complete* edge_index and edge_attr from WNTR Links
            start_nodes_idx = []
            end_nodes_idx = []
            edge_attrs_list = []
            link_name_to_edge_index = {} # Map link name to its index in the final edge list (0 to num_links-1)
            link_counter = 0

            # Define feature extraction functions for different link types
            def get_pipe_features(pipe):
                length = getattr(pipe, 'length', 0.0)
                diameter = getattr(pipe, 'diameter', 0.0)
                roughness = getattr(pipe, 'roughness', 0.0)
                # Add a type identifier (e.g., 0 for pipe)
                return [length, diameter, roughness, 0.0]

            def get_pump_features(pump):
                # Placeholder features for pumps (e.g., type identifier 1)
                # Could add pump curve info, speed, etc. if available and needed
                return [-1.0, -1.0, -1.0, 1.0] # Example

            def get_valve_features(valve):
                # Placeholder features for valves (e.g., type identifier 2)
                # Could add diameter, type, setting, etc.
                return [-2.0, -2.0, -2.0, 2.0] # Example

            link_processing_map = {
                'pipes': (wn.pipes, get_pipe_features),
                'pumps': (wn.pumps, get_pump_features),
                'valves': (wn.valves, get_valve_features),
            }

            for link_type, (link_iterator, feature_func) in link_processing_map.items():
                for link_name, link_obj in link_iterator():
                    try:
                        # Ensure both start and end nodes exist in our mapping
                        if link_obj.start_node_name not in node_name_to_index or \
                           link_obj.end_node_name not in node_name_to_index:
                           # print(f"Warning: Skipping {link_type[:-1]} {link_name} due to missing node(s) in mapping.")
                           continue

                        start_node_idx = node_name_to_index[link_obj.start_node_name]
                        end_node_idx = node_name_to_index[link_obj.end_node_name]

                        # Add edge for both directions (i->j and j->i)
                        start_nodes_idx.extend([start_node_idx, end_node_idx])
                        end_nodes_idx.extend([end_node_idx, start_node_idx])

                        # Extract and add features for both directions
                        features = feature_func(link_obj)
                        edge_attrs_list.extend([features, features])

                        # Map the link name to its unique index (used for labels)
                        # Only increment counter once per unique link
                        link_name_to_edge_index[link_name] = link_counter
                        link_counter += 1

                    except Exception as e:
                        print(f"Warning: Error processing {link_type[:-1]} {link_name}: {e}. Skipping link.")
                        continue

            if not edge_attrs_list:
                print(f"Warning: No edges were created for scenario {h5_path}. Check INP file and node mapping.")
                # Create empty tensors to avoid downstream errors
                edge_index = torch.empty((2, 0), dtype=torch.long)
                # Use feature size if available, otherwise default (e.g., 4)
                feature_size = len(features) if 'features' in locals() and features else 4
                edge_attr = torch.empty((0, feature_size), dtype=torch.float)
            else:
                edge_index = torch.tensor([start_nodes_idx, end_nodes_idx], dtype=torch.long)
                edge_attr = torch.tensor(edge_attrs_list, dtype=torch.float)

            num_all_links = link_counter # Total number of unique links processed

            # 3. Build *Complete* Node Features `x`
            num_node_features = sensor_features.shape[1] # Features: pressure diff + h5_elevation
            # Initialize x with zeros (or other imputation) for all nodes
            x = torch.zeros((num_all_nodes, num_node_features), dtype=torch.float)

            # Fill in features for sensor nodes using the complete node mapping
            sensors_found_in_model = 0
            for i, sensor_id in enumerate(sensor_node_ids):
                if sensor_id in node_name_to_index:
                    node_idx = node_name_to_index[sensor_id]
                    x[node_idx, :] = torch.tensor(sensor_features[i, :], dtype=torch.float)
                    sensors_found_in_model += 1
                else:
                    # This warning is expected if HDF5 nodes are just a subset
                    # print(f"Info: Sensor node ID {sensor_id} from HDF5 not found in WNTR model nodes.")
                    pass
            if sensors_found_in_model == 0 and len(sensor_node_ids) > 0:
                 print(f"Warning: CRITICAL - None of the {len(sensor_node_ids)} sensor nodes from HDF5 were found in the WNTR model for {h5_path}.")


            # Optional: Add static features from WNTR for *all* nodes (e.g., elevation)
            # Example: Add WNTR elevation as an *additional* feature column
            # elevations_wn = torch.zeros((num_all_nodes, 1), dtype=torch.float)
            # for node_name, idx in node_name_to_index.items():
            #     try:
            #         node = wn.get_node(node_name)
            #         elevations_wn[idx, 0] = getattr(node, 'elevation', 0.0) # Use getattr for safety
            #     except Exception as e:
            #         print(f"Warning: Could not get elevation for node {node_name}: {e}")
            #         elevations_wn[idx, 0] = 0.0 # Default value
            # x = torch.cat([x, elevations_wn], dim=1)


            # 4. Build *Complete* Label Vector `y` (per unique link)
            # Find the name of the leaking pipe from HDF5 data
            leaking_link_name = None
            leak_indices_h5 = np.where(y_original_h5 == 1)[0]

            if len(leak_indices_h5) == 1:
                leaking_link_name = pipe_ids_h5[leak_indices_h5[0]] # Assumes leak_vector is only for pipes
            elif len(leak_indices_h5) > 1:
                # Handle multiple leaks if necessary, here we take the first one
                leaking_link_name = pipe_ids_h5[leak_indices_h5[0]]
                print(f"Warning: Multiple leaks ({len(leak_indices_h5)}) indicated in HDF5 for {h5_path}. Using first: {leaking_link_name}")
            # If len is 0, it's a no-leak scenario, leaking_link_name remains None.

            # Initialize y for all directed edges (size = 2 * num_all_links)
            num_edges = edge_index.shape[1] # Total number of directed edges
            y = torch.zeros(num_edges, dtype=torch.float)

            if leaking_link_name is not None:
                if leaking_link_name in link_name_to_edge_index:
                    # Find the base index corresponding to this unique link
                    unique_link_idx = link_name_to_edge_index[leaking_link_name]
                    # Calculate the indices for the two directed edges
                    edge_idx1 = 2 * unique_link_idx
                    edge_idx2 = edge_idx1 + 1

                    # Set the label for both directed edges if indices are valid
                    if edge_idx1 < num_edges and edge_idx2 < num_edges:
                         y[edge_idx1] = 1.0
                         y[edge_idx2] = 1.0
                    else:
                         # This indicates an inconsistency in edge/link counting
                         print(f"Error: Calculated edge indices ({edge_idx1}, {edge_idx2}) are out of bounds for y ({num_edges}) for link {leaking_link_name} in {h5_path}.")
                else:
                    # This might happen if the leaking pipe in HDF5 is not a pipe recognized by WNTR
                    # (e.g., different naming conventions, or it's actually a pump/valve listed as pipe in HDF5)
                    print(f"Warning: Leaking link '{leaking_link_name}' from HDF5 not found among the processed graph links for scenario {h5_path}.")


            # --- Normalization (Applied to the full 'x' tensor) ---
            if norm_stats is not None:
                # Apply pre-computed global stats if available
                try:
                    # Ensure stats are numpy arrays before converting to tensor
                    mean_np = np.array(norm_stats["mean"])
                    std_np = np.array(norm_stats["std"])
                    mean = torch.tensor(mean_np, dtype=torch.float).squeeze()
                    std = torch.tensor(std_np, dtype=torch.float).squeeze()

                    # Check shape compatibility BEFORE applying
                    if mean.shape == (x.shape[1],) and std.shape == (x.shape[1],):
                        x = (x - mean) / (std + 1e-8)
                    else:
                        print(f"Warning: Normalization stats shape mismatch (Mean: {mean.shape}, Std: {std.shape}) vs features ({x.shape[1]}). Skipping normalization for {h5_path}.")
                except Exception as e:
                    print(f"Error applying normalization stats for {h5_path}: {e}. Skipping normalization.")
            else:
                # Per-scenario normalization is generally NOT recommended here
                # due to many non-sensor nodes potentially having zero features.
                # mean = x.mean(dim=0, keepdim=True)
                # std = x.std(dim=0, keepdim=True)
                # x = (x - mean) / (std + 1e-8) # Apply with caution
                pass # Default: No per-scenario normalization


            # Create Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
            data.num_nodes = num_all_nodes # Explicitly set num_nodes for potential checks

            # Add scenario identifier if needed for debugging or analysis
            # scenario_id = os.path.splitext(os.path.basename(h5_path))[0]
            # data.scenario_id = scenario_id

            return data

    except Exception as e:
        print(f"FATAL Error processing scenario {h5_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    paths.ensure_dirs()
    h5_files = sorted(glob.glob(os.path.join(paths.RAW_DATA_DIR, "scenario_*.h5")))
    print(f"Found {len(h5_files)} raw scenarios.")

    # Optionally: compute global normalization stats across all scenarios
    # For simplicity, normalize per scenario here

    for h5_path in tqdm(h5_files, desc="Processing scenarios"):
        scenario_id = os.path.splitext(os.path.basename(h5_path))[0].split("_")[-1]
        data = process_scenario(h5_path)
        torch.save(data, paths.processed_scenario_path(scenario_id))

if __name__ == "__main__":
    main()