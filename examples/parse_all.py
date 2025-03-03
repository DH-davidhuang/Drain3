import os
import subprocess

def parse_all_datasets(base_dir, parser_script="drain_train_infer.py"):
    """
    Recursively parse each dataset folder under `base_dir`.
    1) For each dataset folder, it looks for <dataset>_2k.log and <dataset>_2k.log_structured.csv.
    2) Calls 'drain_train_infer.py' to train on 10%, then run inference on entire file.
    """
    for entry in os.listdir(base_dir):
        dataset_path = os.path.join(base_dir, entry)
        if not os.path.isdir(dataset_path):
            continue  # Skip non-folders

        dataset_name = entry 
        log_file = os.path.join(dataset_path, f"{dataset_name}_2k.log")
        csv_file = os.path.join(dataset_path, f"{dataset_name}_2k.log_structured.csv")

        results_folder = f"results_{dataset_name}"
        os.makedirs(results_folder, exist_ok=True)

        # If the log file exists, run train_infer on it
        if os.path.isfile(log_file):
            print(f"[parse_all] Processing LOG file for dataset '{dataset_name}': {log_file}")
            snapshot_json = os.path.join(results_folder, f"{dataset_name}_snapshot_log.json")
            subprocess.run([
                "python", parser_script,
                "--mode", "log",
                "--path", log_file,
                "--dataset", dataset_name,
                "--results_folder", results_folder,
                "--subset_ratio", "0.1",  # train on 10%
                "--persistence_file", snapshot_json
            ])

        # If the CSV file exists, run train_infer on it
        if os.path.isfile(csv_file):
            print(f"[parse_all] Processing CSV file for dataset '{dataset_name}': {csv_file}")
            snapshot_json = os.path.join(results_folder, f"{dataset_name}_snapshot_csv.json")
            subprocess.run([
                "python", parser_script,
                "--mode", "csv",
                "--path", csv_file,
                "--dataset", dataset_name,
                "--results_folder", results_folder,
                "--content_column", "Content",
                "--subset_ratio", "0.1",
                "--persistence_file", snapshot_json
            ])

# Example usage:
if __name__ == "__main__":
    base_dir = "/home/davidh/logparser/data"
    parse_all_datasets(base_dir, parser_script="drain_train_infer.py")
