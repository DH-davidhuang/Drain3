import json
import logging
import sys
import time
import argparse
import os
from os.path import dirname, abspath
import csv

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from drain3.file_persistence import FilePersistence  

def setup_logging(output_file):
    """
    Sets up logging to both stdout and a specified output file.
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(output_file, mode="w")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

def load_lines_from_logfile(log_file_path):
    """Loads lines from a .log text file into memory, strips them."""
    with open(log_file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f]

def load_lines_from_csv(csv_file_path, content_column="Content"):
    """Loads lines from a CSV file, returns a list of strings (the log content)."""
    lines = []
    with open(csv_file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_line = row[content_column].strip()
            lines.append(raw_line)
    return lines

def train_on_subset(template_miner, lines, subset_ratio=0.1, logger=None):
    """
    Train the template miner on the first 'subset_ratio' portion of lines.
    For example, if subset_ratio=0.1, trains on first 10% of lines.
    """
    num_train = int(len(lines) * subset_ratio)
    if num_train == 0:
        num_train = 1  # Ensure at least one line is used for training

    logger.info(f"Training on first {num_train} lines (out of {len(lines)})...")
    start_time = time.time()
    for i in range(num_train):
        line = lines[i]
        result = template_miner.add_log_message(line)
        if i % 1000 == 0 and i > 0:
            elapsed = time.time() - start_time
            rate = 1000 / elapsed
            logger.info(f"Training line: {i}, rate={rate:.1f} lines/sec, "
                        f"clusters={len(template_miner.drain.clusters)}")
            start_time = time.time()
        if result["change_type"] != "none":
            logger.info(f"InputTrain ({i+1}): {line}")
            logger.info(f"Result: {json.dumps(result)}")

    logger.info(f"Training completed. {len(template_miner.drain.clusters)} clusters formed so far.")

    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    logger.info("== Cluster Summary (After Training) ==")
    for c in sorted_clusters:
        logger.info(str(c))

    logger.info("== Prefix Tree (After Training) ==")
    template_miner.drain.print_tree()

    return num_train

def inference_on_remaining(template_miner, lines, start_index, logger=None):
    """
    Inference on lines from 'start_index' to end (i.e. remaining 90%).
    Returns a list of (LineId, line, cluster_id, template).
    """
    parsed_results = []
    total = len(lines)
    logger.info(f"Starting inference on lines {start_index+1}..{total}.")
    start_time = time.time()
    batch_size = 1000
    for i in range(start_index, total):
        line = lines[i]
        match_result = template_miner.match(line)  
        if match_result is None:
            cluster_id = None
            template_str = None
        else:
            cluster_id = match_result.cluster_id
            template_str = " ".join(match_result.log_template_tokens)

        parsed_results.append((i+1, line, cluster_id, template_str))

        if (i - start_index) % batch_size == 0 and i > start_index:
            elapsed = time.time() - start_time
            rate = batch_size / elapsed
            logger.info(f"[INFER] Processed line: {i+1}, rate={rate:.1f} lines/sec")
            start_time = time.time()

    logger.info("Inference complete.")
    return parsed_results

def save_parsed_csv(output_csv, parsed_lines):
    """
    Saves the final CSV with columns: [LineId, Content, EventId, EventTemplate].
    'parsed_lines' is a list of tuples or dicts with that info.
    """
    import csv
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["LineId", "Content", "EventId", "EventTemplate"])
        for row in parsed_lines:
            writer.writerow(row)

def main():
    parser = argparse.ArgumentParser(description="Train on first 10%, then inference on entire file.")
    parser.add_argument("--mode", choices=["log", "csv"], required=True,
                        help="Choose which input format to parse (raw .log file or CSV).")
    parser.add_argument("--path", required=True,
                        help="Path to the .log file or .csv file.")
    parser.add_argument("--content_column", default="Content",
                        help="CSV column that contains the log text (if mode=csv).")
    parser.add_argument("--dataset", default="HDFS",
                        help="Name of the dataset, used to name the output files.")
    parser.add_argument("--results_folder", default="results",
                        help="Name of the folder where logs & final CSV are saved.")
    parser.add_argument("--subset_ratio", type=float, default=0.1,
                        help="Fraction of lines to use for training (default=0.1 for 10%).")
    parser.add_argument("--persistence_file", default=None,
                        help="Where to save or load the Drain3 state snapshot (JSON). If omitted, no file persistence.")
    args = parser.parse_args()

    if args.mode == "log":
        run_mode_suffix = "_rawlog"
    else:
        run_mode_suffix = "_structured_csv"

    out_logfile = os.path.join(args.results_folder,
                               f"{args.dataset}_train_infer_output{run_mode_suffix}.log")
    output_csv = os.path.join(args.results_folder,
                              f"{args.dataset}_Drain3_parsed{run_mode_suffix}.csv")

    os.makedirs(args.results_folder, exist_ok=True)

    logger = setup_logging(out_logfile)

    config = TemplateMinerConfig()
    config_file = f"{dirname(abspath(__file__))}/drain3.ini"
    config.load(config_file)
    config.profiling_enabled = True

    if args.persistence_file:
        handler = FilePersistence(args.persistence_file)
        config.persistence_handler = handler
        logger.info(f"[INFO] Using file persistence at {args.persistence_file}")
    else:
        logger.info("[INFO] No persistence file specified; no state saving will occur.")

    template_miner = TemplateMiner(config=config)

    if args.mode == "log":
        lines = load_lines_from_logfile(args.path)
    else:
        lines = load_lines_from_csv(args.path, content_column=args.content_column)

    logger.info(f"Loaded {len(lines)} lines total from {args.path}")

    # 5) training on first 10%
    n_train = train_on_subset(template_miner, lines, subset_ratio=args.subset_ratio, logger=logger)

    # 6) save the snapshot if we have file persistence
    if template_miner.persistence_handler is not None and args.persistence_file:
        template_miner.persistence_handler.save_state(template_miner.drain)
        logger.info(f"[INFO] State saved to {args.persistence_file}")
    else:
        logger.info("[INFO] No file persistence to save.")

    if args.persistence_file and os.path.isfile(args.persistence_file):
        inf_config = TemplateMinerConfig()
        inf_config.load(config_file)
        inf_config.profiling_enabled = True

        inf_handler = FilePersistence(args.persistence_file)
        inf_config.persistence_handler = inf_handler

        template_miner_inference = TemplateMiner(config=inf_config)
        logger.info(f"[INFO] Created new TemplateMiner for inference by reloading state from {args.persistence_file}")
    else:
        template_miner_inference = template_miner
        if args.persistence_file:
            logger.info(f"[WARNING] Persistence file '{args.persistence_file}' not found. Reusing in-memory model for inference.")
        else:
            logger.info("[INFO] No persistence file specified; reusing training model for inference.")

    # 8) Inference on entire file
    parsed_lines = []
    start_time = time.time()
    for i, line in enumerate(lines):
        match_result = template_miner_inference.match(line)
        if match_result is None:
            cluster_id = None
            template_str = None
        else:
            cluster_id = match_result.cluster_id
            template_str = " ".join(match_result.log_template_tokens)
        parsed_lines.append((i+1, line, cluster_id, template_str))

    total_time = time.time() - start_time
    logger.info(f"Inference for all lines took {total_time:.2f} seconds.")

    # 9) Save final CSV
    save_parsed_csv(output_csv, parsed_lines)
    logger.info(f"Saved final parsed CSV to: {output_csv}")

    # 10) Print final summaries (after inference)
    logger.info("== Final Cluster Summary ==")
    sorted_clusters = sorted(template_miner_inference.drain.clusters,
                             key=lambda it: it.size, reverse=True)
    for c in sorted_clusters:
        logger.info(str(c))

    logger.info("== Final Prefix Tree ==")
    template_miner_inference.drain.print_tree()
    template_miner_inference.profiler.report(0)
    logger.info("Done.")

if __name__ == "__main__":
    main()
