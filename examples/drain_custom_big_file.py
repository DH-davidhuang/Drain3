# SPDX-License-Identifier: MIT
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

def setup_logging(output_file):
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

def process_log_file(log_file_path, template_miner, logger, batch_size=10000):
    line_count = 0
    start_time = time.time()
    with open(log_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            result = template_miner.add_log_message(line)
            line_count += 1

            if line_count % batch_size == 0:
                elapsed = time.time() - start_time
                rate = batch_size / elapsed
                logger.info(f"[LOG] Processed line: {line_count}, rate={rate:.1f} lines/sec, "
                            f"clusters={len(template_miner.drain.clusters)}")
                start_time = time.time()

            if result["change_type"] != "none":
                logger.info(f"Input ({line_count}): {line}")
                logger.info(f"Result: {json.dumps(result)}")

    logger.info(f"Done processing {line_count} lines from {log_file_path}. "
                f"Total clusters = {len(template_miner.drain.clusters)}")

def process_csv_file(csv_file_path, template_miner, logger, content_column="Content", batch_size=10000):
    line_count = 0
    start_time = time.time()
    with open(csv_file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            line = row[content_column].strip()
            result = template_miner.add_log_message(line)
            line_count += 1

            if line_count % batch_size == 0:
                elapsed = time.time() - start_time
                rate = batch_size / elapsed
                logger.info(f"[CSV] Processed line: {line_count}, rate={rate:.1f} lines/sec, "
                            f"clusters={len(template_miner.drain.clusters)}")
                start_time = time.time()

            if result["change_type"] != "none":
                logger.info(f"Input ({line_count}): {line}")
                logger.info(f"Result: {json.dumps(result)}")

    logger.info(f"Done processing {line_count} lines from CSV ({csv_file_path}). "
                f"Total clusters = {len(template_miner.drain.clusters)}")

def main():
    parser = argparse.ArgumentParser(description="Drain3 Demo with CSV or raw log.")
    parser.add_argument("--mode", choices=["log", "csv"], required=True,
                        help="Choose which input format to parse (log file or CSV).")
    parser.add_argument("--path", required=True,
                        help="Path to the .log file or .csv file.")
    parser.add_argument("--content_column", default="Content",
                        help="CSV column that contains the log text (if mode=csv).")
    parser.add_argument("--dataset", default="HDFS",
                        help="Name of the dataset, used to name the output file.")
    parser.add_argument("--results_folder", default="results",
                        help="Name of the folder where logs are saved (default: 'results').")
    args = parser.parse_args()

    # Make sure the folder for results exists
    os.makedirs(args.results_folder, exist_ok=True)

    # Decide an output filename based on mode & dataset
    if args.mode == "log":
        out_file = f"{args.results_folder}/{args.dataset}_unstructured_log_entries.log"
    else:  # csv mode
        out_file = f"{args.results_folder}/{args.dataset}_structured_csv.log"

    # Set up logging to console + file
    logger = setup_logging(out_file)

    # Load Drain3 config
    config = TemplateMinerConfig()
    config_file = f"{dirname(abspath(__file__))}/drain3.ini"
    config.load(config_file)
    config.profiling_enabled = True

    # Create TemplateMiner
    template_miner = TemplateMiner(config=config)

    # Process input
    overall_start = time.time()
    if args.mode == "log":
        process_log_file(args.path, template_miner, logger)
    else:
        process_csv_file(args.path, template_miner, logger, content_column=args.content_column)

    # Print the summary of discovered clusters
    sorted_clusters = sorted(template_miner.drain.clusters, key=lambda it: it.size, reverse=True)
    logger.info("== Cluster List (sorted by size) ==")
    for c in sorted_clusters:
        logger.info(str(c))

    logger.info("== Prefix Tree ==")
    template_miner.drain.print_tree()

    # Profiling info
    total_time = time.time() - overall_start
    template_miner.profiler.report(0)
    logger.info(f"Completed in {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
