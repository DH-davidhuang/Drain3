#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_all.py

An all-in-one script that:
1) Defines the LogPai evaluate() and get_accuracy() functions locally,
2) Iterates over multiple datasets,
3) Compares your parsed Drain3 CSV to the ground truth CSV,
4) Prints F1 and accuracy, and saves them to 'evaluation_results.csv'.
"""

import os
import sys
import csv
import pandas as pd
from scipy.special import comb

def get_accuracy(series_groundtruth, series_parsedlog, debug=False):
    """
    Compute accuracy metrics between log parsing results and ground truth.

    Arguments
    ---------
        series_groundtruth : pandas.Series
            A sequence of groundtruth event Ids
        series_parsedlog : pandas.Series
            A sequence of parsed event Ids
        debug : bool, default False
            print error log messages when set to True

    Returns
    -------
        precision : float
        recall : float
        f_measure : float
        accuracy : float
    """
    series_groundtruth_valuecounts = series_groundtruth.value_counts()
    real_pairs = 0
    for count in series_groundtruth_valuecounts:
        if count > 1:
            real_pairs += comb(count, 2)

    series_parsedlog_valuecounts = series_parsedlog.value_counts()
    parsed_pairs = 0
    for count in series_parsedlog_valuecounts:
        if count > 1:
            parsed_pairs += comb(count, 2)

    accurate_pairs = 0
    accurate_events = 0  
    for parsed_eventId in series_parsedlog_valuecounts.index:
        logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
        series_gt_valuecounts = series_groundtruth[logIds].value_counts()

        error_eventIds = (parsed_eventId, series_gt_valuecounts.index.tolist())
        error = True
        if series_gt_valuecounts.size == 1:
            groundtruth_eventId = series_gt_valuecounts.index[0]
            if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
                accurate_events += logIds.size
                error = False

        if error and debug:
            print("(parsed_eventId, groundtruth_eventId) =", error_eventIds,
                  "failed", logIds.size, "messages")

        for count in series_gt_valuecounts:
            if count > 1:
                accurate_pairs += comb(count, 2)


    precision = float(accurate_pairs) / parsed_pairs if parsed_pairs != 0 else 0.0
    recall = float(accurate_pairs) / real_pairs if real_pairs != 0 else 0.0
    f_measure = 2.0 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0.0
    accuracy = float(accurate_events) / series_groundtruth.size if series_groundtruth.size != 0 else 0.0
    return precision, recall, f_measure, accuracy

def evaluate(groundtruth, parsedresult):
    """
    Compare 'groundtruth' CSV vs. 'parsedresult' CSV line by line,
    returning (f_measure, accuracy).

    We explicitly align on 'LineId'.
    """
    df_groundtruth = pd.read_csv(groundtruth)
    df_parsedlog = pd.read_csv(parsedresult)

    if "LineId" not in df_groundtruth.columns or "LineId" not in df_parsedlog.columns:
        raise ValueError("Both groundtruth and parsed CSV must have a 'LineId' column!")

    df_groundtruth.set_index("LineId", inplace=True)
    df_parsedlog.set_index("LineId", inplace=True)

    df_groundtruth = df_groundtruth[~df_groundtruth["EventId"].isnull()]

    common_lineids = df_groundtruth.index.intersection(df_parsedlog.index)
    df_groundtruth = df_groundtruth.loc[common_lineids]
    df_parsedlog = df_parsedlog.loc[common_lineids]

    precision, recall, f_measure, accuracy = get_accuracy(
        df_groundtruth["EventId"], df_parsedlog["EventId"]
    )
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, "
          f"F1_measure: {f_measure:.4f}, Parsing_Accuracy: {accuracy:.4f}")
    return f_measure, accuracy

#########  END EVALUATOR LOGIC  #########


def main():
    """
    Example usage that:
     1) Scans a list of dataset names
     2) For each dataset, loads ground truth + parsed CSV
     3) Calls evaluate()
     4) Prints & collects results
     5) Writes to 'evaluation_results.csv'
    """
    # Adjust these paths as needed
    data_dir = "/home/davidh/logparser/data"
    results_dir_prefix = "/home/davidh/logparser/Drain3/examples/results_"
    datasets = [
        "Proxifier",
        "HDFS",
        "BGL",
        "Hadoop",
        "OpenSSH",
        # add more if you want
    ]

    results_list = []
    for dataset in datasets:
        # Ground truth CSV
        gt_csv = os.path.join(data_dir, dataset, f"{dataset}_2k.log_structured.csv")

        # If your parse_all.py & drain_train_infer.py produce both:
        #   <Dataset>_Drain3_parsed_rawlog.csv   and
        #   <Dataset>_Drain3_parsed_structured_csv.csv
        # we can evaluate both
        results_folder = f"{results_dir_prefix}{dataset}"

        # Evaluate RAW log parse
        parsed_csv_raw = os.path.join(results_folder, f"{dataset}_Drain3_parsed_rawlog.csv")
        if os.path.isfile(parsed_csv_raw):
            print(f"\n=== Evaluating {dataset} (raw log) ===")
            f1, acc = evaluate(gt_csv, parsed_csv_raw)
            results_list.append([dataset, "rawlog", f1, acc])

        # Evaluate structured CSV parse
        parsed_csv_struct = os.path.join(results_folder, f"{dataset}_Drain3_parsed_structured_csv.csv")
        if os.path.isfile(parsed_csv_struct):
            print(f"\n=== Evaluating {dataset} (structured csv) ===")
            f1, acc = evaluate(gt_csv, parsed_csv_struct)
            results_list.append([dataset, "structured_csv", f1, acc])

    # Summaries
    if results_list:
        df_results = pd.DataFrame(results_list, columns=["Dataset","Mode","F1","Accuracy"])
        print("\nOverall evaluation summary:")
        print(df_results)
        # Save
        df_results.to_csv("evaluation_results.csv", index=False)
        print("\nSaved 'evaluation_results.csv' in current directory.")
    else:
        print("No results found to evaluate.")

if __name__ == "__main__":
    main()
