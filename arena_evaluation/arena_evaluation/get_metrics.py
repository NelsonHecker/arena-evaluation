#!/usr/bin/env python3

import argparse
import os
import arena_evaluation.scripts.metrics as Metrics
from ament_index_python.packages import get_package_share_directory

try:
    from arena_evaluation.mcap_to_csv import convert_directory
    CAN_CONVERT_MCAP = True
except ImportError as e:
    print(f"MCAP conversion module not loaded: {e}. Skipping MCAP check.")
    CAN_CONVERT_MCAP = False


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", "-d", help="Directory where the data is stored")
    parser.add_argument("--pedsim", action="store_const", const=True, default=False, help="Flag to enable Pedsim metrics")
    arguments = parser.parse_args()

    dir_arg = os.path.join(
        get_package_share_directory(
            "arena_evaluation"),
            "data",
            arguments.dir
    ) 

    if CAN_CONVERT_MCAP:
        convert_directory(dir_arg)

    if arguments.pedsim:
        metrics = Metrics.PedsimMetrics(dir=dir_arg)
    else:
        metrics = Metrics.Metrics(dir=dir_arg)

    # Save the calculated metrics to a CSV file
    metrics.data.to_csv(os.path.join(dir_arg, "metrics.csv"))

if __name__ == "__main__":
    main()