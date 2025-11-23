#!/bin/bash

# Name: prep_data_for_training.py

# Depending if you want baseline subjects who are CONTROL only or CONTROL/MODERATE, there is a global variable in each of the files below that you can change.

python scripts/split_into_groups.py; # Split data into groups based on target label (0 and 1)
python scripts/process_conn_matrices.py; # Get connectivity matrices ready for training (SC, FC, FCgsr)
python scripts/process_demographics.py; # Get demographics ready for training
python scripts/align_training_data.py; # Align data (SC, FC, FCgsr, demos) for training (only keep common subjects)