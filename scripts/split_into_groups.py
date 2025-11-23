#!/usr/bin/env python3

"""
Name: split_into_groups.py
Purpose: Split the subjects who are "control" (or "moderate") at baseline into two groups based on their 'cahalan' status.
    - Group 0 --> No subsequent time points with cahalan 'heavy' or 'heavy_with_binging"
    - Group 1 --> At least one time point with cahalan 'heavy' or 'heavy_with_binging"
"""

import pandas as pd

df = pd.read_csv('data/csv/cahalan_plus_drugs.csv')

# Set CONTROL_ONLY to True to only include subjects who are "control" at baseline
# Set CONTROL_ONLY to False to include subjects who are "control" or "moderate" at baseline
CONTROL_ONLY = False

if CONTROL_ONLY:
    print("Control ONLY")
    # Filter for subjects that are "control" on visit 0
    subjects_visit_0 = df[(df['visit'] == 0) & (df['cahalan'] == 'control')]
else:
    print("Control and Moderate")
    # Filter for subjects that are "control" or "moderate" on visit 0
    subjects_visit_0 = df[(df['visit'] == 0) & ((df['cahalan'] == 'control') | (df['cahalan'] == 'moderate'))]

# Get unique subject IDs for those subjects on visit 0
unique_subjects = subjects_visit_0['subject'].unique()

# Initialize an empty set to hold subjects who later become "heavy" or "heavy_with_binging"
heavy_subjects = set()

# Loop through each unique subject and check their subsequent visits
for subject in unique_subjects:
    subject_data = df[df['subject'] == subject]
    if any(subject_data['cahalan'].isin(['heavy', 'heavy_with_binging'])):
        heavy_subjects.add(subject)

# Convert the set to a list
heavy_subjects = list(heavy_subjects)

# Get all unique subjects
all_subjects = df['subject'].unique()

# Determine subjects in the second group (everyone else)
no_heavy_subjects = list(set(all_subjects) - set(heavy_subjects))

# Update the 'group' column based on the subject's status
# Create a new column 'group' and initialize with '0'
df['group'] = 0

# Set the value to '1' for subjects in the heavy group
df.loc[df['subject'].isin(heavy_subjects), 'group'] = 1

if CONTROL_ONLY:
    print("Control ONLY")
    # Only keep the subjects who are controls (unproblematic drinking) on visit 0 (baseline)
    df = df[(df['visit'] == 0) & (df['cahalan'] == 'control')]
else:
    print("Control and Moderate")
    # Only keep the subjects who are controls or moderate (unproblematic drinking) on visit 0 (baseline)
    df = df[(df['visit'] == 0) & ((df['cahalan'] == 'control') | (df['cahalan'] == 'moderate'))]

# Keep only subject, cahalan, and group columns
df = df[['subject', 'visit', 'cahalan', 'group']]

if CONTROL_ONLY:
    print("Control ONLY")
    # Save to a new CSV file
    df.to_csv('data/csv/control_subjects_with_groups.csv', index=False)
else:
    # Save to a new CSV file
    print("Control and Moderate")
    df.to_csv('data/csv/control_moderate_subjects_with_groups.csv', index=False)