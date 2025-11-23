#!/usr/bin/env python3

"""
Name: process_demographics.py
Purpose: Get X and y arrays for the demographic data to use in the logistic regression model.

- Binary encoding of sex --> Male (1), Female (0)
- One-Hot encoding of race labels (10 categories -> 4 categories)
    Races = ['Caucasian/White', 'Asian', 'African-American/Black',
            'African-American_Caucasian', 'Pacific Islander', 'Asian_White',
            'NativeAmerican_Caucasian', 'Native American/American Indian',
            'Asian_Pacific_Islander', 'Pacific_Islander_Caucasian']
    Mapped to --> ['Caucasian/White', 'Asian', 'African-American/Black', 'Other']
- Standardized the age column
- One-Hot encoding of site labels (5 categories)
    Sites = ['A', 'B', 'C', 'D', 'E'] --> UPMC: A, SRI: B, Duke: C, OHSU: D, UCSD: E
- Binary encoding of scanner models --> GE (1), Siemens (0)
    Scanner Models = ['MR750', 'Prisma_Fit, 'TrioTim']
    Mapped to --> ['GE', 'Siemens'] (MR750: GE, PrismaFit & TrioTim: Siemens)
- Standardized parents' years of education
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Set CONTROL_ONLY to True to only include subjects who are "control" at baseline
# Set CONTROL_ONLY to False to include subjects who are "control" or "moderate" at baseline
CONTROL_ONLY = False

# Load the demographic data
df = pd.read_csv('data/csv/demographics_short.csv')

df = df[df['visit'] == 'baseline'] # Filter for baseline subjects
df = df.drop_duplicates(subset=['subject']) # Drop duplicate subject IDs
df = df[['subject', 'visit', 'sex', 'visit_age', 'race_label', 'site', 'scanner_model']] # Only keep relevant columns

# Binary encoding for 'sex' --> Male is 1, Female is 0
dummy = pd.get_dummies(df['sex']).astype(int)
df = pd.concat([df, dummy], axis=1)
df.drop('sex', axis=1, inplace=True)
df.drop('F', axis=1, inplace=True)
df.rename(columns={'M': 'sex'}, inplace=True)

# PROCESS RACE LABELS
# drop rows where race_label is null
df.dropna(subset=['race_label'], inplace=True)

race_mapping = {
    'African-American/Black': 'African-American/Black',
    'African-American_Caucasian': 'Other',
    'Asian': 'Asian',
    'Asian_Pacific_Islander': 'Other',
    'Asian_White': 'Other',
    'Caucasian/White': 'Caucasian/White',
    'Native American/American Indian': 'Other',
    'NativeAmerican_Caucasian': 'Other',
    'Pacific Islander': 'Other',
    'Pacific_Islander_Caucasian': 'Other'
}

# Apply the mapping to the 'race_label' column
df['race_label'] = df['race_label'].map(race_mapping)
# One-Hot Encoding for Races
dummy = pd.get_dummies(df['race_label']).astype(int)
df = pd.concat([df, dummy], axis=1)
df.drop('race_label', axis=1, inplace=True)

# Standardize the age column
scaler = StandardScaler()
df[['visit_age']] = scaler.fit_transform(df[['visit_age']])

# SES - parents' years of education (standardize)
ses_df = pd.read_csv('data/csv/clinical_short.csv')
ses_df = ses_df[ses_df['visit'] == 'baseline'] # Filter for baseline subjects
ses_df = ses_df.drop_duplicates(subset=['subject']) # Drop duplicate subject IDs
ses_df = ses_df[['subject', 'ses_parent_yoe']] # Only keep relevant columns

df = df.merge(ses_df, on='subject') # Merge the ses data with the demographic data
df[['ses_parent_yoe']] = scaler.fit_transform(df[['ses_parent_yoe']]) # Standardize the parents' years of education column
df.dropna(subset=['ses_parent_yoe'], inplace=True) # drop rows where ses_parent_yoe is null

# PROCESS SITE LABELS
# drop rows where site is null
df.dropna(subset=['site'], inplace=True)

# One-Hot Encoding for Sites
dummy = pd.get_dummies(df['site']).astype(int)
df = pd.concat([df, dummy], axis=1)
df.drop('site', axis=1, inplace=True)

# PROCESS SCANNER MODEL LABELS
# drop rows where scanner_model is null
df.dropna(subset=['scanner_model'], inplace=True)

# Map scanner models to combine "PrismaFit" and "TrioTim" to "Siemens" and rename "MR750" to "GE"
scanner_mapping = {
    'PrismaFit': 'Siemens',
    'TrioTim': 'Siemens',
    'MR750': 'GE'
}

# Apply the mapping to the 'scanner_model' column
df['scanner_model'] = df['scanner_model'].map(scanner_mapping)

# Create a binary encoding: GE as 1, Siemens as 0
df['scanner_model'] = df['scanner_model'].apply(lambda x: 1 if x == 'GE' else 0)

# Load the subjects with group labels data
if CONTROL_ONLY:
    subjects_with_groups_df = pd.read_csv('data/csv/control_subjects_with_groups.csv')
else:
    subjects_with_groups_df = pd.read_csv('data/csv/control_moderate_subjects_with_groups.csv')
    # Add a column to indicate the group label (0 for control, 1 for moderate)
    subjects_with_groups_df['moderate'] = subjects_with_groups_df['cahalan'].apply(lambda x: 0 if x == 'control' else 1)

subjects_with_groups_df.drop(['visit', 'cahalan'], axis=1, inplace=True)

# Merge the demographic data to get corresponding group labels
demos_with_groups = subjects_with_groups_df.merge(df, on='subject')

# Save the demographic data as X and y arrays
# We drop ['Caucasian/White', 'E'] because they serve as the reference categories (to avoid the Dummy)
X = demos_with_groups.drop(['subject', 'visit', 'group', 'Caucasian/White', 'E'], axis=1).values
y = demos_with_groups['group'].values

if CONTROL_ONLY:
    file_name = 'demos_control'
else:
    file_name = 'demos_control_moderate'

np.save(f'data/training_data/unaligned/X_{file_name}.npy', X)
np.save(f'data/training_data/unaligned/y_{file_name}.npy', y)

# *** This was used to save the demographic data with the subject ids ***
df = demos_with_groups.drop(['visit', 'group', 'Caucasian/White', 'E'], axis=1)
if CONTROL_ONLY:
    df.to_csv('data/data_with_subject_ids/control_demos_with_subjects.csv', index=False)
else:
    df.to_csv('data/data_with_subject_ids/control_moderate_demos_with_subjects.csv', index=False)

print("columns", df.columns)
print("shape", df.shape)