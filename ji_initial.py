# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:46:18 2025

@author: ConnorChristensen
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/Analysis/Justice Involved/Master Phase 3 Data - 1-7-25 - Master.csv"
ji = pd.read_csv(file_path)

# Select relevant columns
relevant_columns = [
    "Does offender have a job once released?",
    "If yes, has employment been verified with employer?",
    "Assessment Risk Rating",
    "Completed Institutional Treatment during current term of Incarceration",
    "Has housing been verified?"
]

# Subset and clean data
ji_subset = ji[relevant_columns].dropna()
ji_subset = ji_subset.apply(lambda x: x.astype(str).str.strip().str.lower())

# Convert binary responses to numerical
binary_map = {"yes": 1, "no": 0}
ji_subset["Job_Attained"] = ji_subset["Does offender have a job once released?"].map(binary_map)
ji_subset["Housing_Verified"] = ji_subset["Has housing been verified?"].map(binary_map)
ji_subset["Institutional_Treatment"] = ji_subset["Completed Institutional Treatment during current term of Incarceration"].map(binary_map)

# Encode categorical variables (Risk Rating)
risk_map = {"low": 1, "moderate": 2, "high": 3}
ji_subset["Risk_Rating"] = ji_subset["Assessment Risk Rating"].map(risk_map)

# Debugging Step 1: Check for missing values
print("\nMissing Values Count:")
print(ji_subset.isnull().sum())

# Drop NaN rows from essential columns
ji_subset = ji_subset.dropna(subset=["Job_Attained", "Risk_Rating", "Housing_Verified", "Institutional_Treatment"])

# Debugging Step 2: Check dataset size
print(f"\nDataset size after dropping missing values: {ji_subset.shape}")

# Debugging Step 3: Ensure numeric types
print("\nData Types Before Regression:")
print(ji_subset.dtypes)

# Ensure all predictor variables are numeric
ji_subset[["Risk_Rating", "Housing_Verified", "Institutional_Treatment"]] = ji_subset[
    ["Risk_Rating", "Housing_Verified", "Institutional_Treatment"]
].apply(pd.to_numeric)

# Debugging Step 4: Confirm data balance in dependent variable
print("\nJob Attainment Value Counts:")
print(ji_subset["Job_Attained"].value_counts())

# Check if there are enough positive and negative cases
if ji_subset["Job_Attained"].nunique() < 2:
    print("\nError: Not enough variation in the dependent variable for logistic regression.")
else:
    # Add constant for logistic regression
    ji_subset = ji_subset.assign(constant=1)

    # Logistic Regression Model
    log_model = sm.Logit(ji_subset["Job_Attained"], ji_subset[["constant", "Risk_Rating", "Housing_Verified", "Institutional_Treatment"]]).fit()
    
    # Print model summary
    print(log_model.summary())

    # Visualization
    plt.figure(figsize=(8,5))
    sns.barplot(x=ji_subset["Risk_Rating"], y=ji_subset["Job_Attained"], ci=None)
    plt.xlabel("Risk Rating (Low=1, High=3)")
    plt.ylabel("Job Attainment Rate")
    plt.title("Impact of Risk Rating on Job Attainment")
    plt.show()

    plt.figure(figsize=(8,5))
    sns.barplot(x=ji_subset["Housing_Verified"], y=ji_subset["Job_Attained"], ci=None)
    plt.xlabel("Housing Verified (No=0, Yes=1)")
    plt.ylabel("Job Attainment Rate")
    plt.title("Impact of Housing Verification on Job Attainment")
    plt.show()


