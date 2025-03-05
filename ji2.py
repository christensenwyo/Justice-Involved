# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 16:33:34 2025

@author: ConnorChristensen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load the dataset
file_path = "C:/Users/ConnorChristensen/OneDrive - Wyoming Business Council/Documents/Analysis/Justice Involved/Reporting Data.csv"
ji = pd.read_csv(file_path)

print(ji.columns)

# List of columns to convert
datetime_columns = [
    "Submission Date",
    "Last Update Date",
    "Birthdate",
    "Date Received",
    "Parole Eligibility Date",
    "Projected Max Date",
    "Date of Assessment Completed",
    "Program Completion Date if Required",
    "Date Verified",
    "Date Released from Prison",
    "Termination Date",
    "Parole Warrant Date"
]

# Convert columns to datetime
for col in datetime_columns:
    ji[col] = pd.to_datetime(ji[col], errors='coerce')

# Verify the conversion
print(ji.dtypes)  # This will show the new data types
print(ji[datetime_columns].head())  # View sample converted data

print(ji['Education Level'].value_counts())

# Define an ordered hierarchy for education levels
education_hierarchy = [
    "No education level achieved",
    "GED obtained while incarcerated",
    "GED prior to incarceration",
    "High School Diploma",
    "College"
]

# Function to extract the highest education level
def get_highest_education(levels):
    if isinstance(levels, str):  # Ensure it's a string
        # Split multi-level education entries
        level_list = levels.split("\n")
        # Find the highest level based on hierarchy
        for level in reversed(education_hierarchy):  # Start from highest
            if level in level_list:
                return level
    return levels  # If it's already clean

# Apply function to clean the column
ji["Education Level"] = ji["Education Level"].apply(get_highest_education)

# Check the cleaned values
print(ji["Education Level"].value_counts())

###############################################################################

job_outcome_columns = [
    "Does offender have a job once released?", "Sex", "Current Facility Location", 
    "Offense Type", "Completed Institutional Treatment during current term of Incarceration", 
    "Education Level", "Field Service Office Location", "City", "Required Level of Treatment"
]

# Subset the dataset
ji_job = ji[job_outcome_columns].dropna(subset=["Does offender have a job once released?"])

# Standardize text format
ji_job = ji_job.apply(lambda x: x.astype(str).str.strip().str.lower())

# Convert Job Outcome to binary (Yes -> 1, No -> 0)
job_map = {"yes": 1, "no": 0}
ji_job["Job_Attained"] = ji_job["Does offender have a job once released?"].map(job_map)

# Define categorical columns to analyze
categorical_columns = [
    "Sex", "Current Facility Location", "Offense Type",
    "Completed Institutional Treatment during current term of Incarceration",
    "Education Level", "Field Service Office Location", "City",
    "Required Level of Treatment"
]

# Set Seaborn theme for better visuals
sns.set_theme(style="whitegrid")

# Create bar plots with n-counts
for col in categorical_columns:
    plt.figure(figsize=(10, 6))  # Adjust figure size
    ax = sns.barplot(
        x=ji_job[col], 
        y=ji_job["Job_Attained"], 
        palette="coolwarm",  # Color scheme
        ci=None
    )

    # Get value counts for n labels
    value_counts = ji_job[col].value_counts()

    # Add counts (n) on top of bars
    for i, bar in enumerate(ax.patches):
        x_position = bar.get_x() + bar.get_width() / 2
        height = bar.get_height()
        category = ax.get_xticklabels()[i].get_text()
        n_value = value_counts.get(category, 0)  # Get count, default to 0 if missing

        # Display both percentage and n count
        plt.text(
            x_position, height + 0.02, f"n={n_value}", 
            ha="center", fontsize=12, fontname="Arial", fontweight="bold"
        )

    # Improve visual styling
    plt.xticks(rotation=45, ha="right", fontsize=12, fontname="Arial")
    plt.yticks(fontsize=12, fontname="Arial")
    plt.xlabel(col, fontsize=14, fontname="Arial", fontweight="bold")
    plt.ylabel("Job Attainment Rate", fontsize=14, fontname="Arial", fontweight="bold")
    plt.title(f"Job Attainment by {col}", fontsize=16, fontname="Arial", fontweight="bold")

    # Add gridlines for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    # Show plot
    plt.show()
    
###############################################################################

# Display summary statistics
summary_stats = ji_job.groupby(categorical_columns)["Job_Attained"].mean()
print(summary_stats)

################################################################################

# Compute job attainment percentage and count
treatment_summary = ji_job.groupby("Completed Institutional Treatment during current term of Incarceration")["Job_Attained"].agg(['mean', 'count'])

# Rename columns for clarity
treatment_summary = treatment_summary.rename(columns={'mean': 'Job_Attainment_Percentage', 'count': 'n'})

# Convert percentage to readable format
treatment_summary["Job_Attainment_Percentage"] = (treatment_summary["Job_Attainment_Percentage"] * 100).round(2)

# Display the results
print(treatment_summary)

################################################################################

print(ji["Company Name"].unique())

##################################################################################

industry_list = [
    "Oil & Gas Services", "Glass Installation & Repair", "Oil & Gas Equipment Manufacturing",
    "Mining & Minerals", "Disability Services", "To Be Determined", "Restaurant (Casual Dining)",
    "Steel Fabrication", "Municipal Government", "Equipment Rental", "Restaurant (Casual Dining)",
    "Restaurant (Deli)", "Construction Services", "Hospitality (Hotel)", "Restaurant (Family Dining)",
    "Railroad Track Components", "Hospitality (Hotel)", "Beauty Retail", "Restaurant (Fast Food)",
    "Hospitality (Hotel)", "Restaurant (Casual Dining)", "Building Materials (Trusses)",
    "Fencing Services", "Employment Services", "Inspection Services", "Tourism & Agriculture",
    "Convenience Store & Gas Station", "Restaurant (Fast Food)", "Industrial Services",
    "Restaurant (Mexican Cuisine)", "Landscaping Services", "Restaurant (Casual Dining)",
    "Cleaning Services", "Landscaping Services", "Construction Materials (Concrete)",
    "Restaurant (Casual Dining)", "Employment Services", "Restaurant (Fast Casual)",
    "Landscaping Services", "Restaurant (Fast Food)", "Restaurant (Steakhouse)",
    "Dry Cleaning Services", "Automotive Sales", "Handyman Services", "Manufacturing Services",
    "Construction Services", "Landscaping Services", "Auto Body Repair", "Construction Services",
    "Renewable Energy", "Oil & Gas Services", "Cleaning Services", "Hospitality & Services",
    "Hospitality (Hotel)", "Hospitality (Hotel)", "Restaurant (Fast Food)",
    "Truck Stop & Services", "Industrial Services", "Restaurant (Fast Food)",
    "Technology Services", "Building Materials", "Construction Services", "Mining & Minerals",
    "Coating Services", "Hospitality (Hotel)", "Restaurant (Casual Dining)", "Restaurant (Diner)",
    "Construction Services", "Construction Services", "Industrial Controls Manufacturing",
    "Restaurant (Casual Dining)", "Masonry Services", "Restaurant Management", "Equipment Rental",
    "Agribusiness", "Convenience Store & Gas Station", "Electrical Services", "Metal Fabrication",
    "Grocery Retail", "Gas Station & Convenience Store", "Machining Services", "Restaurant (Fast Food)",
    "Construction Services", "Landscaping Services", "Oil & Gas Equipment Manufacturing",
    "Property Management", "Mining Services", "Hospitality & Services", "Hospitality (Hotel)",
    "Courier & Delivery Services", "Agricultural Cooperative", "Gas Station & Convenience Store",
    "Restaurant (Deli)", "Metal Fabrication", "Oil & Gas Services", "Ranching & Agriculture",
    "Hospitality (Hotel)", "Construction Services", "Restaurant (Barbecue)", "Building Materials (Trusses)",
    "Hospitality (Motel)", "Not Specified", "Industrial Services", "Oil & Gas Services",
    "Employment Services", "Automotive Services", "Oil & Gas Services", "Retail (Jewelry & Accessories)"
]

# Count occurrences of each industry
industry_counts = pd.Series(industry_list).value_counts()

# Generate the bar chart for industry distribution
plt.figure(figsize=(12, 8))
sns.barplot(y=industry_counts.index, x=industry_counts.values, palette="viridis")
plt.xlabel("Number of Companies", fontsize=14, fontweight="bold")
plt.ylabel("Industry", fontsize=14, fontweight="bold")
plt.title("Distribution of Companies by Industry", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()


################################################################################

# Create the industry crosswalk dictionary
industry_crosswalk = {
    "Butler Field Services": "Oil & Gas Services",
    "DJ's Glass": "Glass Installation & Repair",
    "Sivalls": "Oil & Gas Equipment Manufacturing",
    "Blackhills Bentonite (Trademark)": "Mining & Minerals",
    "SSDI": "Disability Services",
    "TBD": "To Be Determined",
    "Old Chicago": "Restaurant (Casual Dining)",
    "Puma Steel": "Steel Fabrication",
    "City of Cheyenne": "Municipal Government",
    "A1 Rental": "Equipment Rental",
    "Old Chicago's": "Restaurant (Casual Dining)",
    "Smiling Moose Deli": "Restaurant (Deli)",
    "Tru Grit": "Construction Services",
    "Travelodge": "Hospitality (Hotel)",
    "Village Inn": "Restaurant (Family Dining)",
    "Nortrak": "Railroad Track Components",
    "Red Lion Inn": "Hospitality (Hotel)",
    "Ulta Beauty": "Beauty Retail",
    "Subway (Center St)": "Restaurant (Fast Food)",
    "Wingate Hotel": "Hospitality (Hotel)",
    "Sanfords": "Restaurant (Casual Dining)",
    "Trusscraft": "Building Materials (Trusses)",
    "Corner to Corner Fencing": "Fencing Services",
    "Workforce Services": "Employment Services",
    "Pathfinder Inspections": "Inspection Services",
    "Terry Bison Ranch": "Tourism & Agriculture",
    "Maverik Adventure's First Stop": "Convenience Store & Gas Station",
    "A&W Long John Silver": "Restaurant (Fast Food)",
    "TCRI": "Industrial Services",
    "La Cocina": "Restaurant (Mexican Cuisine)",
    "Earth Works": "Landscaping Services",
    "Two Doors Down": "Restaurant (Casual Dining)",
    "Will Clean it Up Services": "Cleaning Services",
    "Earthworks": "Landscaping Services",
    "Croell Concrete": "Construction Materials (Concrete)",
    "Outback Steakhouse": "Restaurant (Casual Dining)",
    "Express Employment": "Employment Services",
    "Q'Doba": "Restaurant (Fast Casual)",
    "Coloscapes": "Landscaping Services",
    "McDonalds": "Restaurant (Fast Food)",
    "Rib and Chop": "Restaurant (Steakhouse)",
    "Deluxe Cleaners": "Dry Cleaning Services",
    "MJ Auto Sales": "Automotive Sales",
    "New World Handyman": "Handyman Services",
    "Extreme Precision": "Manufacturing Services",
    "TenCo": "Construction Services",
    "AAA Landscaping": "Landscaping Services",
    "Hillcrest Auto Body": "Auto Body Repair",
    "Whispering Springs Construction": "Construction Services",
    "Green Reserve Energy": "Renewable Energy",
    "C&H Well Service": "Oil & Gas Services",
    "We'll Clean it UP": "Cleaning Services",
    "Daniel Junction": "Hospitality & Services",
    "Ramada Inn": "Hospitality (Hotel)",
    "Plains Hotel": "Hospitality (Hotel)",
    "Carl's Jr": "Restaurant (Fast Food)",
    "Wagon Wheel Truck Stop": "Truck Stop & Services",
    "Titan Solutions": "Industrial Services",
    "Wendy's": "Restaurant (Fast Food)",
    "Advanced Systems": "Technology Services",
    "Capitol Lumber": "Building Materials",
    "First Pass Construction": "Construction Services",
    "Black Hills Bentonite": "Mining & Minerals",
    "Innovative Coating Solutions": "Coating Services",
    "Wingate Inn": "Hospitality (Hotel)",
    "Buffalo Wild Wings": "Restaurant (Casual Dining)",
    "Denny's": "Restaurant (Diner)",
    "Van Ewing Construction": "Construction Services",
    "Gateway Construction": "Construction Services",
    "Master Controls": "Industrial Controls Manufacturing",
    "Sanford's": "Restaurant (Casual Dining)",
    "Accent Masonry": "Masonry Services",
    "DRM": "Restaurant Management",
    "Sunbelt Rentals": "Equipment Rental",
    "Simplot": "Agribusiness",
    "Shell Food Mart": "Convenience Store & Gas Station",
    "RMS Electric": "Electrical Services",
    "Iron Arc": "Metal Fabrication",
    "Albertsons": "Grocery Retail",
    "Sinclair Big D": "Gas Station & Convenience Store",
    "Pro Line Machining": "Machining Services",
    "Carls Jr.": "Restaurant (Fast Food)",
    "JE Dunn": "Construction Services",
    "Curb Appeal": "Landscaping Services",
    "Sivalls Inc.": "Oil & Gas Equipment Manufacturing",
    "One Way Property Mgmt.": "Property Management",
    "Moore Mining": "Mining Services",
    "Willow Creek": "Hospitality & Services",
    "Wingate": "Hospitality (Hotel)",
    "FedEx": "Courier & Delivery Services",
    "CBH CO-OP": "Agricultural Cooperative",
    "Ghost Town/Stinkers": "Gas Station & Convenience Store",
    "Steamboat Deli": "Restaurant (Deli)",
    "Red Deer Iron Works": "Metal Fabrication",
    "Integrity Field Services": "Oil & Gas Services",
    "33 Mile Ranch LLC": "Ranching & Agriculture",
    "Red Lion Hotel": "Hospitality (Hotel)",
    "Welch Construction": "Construction Services",
    "Dickey's BBQ": "Restaurant (Barbecue)",
    "Truss Craft of Wyoming": "Building Materials (Trusses)",
    "Red Lion Motel": "Hospitality (Motel)",
    "Unknown": "Not Specified",
    "Tenco": "Industrial Services",
    "Kissack Water and Oil": "Oil & Gas Services",
    "Work Force Services": "Employment Services",
    "Big O Tires": "Automotive Services",
    "Edgerton Services & Equipment": "Oil & Gas Services",
    "Claire's": "Retail (Jewelry & Accessories)"
}

# Assume the dataset has a 'Company Name' column, create a new 'Industry' column
ji["Industry"] = ji["Company Name"].map(industry_crosswalk)

# Count the top 10 most represented industries
industry_counts = ji["Industry"].value_counts().head(10)

# Generate the bar chart for the top 10 industries
plt.figure(figsize=(12, 6))
sns.barplot(y=industry_counts.index, x=industry_counts.values, palette="coolwarm")
plt.xlabel("Number of Jobs", fontsize=14, fontweight="bold")
plt.ylabel("Industry", fontsize=14, fontweight="bold")
plt.title("Top 10 Industries for Justice-Involved Employment", fontsize=16, fontweight="bold")
plt.grid(axis="x", linestyle="--", alpha=0.7)
plt.show()

###############################################################################

# Map job attainment column to binary values
job_map = {"yes": 1, "no": 0}
ji["Job_Attained"] = ji["Does offender have a job once released?"].str.lower().map(job_map)

# Convert 'Days Between Release & Parole Warrant Date' to categorical groups
ji["Parole_Violation_Group"] = ji["Days Between Release & Parole Warrant Date"].apply(
    lambda x: "0 (No Violation)" if pd.isna(x) else 
              "1-30 Days" if x <= 30 else 
              "31-90 Days" if x <= 90 else 
              "91-180 Days" if x <= 180 else "181+ Days"
)

# Group by parole violation group and calculate job attainment rate and counts
parole_violation_summary = ji.groupby("Parole_Violation_Group")["Job_Attained"].agg(['mean', 'count'])

# Rename columns for clarity
parole_violation_summary = parole_violation_summary.rename(columns={'mean': 'Job_Attainment_Rate', 'count': 'n'})

# Convert percentage to readable format
parole_violation_summary["Job_Attainment_Rate"] = (parole_violation_summary["Job_Attainment_Rate"] * 100).round(2)

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(x=parole_violation_summary.index, y=parole_violation_summary["Job_Attainment_Rate"], palette="coolwarm")
plt.xlabel("Parole Violation Group", fontsize=14, fontweight="bold")
plt.ylabel("Job Attainment Rate (%)", fontsize=14, fontweight="bold")
plt.title("Job Attainment by Parole Violation Group", fontsize=16, fontweight="bold")

# Annotate bars with the count (n)
for index, value in enumerate(parole_violation_summary["Job_Attainment_Rate"]):
    plt.text(index, value + 1, f"n={parole_violation_summary['n'][index]}", ha='center', fontsize=12, fontweight="bold")

plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.show()
