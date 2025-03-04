#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 20:34:28 2024

@author: Andrea Campos

Code to analyse Behavioral Data of Touchscreen task - Reaction times
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scs
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import ptitprince as pt
from statsmodels.stats.anova import AnovaRM


#%% Data filenames
file_sub1 = "touchscreen_task_results_sub1_2024-04-26_11-13-43.txt"
file_sub2 = "touchscreen_task_results_sub2_2024-04-26_14-26-50.txt"
file_sub3 = "touchscreen_task_results_sub3_2024-04-25_14-58-39.txt"
file_sub4 = "touchscreen_task_results_sub4_2024-04-29_11-23-51.txt"


column_names = ["Hit/Miss", "Stim_coordinates_x1", "Stim_coordinates_y1", "Stim_coordinates_x2", "Stim_coordinates_y2", "Touch_coordinates_x",  "Touch_coordinates_y", "Stim_presentation_time", "Response_time"]

# Load the text file into a DataFrame
df_s1 = pd.read_csv(file_sub1, header=None, names=column_names, skiprows=1)
df_s2 = pd.read_csv(file_sub2, header=None, names=column_names, skiprows=1)
df_s3 = pd.read_csv(file_sub3, header=None, names=column_names, skiprows=1)
df_s4 = pd.read_csv(file_sub4, header=None, names=column_names, skiprows=1)


#%% Calculate percentage of hits

acc_s1 = df_s1["Hit/Miss"].mean()
acc_s2 = df_s2["Hit/Miss"].mean()
acc_s3 = df_s3["Hit/Miss"].mean()
acc_s4 = df_s4["Hit/Miss"].mean()

#%% Calculate reaction times per subject

rt_s1 = df_s1["Response_time"] - df_s1["Stim_presentation_time"]
rt_s2 = df_s2["Response_time"] - df_s2["Stim_presentation_time"]
rt_s3 = df_s3["Response_time"] - df_s3["Stim_presentation_time"]
rt_s4 = df_s4["Response_time"] - df_s4["Stim_presentation_time"]

# Compute mean and std separately
stats_rt_s1 = np.mean(rt_s1), np.std(rt_s1)
stats_rt_s2 = np.mean(rt_s2), np.std(rt_s2)
stats_rt_s3 = np.mean(rt_s3), np.std(rt_s3)
stats_rt_s4 = np.mean(rt_s4), np.std(rt_s4)

#%% Distribution plots
#REMOVE OUTLIERS (JUST FOR PLOTTING)
percentile_threshold = 98

# Calculate the percentile for each numeric column
percentiles_s1 = rt_s1.quantile(q=(percentile_threshold / 100))
percentiles_s2 = rt_s2.quantile(q=(percentile_threshold / 100))
percentiles_s3 = rt_s3.quantile(q=(percentile_threshold / 100))
percentiles_s4 = rt_s4.quantile(q=(percentile_threshold / 100))

# Filter the DataFrame to keep only values below the percentile threshold
rt_s1_fin = rt_s1[rt_s1 <= percentiles_s1]
rt_s1_fin = rt_s1_fin.dropna()
rt_s2_fin = rt_s2[rt_s2 <= percentiles_s2]
rt_s2_fin = rt_s2_fin.dropna()
rt_s3_fin = rt_s3[rt_s3 <= percentiles_s3]
rt_s3_fin = rt_s3_fin.dropna()
rt_s4_fin = rt_s4[rt_s4 <= percentiles_s4]
rt_s4_fin = rt_s4_fin.dropna()

plt.figure(figsize=(14, 7))
sns.distplot(rt_s4_fin, label = "Subject 4")
sns.distplot(rt_s3_fin, color = "#D24545", label = "Subject 3")
sns.distplot(rt_s2_fin, color = "#864AF9", label = "Subject 2")
sns.distplot(rt_s1_fin, color = "#65B741", label = "Subject 1")
plt.legend(loc='best')
plt.xticks(size=12)
plt.yticks(size=12)
#plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
#plt.ylim(0, 10)
plt.show()


#%% LINEAR REGRESSION ANALYSIS
n_trials = np.linspace(1, percentile_threshold, len(rt_s3_fin))

#Subject 1
slope_s1, intercept_s1, r_value_s1, p_value_s1, std_err_s1 = scs.linregress(n_trials, rt_s1_fin)
predictions_s1 = slope_s1 * n_trials + intercept_s1
SSres_s1 = np.sum((rt_s1_fin - predictions_s1) ** 2)
SStot_s1 = np.sum((rt_s1_fin - np.mean(rt_s1_fin)) ** 2)
# Calcula R cuadrado
R2_s1= 1 - (SSres_s1 / SStot_s1)

#Subject 2
slope_s2, intercept_s2, r_value_s2, p_value_s2, std_err_s2 = scs.linregress(n_trials, rt_s2_fin)
predictions_s2 = slope_s2 * n_trials + intercept_s2
SSres_s2 = np.sum((rt_s2_fin - predictions_s2) ** 2)
SStot_s2 = np.sum((rt_s2_fin - np.mean(rt_s2_fin)) ** 2)
# Calcula R cuadrado
R2_s2= 1 - (SSres_s2 / SStot_s2)

#Subject 3
slope_s3, intercept_s3, r_value_s3, p_value_s3, std_err_s3 = scs.linregress(n_trials, rt_s3_fin)
predictions_s3 = slope_s3 * n_trials + intercept_s3
SSres_s3 = np.sum((rt_s3_fin - predictions_s3) ** 2)
SStot_s3 = np.sum((rt_s3_fin - np.mean(rt_s3_fin)) ** 2)
# Calcula R cuadrado
R2_s3= 1 - (SSres_s3 / SStot_s3)

#Subject 4
slope_s4, intercept_s4, r_value_s4, p_value_s4, std_err_s4 = scs.linregress(n_trials, rt_s4_fin)
predictions_s4 = slope_s4 * n_trials + intercept_s4
SSres_s4 = np.sum((rt_s4_fin - predictions_s4) ** 2)
SStot_s4 = np.sum((rt_s4_fin - np.mean(rt_s4_fin)) ** 2)
# Calcula R cuadrado
R2_s4= 1 - (SSres_s4 / SStot_s4)


# Regression plot
plt.figure(figsize=(8, 4))
sns.regplot(n_trials, rt_s4_fin, scatter =True, color ="#86A7FC")
sns.regplot(n_trials, rt_s3_fin, scatter =True, color = "#D24545")
sns.regplot(n_trials, rt_s2_fin, scatter =True, color = "#864AF9")
sns.regplot(n_trials, rt_s1_fin, scatter =True, color = "#65B741")
plt.ylabel("Reaction time (s)")
plt.xlabel("# Trial")
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()


plt.figure(figsize=(8, 4))
sns.regplot(n_trials, rt_s4_fin, scatter =False, color ="#86A7FC")
sns.regplot(n_trials, rt_s3_fin, scatter =False, color = "#D24545")
sns.regplot(n_trials, rt_s2_fin, scatter =False, color = "#864AF9")
sns.regplot(n_trials, rt_s1_fin, scatter =False, color = "#65B741")
plt.ylabel("Reaction time (s)")
plt.xlabel("# Trial")
plt.xticks(size=12)
plt.yticks(size=12)
plt.show()


# Scatter plot in log scale
plt.figure(figsize=(8,4))
plt.scatter(n_trials, rt_s4_fin, color = "#86A7FC")
plt.scatter(n_trials, rt_s3_fin, color = "#D24545", alpha = 0.7)
plt.scatter(n_trials, rt_s2_fin, color = "#864AF9", alpha = 0.7)
plt.scatter(n_trials, rt_s1_fin, color = "#65B741")
plt.xscale('log')
plt.yscale('log')
plt.show()

#%% Violinplot of reaction times  version 1

fig, (ax_violin, ax_dist) = plt.subplots(ncols=2, figsize=(12, 6), gridspec_kw={"width_ratios": [3, 1]})

# Create a violin plot
sns.violinplot(data=rt_s3, ax=ax_violin, color='#D24545')

# Add a density plot on the side
sns.distplot(rt_s3, kde=True, ax=ax_dist, color='#D24545', label='Density')

# Set labels and title
ax_violin.set_xlabel('Values')
ax_violin.set_ylabel('Density')

ax_dist.set_xlabel('Reaction times')
ax_dist.set_ylabel('')

plt.suptitle("Reaction times: Subject 3")

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()


#%% BOXPLOT WITH DATAPOINTS AND DENSITY DISTRIBUTION PER SUBJECT
#Combine dataframes

SUBS = ["Subject 1", "Subject 2", "Subject 3", "Subject 4"]

df_rt_s1 = pd.DataFrame({'reaction_time': rt_s1_fin}) #, 'subject': 'Subject 1'})
df_rt_s1['subject'] = 'Subject 1'
df_rt_s2 = pd.DataFrame({'reaction_time': rt_s2_fin}) #, 'subject': 'Subject 1'})
df_rt_s2['subject'] = 'Subject 2'
df_rt_s3 = pd.DataFrame({'reaction_time': rt_s3_fin}) #, 'subject': 'Subject 1'})
df_rt_s3['subject'] = 'Subject 3'
df_rt_s4 = pd.DataFrame({'reaction_time': rt_s4_fin}) #, 'subject': 'Subject 1'})
df_rt_s4['subject'] = 'Subject 4'

combined_df = pd.concat([df_rt_s4, df_rt_s3, df_rt_s2, df_rt_s1])

COLORS = ["#86A7FC", "#D24545", "#864AF9", "#65B741" ]

boxplot_data = [
    combined_df[combined_df["subject"] == subjects]["reaction_time"].values 
    for subjects in SUBS]

# The style of the line that represents the median.
medianprops = {"linewidth": 1.5, "color": "#607274", "solid_capstyle": "butt"}
# The style of the box ... This is also used for the whiskers
boxprops = {"linewidth": 1.5, "color": "#607274"}


fig, ax = plt.subplots(figsize=(10, 7))
pt.half_violinplot( y="subject",  x="reaction_time", data=combined_df, scale ="area", inner= None, palette = COLORS, width=1, ax=ax, alpha = 0.5);
for i, subjects in enumerate(SUBS):
    # Subset the data
    data = combined_df[combined_df["subject"] == subjects]
    
    # Jitter the values on the vertical axis
    y = i + np.random.uniform(high=0.2, size=len(data))
    
    # Select the values of the horizontal axis
    x = data["reaction_time"]
    
    # Add the rain using the scatter method.
    ax.scatter(x, y, color=COLORS[i], alpha=0.6)

SHIFT = 0.15
POSITIONS = [0 + SHIFT, 1 + SHIFT, 2 + SHIFT, 3 + SHIFT]

ax.boxplot(
    boxplot_data, 
    vert=False, 
    positions=POSITIONS, 
    manage_ticks=False,
    showfliers = False, # Do not show the outliers beyond the caps.
    showcaps = False,   # Do not show the caps
    medianprops = medianprops,
    whiskerprops = boxprops,
    boxprops = boxprops
)

ax.tick_params(labelsize=15)


#%% Other plots

#Plot of slopes
slope_all = [slope_s4, slope_s3, slope_s2, slope_s1]
std_err_all = [std_err_s4, std_err_s3, std_err_s2, std_err_s1]

plt.figure(figsize=(8, 4))
# Create a bar plot
sns.barplot(SUBS, slope_all,  palette = COLORS, yerr = std_err_all)
# Set labels and title
plt.xlabel('Subjects')
plt.ylabel('Values')
plt.xticks(size=12)
plt.yticks(size=12)
# Show plot
plt.show()


#%% ANOVA 

SUBS = ["Subject 1", "Subject 2", "Subject 3", "Subject 4"]
df_rt_s1 = pd.DataFrame({'reaction_time': rt_s1}) 
df_rt_s1['subject'] = 'Subject 1'
df_rt_s2 = pd.DataFrame({'reaction_time': rt_s2})
df_rt_s2['subject'] = 'Subject 2'
df_rt_s3 = pd.DataFrame({'reaction_time': rt_s3})
df_rt_s3['subject'] = 'Subject 3'
df_rt_s4 = pd.DataFrame({'reaction_time': rt_s4}) 
df_rt_s4['subject'] = 'Subject 4'


combined_df_anova = pd.concat([df_rt_s4, df_rt_s3, df_rt_s2, df_rt_s1])

combined_df_anova['trial'] = np.tile(np.arange(1, 101), 4)

# Perform repeated measures ANOVA
anova_table = AnovaRM(combined_df_anova, 'reaction_time', 'trial', within=['subject']).fit()

# Print ANOVA results
print(anova_table)

# Perform Tukey's HSD test
tukey_results = pairwise_tukeyhsd(combined_df_anova['reaction_time'], combined_df_anova['subject'])

# Print Tukey's HSD results
print(tukey_results)