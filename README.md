## **Reaction Time Analysis in Touchscreen Task**

### **Author**
- Andrea Fernanda Campos Perez

*April, 2024.*

---
### Languages Used

![Python](https://upload.wikimedia.org/wikipedia/commons/c/c3/Python-logo-notext.svg)

---
### **Overview**
This repository contains a **Python-based data analysis script** for evaluating reaction times in a **touchscreen behavioral task**. The script processes data from multiple subjects and performs statistical analysis to assess performance differences and trends in reaction times.

The repository includes:
- A Python script that loads, cleans, and analyzes reaction time data.
- Statistical analysis using ANOVA and Tukey's HSD test.
- Visualization of reaction time distributions and regression trends.
- Example behavioral data of subjects performing the task.

---

### **Background**
Touchscreen-based behavioral tasks are widely used in cognitive and neuroscientific research to assess **decision-making, attention, and motor response times**. Reaction time (RT) analysis is a key measure to understand **cognitive processing and motor execution speed**.

This study evaluates reaction times from a **touchscreen task** performed by subjects. The analysis includes:
- Calculation of hit accuracy per subject.
- Reaction time distributions.
- Visualize data using **histograms, regression plots, and violin plots**.
- Regression analysis to identify trends in reaction times across trials.
- Group-level statistical comparisons using Repeated Measures ANOVA and Tukey’s HSD test.

---

### **Files in This Repository**
| File | Description |
|------|------------|
| `RT_analysis_behavior.py` | Python script for processing reaction time data and statistical analysis |
| `touchscreen_task_results_sub1.txt` | Raw reaction time data for Subject 1 |
| `touchscreen_task_results_sub2.txt` | Raw reaction time data for Subject 2 |
| `touchscreen_task_results_sub3.txt` | Raw reaction time data for Subject 3 |
| `touchscreen_task_results_sub4.txt` | Raw reaction time data for Subject 4 |

---

### **Results Summary**
- **Hit Accuracy:** Computed for each subject.
- **Reaction Time Distributions:** Visualized using histograms and violin plots.
- **Regression Analysis:** Indicates changes in reaction time trends across trials.
- **ANOVA Results:** Tests for significant differences between subjects.
- **Tukey’s HSD:** Identifies specific differences between pairs of subjects.

These findings provide insights into individual and group-level variations in reaction times within the touchscreen task.

---

### **How to Run the Analysis**
1. Install **Python** (version 3.8 or later).
2. Clone this repository:
   ```sh
   git clone https://github.com/andreaf-campos/RT_behavior_analysis.git
   cd RT_behavior_analysis
   ```
3. Install required Python packages:
   ```sh
   pip install numpy pandas matplotlib seaborn scipy statsmodels ptitprince
   ```
4. Run the analysis script:
   ```sh
   python RT_analysis_behavior.py
   ```
5. View results in generated plots and printed statistical outputs.

---

### **License**
This project is licensed under the MIT License.

If you use this work in academic research or derivative studies, please cite the original author:

*Campos, A. (2024). Reaction Time Analysis in Touchscreen Task.*
