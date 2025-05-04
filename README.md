# FYP
***Title***: Exploratory Analysis of Bank Marketing Campaign Effectiveness Using Python and R

Overview
This project aims to analyse customer data from a Portuguese banking institution to understand the factors influencing a client’s decision to subscribe to a term deposit. Initially planned with Python and SQL for exploratory analysis and visualisation, the approach has evolved to incorporate R for machine learning implementation and predictive modelling. The analysis includes data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning modelling in Python, followed by full-fledged machine learning implementation in R. The findings will provide insights into the effectiveness of bank marketing campaigns and suggest improvements for future strategies.

🎯 Objectives
- Conduct exploratory data analysis (EDA) to understand trends and key indicators.
- Engineer new features to improve model insights and predictive power.
- Apply machine learning models (Logistic Regression, and Random Forest) to forecast customer subscription.
- Evaluate and compare model performance across tools.
- Derive actionable business recommendations based on findings.

🛠 Tools & Technologies
- **Languages**: Python (Jupyter Notebook), R (RStudio)
- **Libraries (Python)**: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imblearn`, `statsmodels`
- **Libraries (R)**: `caret`, `randomForest`, `glmnet`, `ggplot2`, `pROC`, `dplyr`
- **Version Control**: Git & GitHub

📁 Project Structure


📦 Bank-Marketing-EDA-ML-Project

  ├── 📂 week1_initial_data_exploration.ipynb

  ├── 📂 week2_summary_and_variable_inspection.ipynb

  ├── 📂 week3_eda_plots_and_visuals.ipynb

  ├── 📂 week4_feature_engineering.ipynb

  ├── 📂 week5_model_building_python.ipynb

├── 📂 r_code

  │ ├── 01_data_loading_cleaning.R
  
  │ ├── 02_exploratory_analysis.R

  │ ├── 03_feature_engineering.R

  │ ├── 04_logistic_regression.R

  │ ├── 05_random_forest_modeling.R
  
  │ ├── 06_feature_importance.R

  │ ├── FYP_Bank_Analysis.R
  
├── 📂 figures

  │ ├── age_distribution.png
  
  │ ├── balance_vs_subscription_boxplot.png

  │ ├── random_forest_confusion_matrix.png
  
├── README.md


 📅 Weekly Progress Breakdown

| Week | Summary                                                                 |
|------|-------------------------------------------------------------------------|
| 1    | Imported dataset, cleaned data, checked for missing values             |
| 2    | Explored variables, generated summary stats, identified key patterns   |
| 3    | Created visualizations (heatmap, boxplots, bar plots) for deeper EDA   |
| 4    | Engineered new features and handled class imbalance with SMOTE         |
| 5    | Built and evaluated models in Python; later re-implemented in R        |

---

🧠 Key Insights
- Longer call durations significantly increase subscription chances.
- Retired and younger clients tend to subscribe more.
- Previous campaign outcomes strongly influence new campaign success.
- Random Forest achieved the highest AUC (0.96+) in R implementation.

📚 Data Source
- [UCI Machine Learning Repository: Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)


📄 License
This project is for academic purposes and is shared under the [MIT License](LICENSE).

🙏 Acknowledgements
Special thanks to the project supervisor (Dimitrios Airantzis), Birkbeck University of London, and the UCI repository for open data access.


