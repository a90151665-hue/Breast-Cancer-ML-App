 Breast-Cancer-ML-App
 
 Project Overview

This project focuses on building a machine learning–based breast cancer prediction system using the Breast Cancer Wisconsin (Diagnostic) Dataset.
The goal is to predict whether a breast tumor is Malignant or Benign based on various medical features extracted from digitized images.

The project includes:

Exploratory Data Analysis (EDA)
Data visualization and preprocessing
Machine learning model training
Train–test data splitting
A deployed prediction web application


Technologies Used

Programming Language: Python
IDE: Visual Studio Code
Libraries:
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Web Framework: Flask / Streamlit (whichever you used)
Version Control: Git & GitHub

 Target Variable
Column	           Description
diagnosis        	Indicates whether the tumor is malignant or benign
Values	          M = Malignant, B = Benign

For model training:

M → 1
B → 0


Data Preprocessing
The following preprocessing steps were performed:

Data Cleaning
Removed unnecessary columns (e.g., ID column)
Verified absence of missing values

Encoding
Converted categorical target values into numeric format

Feature Scaling
Applied standardization to normalize feature values

Train-Test Split
Dataset split into training and testing sets
Ensures unbiased model evaluation


Project Structure

│── train_model.py

│── trained_model.pkl

├── app.py

├── requirements.txt
 


Conclusion

This project demonstrates a complete machine learning workflow, from data exploration to model deployment.
It highlights the importance of data analysis, preprocessing, and evaluation in building reliable healthcare prediction systems.
