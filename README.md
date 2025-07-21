# Credit Risk Detection - Machine Learning Project

This repository contains code for the **Credit Risk Detection** project using various machine learning models. The goal of this project is to predict the likelihood of credit risk based on a dataset from Kaggle.

## Overview

The project involves preprocessing a Kaggle dataset and training 5 different machine learning models to predict credit risk. The models we compare are:

- **Random Forest**
- **Neural Network**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Decision Trees**

Through this comparison, we evaluate and determine the best-performing model for the task.

## Project Details

This is a **CS 725 - FML** course project. The main objective of this project was to analyze credit risk using a Kaggle dataset and apply machine learning models to predict whether a person is at risk of defaulting on their credit.

The models were trained on the dataset, and the best results were observed from the **Random Forest classifier**, which achieved an accuracy of up to **93%**. We also explored preprocessing techniques and various hyperparameters tuning.

## Contributors

- **Aditya Hol** (24M0778)
- **Aryan Khilwani** (24M0781)
- **Tushar Katakiya** (24M2110)

## Files and Structure

- **Preprocessing Notebook**: Jupyter Notebooks with code for preprocessing.
- **Models**: Custom models made by our team.
- **Model Training**: Code for training the 5 models.
- **Plots**: Visualizations of model performance and learning curves.

## Steps to Run the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/credit-risk-detection.git
   ```
2. Install the required dependencies.
3. Preprocess the data:
   - Run the `datapreprocessing.ipynb` script to load and clean the dataset.
4. Train the models:
   - Use the provided scripts to train the models (Random Forest, SVM, Neural Network, Logistic Regression, Decision Trees).

## Model Performance

- **Random Forest** achieved the best performance with an accuracy of up to **93%**.
- **Logistic Regression**, **SVM**, and **Decision Trees** performed reasonably well but showed lower accuracy than the Random Forest model.

## Comments

- **SVMs.py** contains our implementation of the SVM model but the RBF kernel is not optimized for performance hence takes very long time or makes a very large memory overhead so for testing purposes we used SKlearn SVM model with RBF kernel.
- **Random Forest** added plots to print the whole decision tree but as the number of estimators and max depth is large so, its quite cluttered and not providing any useful visualization.
