# Car Dheko - Used Car Price Prediction

## Project Overview
This project aims to enhance the customer experience and streamline the pricing process for used cars by leveraging machine learning. We developed an accurate and user-friendly Streamlit-based web application that predicts the prices of used cars based on various features. This tool is designed for seamless use by both customers and sales representatives.

## Skills Acquired
- **Data Cleaning and Preprocessing**
- **Exploratory Data Analysis (EDA)**
- **Machine Learning Model Development**
- **Price Prediction Techniques**
- **Model Evaluation and Optimization**
- **Model Deployment**
- **Streamlit Application Development**
- **Documentation and Reporting**

## Domain
- Automotive Industry
- Data Science
- Machine Learning

## Problem Statement

### Objective
To create an interactive web application that predicts the prices of used cars based on historical data. The application is tailored for customers and sales representatives, providing real-time price predictions.

### Project Scope
The dataset includes historical used car data from CarDekho, including features like make, model, year, fuel type, transmission type, and city. The project focuses on:
- Developing a robust machine learning model.
- Deploying the model in a Streamlit application.
- Ensuring user-friendly interaction and accurate predictions.

## Approach

### Data Processing
1. **Import and Concatenate**:
   - Import and merge datasets for all cities.
   - Add a column for ‘City’ to identify the source of each row.
2. **Handling Missing Values**:
   - Impute missing numerical values using mean, median, or mode.
   - Handle missing categorical values by mode imputation or introducing a new category.
3. **Standardizing Data Formats**:
   - Convert units (e.g., "70 kms" to integers) and standardize data types.
4. **Encoding Categorical Variables**:
   - Use one-hot encoding for nominal variables.
   - Use label or ordinal encoding for ordinal variables.
5. **Normalizing Numerical Features**:
   - Apply Min-Max Scaling or Standard Scaling for selected algorithms.
6. **Outlier Removal**:
   - Detect and handle outliers using IQR or Z-score methods.

### Exploratory Data Analysis (EDA)
- **Descriptive Statistics**: Summarize data distribution with mean, median, and standard deviation.
- **Visualization**: Use scatter plots, histograms, box plots, and heatmaps to uncover patterns.
- **Feature Selection**: Identify impactful features through correlation analysis, feature importance, and domain knowledge.

### Model Development
1. **Train-Test Split**: Split data into training (70-80%) and testing sets.
2. **Model Selection**:
   - Evaluate algorithms like Linear Regression, Random Forest, and Gradient Boosting Machines.
3. **Model Training**:
   - Use cross-validation for robust performance.
4. **Hyperparameter Tuning**: Optimize model parameters using Grid or Random Search.

### Model Evaluation
- Use metrics such as:
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - R-squared
- Compare models to select the best performer.

### Optimization
- **Feature Engineering**: Enhance features using domain knowledge.
- **Regularization**: Apply Lasso (L1) and Ridge (L2) techniques to prevent overfitting.

### Deployment
- **Streamlit Application**:
  - Interactive UI for real-time predictions.
  - Allow users to input car details and obtain price estimates.
- **Design**:
  - Ensure intuitive navigation and error handling.

## Results
- A functional machine learning model for accurate price predictions.
- Comprehensive analysis and visualizations of the dataset.
- A deployed Streamlit application for real-time use.

## Project Evaluation Metrics
1. **Model Performance**:
   - MAE, MSE, R-squared.
2. **Data Quality**:
   - Completeness and accuracy of preprocessed data.
3. **Application Usability**:
   - User satisfaction and feedback.
4. **Documentation**:
   - Clarity and completeness of reports and code comments.

## Deliverables
- Source code for preprocessing and model development.
- Documentation detailing methodology, models, and evaluation.
- Visualizations and analysis reports from EDA.
- Deployed Streamlit application.
- User guide explaining the approach and model selection.

## Technical Tags
- Data Preprocessing
- Machine Learning
- Price Prediction
- Regression
- Python
- Pandas
- Scikit-Learn
- Exploratory Data Analysis (EDA)
- Streamlit
- Model Deployment

## Dataset
- The dataset includes multiple Excel files with details of used cars from various cities.
- Preprocessing steps involve handling missing values, standardization, encoding, and normalization.



## Authors
- **Praveen R**
- [GitHub Profile](https://github.com/Praveen-R-22)



For any issues or inquiries, please contact the author or raise an issue in the repository.

# Car-Dheko---Used-Car-Price-Prediction
