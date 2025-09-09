# MV-CM: Multivariate Convergent Mapping for Gap Filling in Time Series

## Project Description

This project implements the MV-CM method for analyzing and filling gaps in interdependent multivariate time series.  
The method is based on SMAP (Sequential Locally Weighted Global Linear Maps) and uses multivariate lag embeddings to predict missing values, considering relationships between variables.

## Key Features

- Generation of synthetic correlated time series (4 variables).  
- Artificial creation of missing data in one or more variables.  
- Gap filling with the MV-CM algorithm selecting the best subsets of variables (embedding views).  
- Quality evaluation of gap filling via Mean Squared Error (MSE).  
- Visualization of original data, missing points, and imputation results.  
- Parameter flexibility: embedding size, number of neighbors, theta parameter, number of top views.

## Used Libraries

- NumPy  
- Matplotlib  
- scikit-learn (LinearRegression, NearestNeighbors)  
- itertools, datetime (standard library)

## How to Use

1. Clone or download the project repository.  
2. Install dependencies:  pip install numpy matplotlib scikit-learn
3. Run the main script:  python mvcm_analysys.py
4. Modify parameters in the script as needed: number of points, embedding_dim, theta, number of views, etc.

## Algorithm Description

- For each target variable, multivariate embedding vectors are created from other variables.  
- Local regression with distance-dependent weights (SMAP) is performed on neighboring points.  
- The most predictive subsets of variables (views) are selected.  
- Gap filling is done as a weighted prediction over the selected views.

- Embeddings are constructed using lagged observations for each variable subset.  
- Leave-one-out cross validation estimates prediction skill for each view.  
- Weights for views are proportional to prediction skill to combine results.  
- Missing values are imputed sequentially considering the multivariate dynamics.

## Results and Visualization

- Included plots show the correlated variables, highlight missing points, compare original and imputed data, and show error dependence on model settings.

## Contact

Author: Somyacom  
GitHub:  https://github.com/Somyacom
Email: [finutsana@gmail.com](mailto:finutsana@gmail.com)


This project is suitable for researchers dealing with time series data with missing values and inter-variable dependencies.

