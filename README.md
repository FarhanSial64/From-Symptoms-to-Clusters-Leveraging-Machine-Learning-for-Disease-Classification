# Disease Clustering and Evaluation

## Project Overview

This project leverages machine learning and natural language processing (NLP) techniques to perform disease clustering and evaluate the effectiveness of different feature encoding methods. The dataset includes information about various diseases, along with their associated risk factors, symptoms, signs, and subtypes. The project explores two main encoding strategies: **TF-IDF** (Term Frequency-Inverse Document Frequency) and **One-Hot Encoding**.

The goal is to:
1. Compare the sparsity and performance of these encoding methods.
2. Visualize the diseases in a reduced dimensionality space using techniques like PCA and SVD.
3. Evaluate classification models (K-Nearest Neighbors and Logistic Regression) on these encoded features.
4. Generate insights on how well the clusters align with real-world disease categories and discuss the advantages and limitations of both encoding methods.

## Project Structure

- **`disease_clustering.ipynb`**: This Jupyter Notebook file contains the entire data preprocessing, model evaluation, and performance reporting pipeline. It includes:
  - **Data Loading and Preprocessing**
  - **TF-IDF and One-Hot Encoding**
  - **Sparsity Analysis**
  - **Dimensionality Reduction using PCA and SVD**
  - **Model Evaluation (KNN and Logistic Regression)**

- **`disease_clustering_streamlit.py`**: This file is a Streamlit application to visualize the analysis interactively. It includes:
  - File upload for datasets.
  - Sparsity comparison visualization.
  - Dimensionality reduction visualization.
  - Interactive model evaluation and performance display.
  - A report button to explain TF-IDF vs One-Hot Encoding and its clinical relevance.

## Installation Instructions

To get started with the project, follow these steps:

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/disease-clustering.git
   cd disease-clustering
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   `requirements.txt` contains the following libraries:
   ```txt
   streamlit==1.8.0
   pandas==1.3.4
   numpy==1.21.2
   scikit-learn==0.24.2
   matplotlib==3.4.3
   seaborn==0.11.2
   scipy==1.7.1
   ast==1.4
   ```

## How to Use

### Jupyter Notebook

1. Open the `disease_clustering.ipynb` file in Jupyter Notebook:
   ```bash
   jupyter notebook disease_clustering.ipynb
   ```

2. Follow the steps in the notebook for data loading, encoding, dimensionality reduction, and model evaluation.

3. The notebook will output a variety of visualizations and performance metrics, including:
   - Sparsity of the feature matrices (TF-IDF vs One-Hot).
   - Dimensionality reduction visualizations using PCA and SVD.
   - Evaluation results for K-Nearest Neighbors and Logistic Regression.

### Streamlit Application

1. To run the Streamlit app, navigate to the project directory and execute the following command:
   ```bash
   streamlit run disease_clustering_streamlit.py
   ```

2. The app will open in your default web browser. You can:
   - Upload the `disease_features.csv` and `encoded_output2.csv` files.
   - View the sparsity comparison and model evaluation metrics.
   - Visualize the data after dimensionality reduction.
   - Generate a detailed report comparing TF-IDF and One-Hot encoding and discussing their clinical relevance.

## Key Features

1. **Data Preprocessing**: The dataset is cleaned and transformed, with textual columns (risk factors, symptoms, signs, subtypes) being vectorized using both TF-IDF and One-Hot Encoding techniques.

2. **Sparsity Comparison**: The sparsity of both encoding methods is compared and visualized to assess their effectiveness and impact on the model's performance.

3. **Dimensionality Reduction**: Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) are applied to reduce the dimensionality of the encoded data and provide visual insights into how well the diseases are clustered.

4. **Model Evaluation**: Multiple classification models (K-Nearest Neighbors and Logistic Regression) are evaluated using cross-validation, and their performance is compared on both TF-IDF and One-Hot encoded features.

5. **Streamlit App**: The app offers an interactive interface for file uploads, model evaluation, and visualization, making it easy to explore the results and insights.

## Clinical Relevance

- **TF-IDF vs One-Hot Encoding**: The report generated by the Streamlit app highlights the advantages and limitations of both encoding methods. TF-IDF may outperform One-Hot Encoding in capturing the importance of certain words or features, especially when the dataset contains a significant number of rare or unique terms. However, One-Hot Encoding is simple to implement and may work well when the features are categorical and equally important.
  
- **Disease Clustering**: The dimensionality reduction visualizations show how well the diseases are clustered using these encoding methods. The diseases are color-coded based on their clinical category (e.g., Cardiovascular, Neurological), providing insights into whether TF-IDF clusters align with real-world disease categories.

## Limitations

- **Feature Representation**: While TF-IDF and One-Hot Encoding are widely used, they have limitations in terms of capturing relationships between features. More advanced techniques like Word2Vec or BERT may provide better results for more complex datasets.

- **Model Choice**: K-Nearest Neighbors and Logistic Regression are relatively simple models. More advanced models such as Random Forests, Support Vector Machines, or Neural Networks could potentially yield better performance.

## Conclusion

This project provides a comprehensive analysis of disease clustering using machine learning and NLP techniques. By comparing different encoding methods and evaluating various models, it offers valuable insights into how different feature representations impact model performance. The Streamlit app adds interactivity, allowing users to explore the results in a user-friendly manner.

## Future Work

- **Advanced Feature Encoding**: Implementing advanced NLP techniques like Word2Vec or BERT to better capture semantic relationships between features.
- **Model Improvement**: Evaluating more advanced models like Random Forests, XGBoost, or Neural Networks.
- **Real-World Application**: Integrating this analysis into a healthcare application for real-time disease classification and recommendation.

