# MNIST Digit Recognition with Machine Learning

This repository contains the implementation of a project focused on handwritten digit recognition using the MNIST dataset. The project applies image processing techniques, dimensionality reduction, and machine learning classifiers to improve accuracy in identifying digits from 0 to 9. It was developed as part of the Computational Intelligence course at Ferdowsi University of Mashhad.

This was one of my first hands-on experiences in the Computational Intelligence course, completed during Fall 2024. It helped me explore foundational concepts in image processing, feature extraction, and model optimization.

## Project Overview

The goal of this project is to enhance handwritten digit recognition using classification algorithms like Decision Trees and Support Vector Machines (SVM). Key steps include:
- **Image Filtering**: Applying filters (Sobel, HOG, and a custom sharpen filter) to extract features from images.
- **Centering and Dimensionality Reduction**: Computing the mean image, centering the dataset, and using PCA to reduce dimensions while preserving variance.
- **Hyperparameter Tuning**: Optimizing Decision Tree parameters with Grid Search and K-Fold Cross-Validation to prevent overfitting/underfitting.
- **Model Evaluation**: Training and comparing Decision Tree and SVM models on raw and processed data, evaluating with metrics like Accuracy, Precision, Recall, F1-Score, and Confusion Matrices.
- **Overfitting Mitigation**: Applying pre-pruning techniques on the Decision Tree.

The project uses the MNIST dataset (70,000 grayscale images of 28x28 pixels) and evaluates the impact of each processing phase on final model performance.

## Dataset
- **MNIST**: 60,000 training images and 10,000 test images of handwritten digits (0-9).
- Loaded via `sklearn.datasets.fetch_openml`.

## Project Structure
- **proj-2.ipynb**: The main Jupyter Notebook with code implementation, including functions for filtering, PCA, model training, and evaluation.
- **project2.pdf**: Project description and requirements (in Persian).
- **document-2.pdf**: Detailed documentation of the implementation, results, and analysis (in Persian, authored by group members Elnaz Mohammadi and Zahra Rostami).


## Key Phases and Results
### Phase 1: Image Filtering
- Applied Sobel (edge detection), HOG (texture and geometry), and a custom sharpen filter.
- Combined features (e.g., Sobel + HOG) to create feature vectors.
- Visualized filtered images to observe effects.

### Phase 2: Image Centering and PCA
- Computed mean image and centered the dataset.
- Used Scree Plot to select optimal PCA components (e.g., retaining 80% variance).
- Split data into 80% train and 20% test.

### Phase 3: Decision Tree Hyperparameter Tuning
- Optimized parameters like `max_depth`, `min_samples_split`, `min_samples_leaf` using GridSearchCV with 5-Fold CV.
- Compared with SVM on raw and filtered data.
- Results: SVM outperformed Decision Tree (e.g., ~98% accuracy on raw data vs. ~81% for DT).

### Phase 4: Model Accuracy Analysis
- Computed Precision, Recall, F1-Score per class.
- Visualized Confusion Matrix as Heatmap to identify misclassifications (e.g., 3 and 8 often confused).

### Phase 5: Overfitting and Pruning
- Applied pre-pruning by limiting tree depth and samples per split/leaf.
- Demonstrated reduced overfitting through improved generalization.

## Results Highlights
- **Best Model**: SVM on raw data achieved ~98% test accuracy.
- Filtered data (e.g., Sobel + HOG) showed slight accuracy drops due to potential information loss, but useful for feature emphasis.
- Decision Tree accuracy: ~80-82% across datasets, improved with tuning.
- Insights: Raw data often performed best; filters help in noisy scenarios but may over-simplify for MNIST.

## Usage
- Run cells in `proj-2.ipynb` sequentially to load data, apply filters, train models, and visualize results.
- Models can be saved/loaded using Joblib for reuse.
