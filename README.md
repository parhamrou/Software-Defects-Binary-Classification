# Software-Defects-Binary-Classification
This repository contains the code for the Kaggle competition [Predicting defects in C programs](https://www.kaggle.com/competitions/playground-series-s3e23/overview). The goal of the competition is to predict defects in C programs given various attributes of the code.

## Dataset

The dataset used for this competition can be found [here](https://www.kaggle.com/competitions/playground-series-s3e23/data). It consists of two files: `train.csv` and `test.csv`. The `train.csv` file contains the labeled training data, while the `test.csv` file contains the unlabeled test data.

## Getting Started

To run the code in this repository, follow these steps:

1. Clone the repository to your local machine.
2. Install the required libraries by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Download the dataset from the competition page and place the `train.csv` and `test.csv` files in the repository's root directory.

## Code Structure

The code is organized into the following sections:

1. **Importing libraries and loading dataset**: This section installs the required libraries and loads the dataset into pandas DataFrames.

2. **EDA**: This section performs exploratory data analysis on the dataset, including examining the data, checking for missing values, and transforming the target variable.

3. **Model selection**: This section selects the best model among the five models mentioned: Random Forest Classifier, Multinomial Naive Bayes, KNN, Logistic Regression, and XGB Classifier. It uses KFold cross-validation and Randomized Search CV to evaluate the performance of each model and find the best hyperparameters.

4. **Applying the best model to the test data**: This section applies the selected XGB Classifier model to the test data and predicts probabilities for each instance.

5. **Explanation of Techniques and Models**:

   - **KFold Cross Validation**: This technique helps us evaluate the performance of each model by splitting the data into K folds and training the model K times, each time using a different fold as the validation set. This allows us to get a more reliable estimate of the model's performance.

   - **Randomized Search CV**: This technique helps us find the best hyperparameters for each model. It performs a randomized search over a predefined hyperparameter space and evaluates the model's performance using cross-validation. By trying different combinations of hyperparameters, we can find the set of parameters that gives the best performance for each model.

   - **Random Forest Classifier**: This model is an ensemble learning method that combines multiple decision trees to make predictions. It is known for its ability to handle complex datasets and provide good accuracy.

   - **Multinomial Naive Bayes**: This model is based on the Bayes' theorem and is commonly used for text classification tasks. It assumes that the features are conditionally independent given the class and uses probability distributions to make predictions.

   - **KNN (K-Nearest Neighbors)**: This model is a non-parametric algorithm that classifies new instances based on the majority vote of their k nearest neighbors in the training data. It is simple and intuitive but can be computationally expensive for large datasets.

   - **Logistic Regression**: This model is a linear classifier that uses a logistic function to model the probability of a binary outcome. It is widely used for binary classification tasks and can handle both numerical and categorical features.

   - **XGB Classifier (Extreme Gradient Boosting Classifier)**: This model is an implementation of gradient boosting that uses a combination of weak learners (decision trees) to make predictions. It is known for its high performance and ability to handle large datasets.

## Results

At the end of the competition, the code in this repository achieved a ROC_AUC score of 0.79344. Out of 1704 participants, it secured a position of 370, with a score that was only 0.00085 less than the winning score.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
