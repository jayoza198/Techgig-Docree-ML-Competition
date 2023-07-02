# Techgig-Docree-ML-Competition

## Solution Proposed


1. **Data Preprocessing**: Load the training data from a CSV file using `pd.read_csv()`. Remove any unnecessary columns that are not relevant to the classification task. Handle missing values by either dropping rows with missing values or filling them with appropriate values based on the context of the data.

2. **Feature Engineering**: Extract relevant information from the existing features to create new features that might be more informative for the classification task. For example, you can extract features such as the length of the URL or the presence of certain keywords in the URL. This step aims to enhance the predictive power of the model by providing more meaningful features.

3. **Data Encoding**: Encode categorical features using techniques like one-hot encoding or label encoding, depending on the nature of the data and the algorithms being used. This step ensures that categorical features are represented in a suitable format for the machine learning models.

4. **Model Selection**: Choose a set of classification algorithms that are suitable for the problem at hand. Consider a variety of algorithms such as Logistic Regression, Decision Trees, Random Forests, Gradient Boosting, or Support Vector Machines (SVM). Each algorithm has its own strengths and weaknesses, so selecting a diverse set of models can provide better overall performance.

5. **Hyperparameter Tuning**: Perform hyperparameter tuning for each selected model to find the best set of hyperparameters that optimize the model's performance. Use techniques like grid search or random search to explore different combinations of hyperparameters and evaluate their impact on the model's performance using cross-validation.

6. **Model Training and Evaluation**: Train each model on the preprocessed and encoded training data using the optimized hyperparameters. Evaluate the performance of each model using appropriate evaluation metrics such as accuracy, precision, recall, or F1 score. This step helps to identify the models that perform well on the given problem.

7. **Ensemble Learning**: Create an ensemble of the best-performing models by combining their predictions. This can be done using techniques like voting (majority voting or weighted voting), stacking, or boosting. Ensemble methods often lead to improved performance by leveraging the strengths of different models and reducing individual model biases.

8. **Final Model Evaluation**: Evaluate the ensemble model on the test data to assess its generalization performance. Calculate appropriate evaluation metrics to measure the model's accuracy and compare it against other models used in the solution. This step helps to assess the final model's effectiveness in predicting the target variable.

9. **Generating Output**: Use the final ensemble model to make predictions on the provided testing data. Create a submission file with the predicted values in the required format and save it as an output CSV file.

In summary, the proposed solution involves data preprocessing, feature engineering, data encoding, model selection, hyperparameter tuning, model training, ensemble learning, final model evaluation, and generating the output file. This approach aims to leverage the strengths of different models, optimize their performance through hyperparameter tuning, and combine their predictions to achieve a robust and accurate classification model.

## Approach

1. **Data Preprocessing**: The code starts by reading the training data from a CSV file using `pd.read_csv()`. Then, unnecessary columns ('ID', 'USERCITY', 'USERZIPCODE', 'TAXONOMY') are dropped from the DataFrame using the `drop()` function.

2. **Handling Missing Values**: Missing values in the dataset are handled using the `fillna()` function. The missing values in the 'USERPLATFORMUID' column are replaced with the string 'Unknown', and the missing values in the 'IS_HCP' column are replaced with the value 0.

3. **Splitting the Data**: The dataset is split into input features (X) and the target variable (y) using the `train_test_split()` function from scikit-learn. The testing data is set to 20% of the whole dataset, and a random state of 42 is used for reproducibility.

4. **Feature Encoding**: The categorical columns in the dataset ('DEVICETYPE', 'PLATFORM_ID', 'BIDREQUESTIP', 'USERPLATFORMUID', 'USERAGENT', 'PLATFORMTYPE', 'CHANNELTYPE', 'URL', 'KEYWORDS') are one-hot encoded using the `OneHotEncoder()` from scikit-learn. The encoder is fitted on the training data using `fit_transform()`, and the same encoder is applied to the testing data using `transform()`.

5. **Model Training and Evaluation**: Three classification models (Logistic Regression, Decision Tree, Random Forest) are trained on the encoded training data. For each model, the `fit()` function is used to train the model on the encoded training features (X_train_encoded) and the target variable (y_train). Then, the `predict()` function is used to make predictions on the encoded testing features (X_test_encoded). The accuracy of each model is calculated using the `accuracy_score()` function from scikit-learn and printed.

6. **Voting Classifier**: A VotingClassifier is created with the three trained models. The VotingClassifier combines the predictions from the three models using majority voting. The VotingClassifier is fitted on the encoded training data (X_train_encoded, y_train), and predictions are made on the encoded testing data (X_test_encoded) using the `predict()` function. The accuracy of the VotingClassifier is calculated using the `accuracy_score()` function and printed.

Overall, this approach involves preprocessing the data, splitting it into training and testing sets, encoding the categorical features, training multiple classification models, evaluating their individual performances, and combining their predictions using a VotingClassifier. The accuracy of the models and the VotingClassifier is reported as the output.

## Proposed Architecture for ML Pipeline

![Untitled Diagram drawio(3)](https://github.com/jayoza198/Techgig-Docree-ML-Competition/assets/71382456/29937d2b-4725-47ed-aefe-d132fbbf25ba)
