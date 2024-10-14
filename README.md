# Achieved Top 5% ranking with a score of 79.186

## Project Summary
This project explores feature engineering and various machine learning models to predict passenger survival on the Titanic based on Kaggle's well-known Titanic dataset.

## Models
- **Logistic Regression (LR)**: A statistical method used for binary classification, providing insights into which features influence survival.

- **Decision Trees (DT)**: A tree-like model of decisions, aimed at capturing non-linear relationships between features and survival.

- **Random Forest (RF)**: Similar to decision trees, is made of many trees. Robust and reliable for classification.

- **Neural Networks (NN)**: A more complex model used to learn intricate relationships within the data, increasing the prediction accuracy. But not so ideal for this question.

## Key Techniques Used

- **Data Cleaning**: The data underwent several preprocessing steps to handle missing values and standardize the feature formats:

  - **Name Title Unification**: Titles extracted from the `Name` column were unified to standardize categories such as Mr, Mrs, Miss, etc. This unification helped in capturing social status, which was an important predictor of survival.

  - **Handling Missing Values**: Missing values in the `Age` and `Fare` columns were filled using appropriate strategies:

    - `Age`: Missing Age values were filled using the median age for passengers based on `Pclass`, `Sex`, and other relevant attributes. This ensured that the imputed values were representative of each group.

    - `Fare`: Missing Fare values were filled with the median fare for passengers in the same `Pclass` to maintain consistency and account for economic status.

  - **Cabin Letters**: The `Cabin` column was used to extract the first letter, representing the deck. Missing cabin values were assigned as `Cabin_Letter_Unknown` to distinguish passengers without cabin information.

- **Feature Engineering**: Created new features like `family size`, `name titles`, `age bins`, `cabin number bins`, `fare bins`, and `cabin letters`. Feature engineering was crucial for capturing important aspects of the data that were not directly available. For instance, extracting titles from passenger names provided insights into social status, which was a significant predictor of survival.
  - **Age and Fare Binning**: Both Age and Cabin_Number were binned to reduce noise and simplify the model's learning process. Binning Age into categories like Child, Teen, Young Adult, Adult, and Senior was particularly useful because children were more likely to survive, as seen in visualizations and historical accounts. Cabin_Number was also binned to provide better generalization without overfitting on specific cabin numbers.

  - **Categorical Features**: One-hot encoding was applied to categorical features like `Cabin_Letter`, `Name_Title`, and `Embarked` to allow the model to understand these non-numeric features. Categories like `Cabin_Letter_Unknown`, `Fare_bin_Very High`, `Age_bin_Child`, and `Sex` were found to be significant predictors based on their impact on survival rates.

  - **Interaction Features**: Created interaction features such as `Sex_Pclass` and `Fare_Pclass` to capture combined effects. For instance, `Sex_Pclass` emphasized the influence of passenger class based on gender, highlighting how women in higher classes had a better chance of survival.

  - **Family Size Features**: Added features for Sibling/Spouse count (`SibSp`) and Parent/Child count (`Parch`) to emphasize family connections. Children (represented by higher Parch values) were more likely to be saved, as seen in visualizations and historical accounts, making these features crucial for predicting survival.

- **Feature Selection**: To determine which features were most useful, a combination of approaches was used:
  - **Linear Correlation Matrix**: A correlation matrix was created to identify features with strong linear relationships to the target variable. However, correlation alone was not sufficient for identifying all important features.
  - **Bar Charts and Logistic Regression**: Bar charts were created to visually compare the significance of different features, and a basic logistic regression model was used to evaluate the importance of each feature. This helped in selecting features that had the most significant impact on survival, such as passenger class (Pclass), fare, and family size.

- **Model Selection and Evaluation**: Compared multiple models to select the best-performing one based on accuracy and cross-validation scores. A **Random Forest** model was chosen as the final model.

- **Hyperparameter Tuning**: Adjusted hyperparameters for Logistic Regression, Decision Trees, and Neural Networks to enhance model performance.

- **Bagging for Stability**: To further improve the stability and performance of the Random Forest model, bagging (an ensemble technique) was used. Bagging helped reduce variance and ensured that the model was less sensitive to fluctuations in the training data, leading to more robust predictions.
