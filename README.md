# Achieved Top 5% ranking with a score of 79.186

## Project Summary
This project explores feature engineering and various machine learning models to predict passenger survival on the Titanic based on Kaggle's well-known Titanic dataset.

## Data Preprocessing

- **Data Cleaning**: The data underwent several preprocessing steps to handle missing values and standardize the feature formats:

  - **Name Title Unification**: Titles extracted from the `Name` column were unified to standardize categories such as Mr, Mrs, Miss, etc. This unification helped in capturing social status, which was an important predictor of survival.

  - **Handling Missing Values**: Missing values in the `Age` and `Fare` columns were filled using appropriate strategies:

    - `Age`: Missing Age values were filled using the median age for passengers based on `Pclass`, `Sex`, and other relevant attributes. This ensured that the imputed values were representative of each group.

    - `Fare`: Missing Fare values were filled with the median fare for passengers in the same `Pclass` to maintain consistency and account for economic status.

  - **Cabin Letters**: The `Cabin` column was used to extract the first letter, representing the deck. Missing cabin values were assigned as `Cabin_Letter_Unknown` to distinguish passengers without cabin information.

- **Feature Engineering**: Created new features like `family size`, `name titles`, `age bins`, `cabin number bins`, `fare bins`, and `cabin letters`. Feature engineering was crucial for capturing important aspects of the data that were not directly available. For instance, extracting titles from passenger names provided insights into social status, which was a significant predictor of survival.
  - **Age and Fare Binning**: Both Age and `Cabin_Number` were binned to reduce noise and simplify the model's learning process. Binning `Age` into categories like Child, Teen, Young Adult, Adult, and Senior was particularly useful because children were more likely to survive, as seen in visualizations and historical accounts. `Cabin_Number` was also binned to provide better generalization without overfitting specific cabin numbers.
    ![image](https://github.com/user-attachments/assets/f4209b11-7c63-41a6-bed0-7004f64ad52c)

  - **Categorical Features**: One-hot encoding was applied to categorical features like `Cabin_Letter`, `Name_Title`, and `Embarked` to allow the model to understand these non-numeric features. Categories like `Cabin_Letter_Unknown`, `Fare_bin_Very High`, `Age_bin_Child`, and `Sex` were found to be significant predictors based on their impact on survival rates.
    ![image](https://github.com/user-attachments/assets/6288c05c-37a9-42e0-8d58-caae7dad0586)

  - **Interaction Features**: Created interaction features such as `Sex_Pclass` and `Fare_Pclass` to capture combined effects. For instance, `Sex_Pclass` emphasized the influence of passenger class based on gender, highlighting how women in higher classes had a better chance of survival.

  - **Family Size Features**: Added features for Sibling/Spouse count (`SibSp`) and Parent/Child count (`Parch`) to emphasize family connections. Children (represented by higher Parch values) were more likely to be saved, as seen in visualizations and historical accounts, making these features crucial for predicting survival.
    ![image](https://github.com/user-attachments/assets/b253bc31-99b6-4965-88b1-e458eb0287ed)

- **Feature Selection**: To determine which features were most useful, a combination of approaches was used:
  - **Linear Correlation Matrix**: A correlation matrix was created to identify features with strong linear relationships to the target variable. However, correlation alone was not sufficient for identifying all important features.
  - **Bar Charts and Logistic Regression**: Bar charts were created to visually compare the significance of different features, and a basic logistic regression model was used to evaluate the importance of each feature. This helped in selecting features that had the most significant impact on survival, such as passenger class (Pclass), fare, and family size.
    ![image](https://github.com/user-attachments/assets/593e97e6-0805-4354-ac1e-b622c11fb03d)
### Final Features selected:
| Feature                         | Description                                      |
| ---                             | ---                                              |
| `Cabin_Number_Bin_0-20`         | Cabin number range from 0 to 20                  |
| `Cabin_Number_Bin_21-40`        | Cabin number range from 21 to 40                 |
| `Cabin_Number_Bin_41-60`        | Cabin number range from 41 to 60                 |
| `Cabin_Number_Bin_61-80`        | Cabin number range from 61 to 80                 |
| `Cabin_Number_Bin_81-100`       | Cabin number range from 81 to 100                |
| `Cabin_Number_Bin_101-140`      | Cabin number range from 101 to 140               |
| `Age_bin_Child`                 | Binned age group representing children           |
| `Age_bin_Teen`                  | Binned age group representing teenagers          |
| `Age_bin_Young Adult`           | Binned age group representing young adults       |
| `Age_bin_Adult`                 | Binned age group representing adults             |
| `Age_bin_Senior`                | Binned age group representing seniors            |
| `Pclass`                        | Passenger class (1st, 2nd, 3rd)                  |
| `Sex`                           | Gender of the passenger                          |
| `Fare`                          | Ticket fare                                      |
| `Name_Title_Miss`               | Passenger title indicating `Miss`                |
| `Name_Title_Mr`                 | Passenger title indicating `Mr`                  |
| `Name_Title_Mrs`                | Passenger title indicating `Mrs`                 |
| `Name_Title_Master`             | Passenger title indicating `Master`              |
| `SibSp`                         | Number of siblings or spouses aboard             |
| `Parch`                         | Number of parents or children aboard             |
| `Cabin_Number`                  | Cabin number                                     |
| `Fare_Pclass`                   | Interaction feature between fare and class       |
| `Sex_Pclass`                    | Interaction feature between sex and class        |
| `Sex_Fare`                      | Interaction feature between sex and fare         |
| `Embarked_C`                    | Embarked at port C                               |
| `Embarked_S`                    | Embarked at port S                               |
| `Name_Length`                   | Length of the passenger's name                   |
| `Family_Size_Category_Medium`   | Medium-sized family category                     |
| `Cabin_Letter_Unknown`          | Cabin letter unknown                             |
| `Fare_bin_Very High`            | Binned fare representing very high fares         |

## Models
- **Logistic Regression (LR)**: A statistical method used for binary classification, providing insights into which features influence survival.

- **Decision Trees (DT)**: A tree-like model of decisions, aimed at capturing non-linear relationships between features and survival.

- **Random Forest (RF)**: Similar to decision trees, is made of many trees. Robust and reliable for classification.

- **Neural Networks (NN)**: A more complex model used to learn intricate relationships within the data, increasing the prediction accuracy. But not so ideal for this question.

## Model Optimization
- **Model Selection and Evaluation**: Compared multiple models to select the best-performing one based on accuracy and cross-validation scores.

  A **Random Forest** model was chosen as the final model.
  A local result after using Random Forest (average of 5 cross validations)
  Metric | Class 0 | Class 1 | Average
  | -----|-------- | --------|----- |
  Precision | 0.83 | 0.81 | 0.82 (weighted)
  Recall | 0.88 | 0.74 | 0.82 (accuracy)
  F1-Score | 0.85 | 0.77 | 0.82 (weighted)
  Support | 105 | 74 | 179
  
- **Hyperparameter Tuning**: Adjusted hyperparameters for Logistic Regression, Decision Trees, and Neural Networks to enhance model performance.

- **Bagging for Stability**: To further improve the stability and performance of the Random Forest model, bagging (an ensemble technique) was used. Bagging helped reduce variance and ensured that the model was less sensitive to fluctuations in the training data, leading to more robust predictions.


## Appendix
![image](https://github.com/user-attachments/assets/3453f5c6-40fa-47f2-8568-44cf4b49b67f)
