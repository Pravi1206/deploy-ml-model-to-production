# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a Random Forest Classifier trained to predict whether an individual's income exceeds $50K/year based on census data. The model was developed using scikit-learn's `RandomForestClassifier` with the following hyperparameters:
- Number of estimators: 100
- Maximum depth: 10
- Random state: 42
- Number of jobs: -1 (using all available CPU cores)

## Intended Use

This model is intended for educational and demonstration purposes to showcase a complete ML pipeline from data preprocessing to model deployment. The primary use case is to predict income levels based on demographic and employment-related features.

## Training Data

The model was trained on the Census Income dataset (also known as "Adult" dataset from the UCI Machine Learning Repository). The dataset contains demographic information extracted from the 1994 Census database.

## Evaluation Data

The evaluation data consists of 20% of the original Census Income dataset, randomly split from the full dataset. The same preprocessing steps applied to the training data were applied to the evaluation data using the fitted encoders from the training phase.

## Metrics

The model's performance is evaluated using three primary metrics:

- **Precision:** Measures the proportion of positive predictions that are actually correct (True Positives / (True Positives + False Positives))
- **Recall:** Measures the proportion of actual positive cases that are correctly identified (True Positives / (True Positives + False Negatives))
- **F-beta Score (F1):** Harmonic mean of precision and recall, providing a balanced measure of model performance

## Ethical Considerations

**Bias Concerns:**
- The model is trained on 1994 census data, which may not reflect current demographic distributions
- Historical biases in income inequality based on race, sex, and other protected attributes are present in the training data and may be learned by the model
- The model could potentially perpetuate or amplify existing societal biases if used in decision-making contexts

**Fairness:**
- The model's performance should be evaluated across different demographic groups to ensure fairness
- Features like sex, race, and native-country could lead to discriminatory predictions
- Regular audits for disparate impact across protected groups are recommended if this model were to be used in production

**Privacy:**
- The dataset contains demographic information that could be sensitive
- Proper data anonymization and privacy protections should be maintained

## Caveats and Recommendations

**Limitations:**
- The model is trained on data from 1994, making it potentially outdated for current predictions
- The binary classification of income (<=50K vs >50K) is a simplification that may not capture economic complexity
- Missing values were simply dropped, which could introduce bias if missingness is not random
- The model does not account for temporal changes in economic conditions, inflation, or cost of living

**Recommendations:**
- This model should only be used for educational and demonstration purposes
- For production use, the model should be retrained on current data
- Implement fairness-aware ML techniques to mitigate bias
- Conduct thorough bias testing across demographic subgroups before any real-world application
- Consider using more recent and representative datasets
- Implement monitoring for model drift and performance degradation over time
- Use ensemble methods or more sophisticated models for improved performance if deploying to production
