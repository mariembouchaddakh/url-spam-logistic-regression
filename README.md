# URL Spam Detection with Logistic Regression

University machine learning project: detecting spam URLs from numeric features extracted from the URL (length, token counts, character counts, and similar), using logistic regression on the ISCX-URL-2016 dataset.

The notebook (`detection_spam_regression_logistique.ipynb`) is written in French.

## Dataset

- **ISCX-URL-2016**, Canadian Institute for Cybersecurity: <https://www.unb.ca/cic/datasets/url-2016.html>. The file `All.csv` is not included in this repository; download it from that page and place it next to the notebook.
- 36,707 URLs, 80 columns, five classes: Defacement, benign, phishing, malware, spam.
- The task is reduced to a **binary problem**: spam (1) against all other classes (0). Spam is 18.2% of the data (6,698 URLs).

## Method

1. **Exploration**: class distribution, missing values, feature distributions, correlation heatmap.
2. **Preprocessing**: infinite values set to NaN, stratified 80/20 train/test split (`random_state=42`), NaN replaced by the median computed on the training set only, `StandardScaler` fitted on the training set only.
3. **Baseline**: `LogisticRegression` (C=1.0, L2, lbfgs).
4. **Tuning**: `GridSearchCV` over 15 combinations (C, penalty, solver), 5 folds, scored on F1.
5. **Class imbalance**: SMOTE applied to the training set only, then compared with the other two models.
6. **Evaluation**: accuracy, precision, recall, F1, confusion matrix, ROC curve and AUC, 5-fold cross-validation, coefficient analysis.

## Results (test set, 7,342 URLs)

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline | 0.9594 | 0.9008 | 0.8739 | 0.8871 | 0.9731 |
| GridSearchCV (C=100, L1, liblinear) | 0.9613 | 0.9037 | 0.8821 | 0.8927 | 0.9750 |
| SMOTE + tuned | 0.9350 | 0.7561 | 0.9507 | 0.8423 | 0.9756 |

- The tuned model has a cross-validated F1 of 0.898 (standard deviation 0.0073).
- SMOTE raises recall (missed spam drops from 158 to 66) but lowers precision (false alarms rise from 126 to 411), so the F1 goes down. The AUC barely changes: SMOTE shifts the trade-off rather than improving the model. Which model is better depends on the cost of a missed spam compared with a false alarm.

## Limitations

- The binary reduction (spam against everything else) hides the differences between the other four classes.
- Only one linear model and one dataset are used; no other model families were compared.
- The decision threshold is left at 0.5; tuning it would be a simpler alternative to SMOTE for trading precision against recall.

## Run it

```bash
pip install -r requirements.txt
jupyter notebook detection_spam_regression_logistique.ipynb
```

Put `All.csv` in the same folder as the notebook first.
