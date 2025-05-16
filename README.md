This repository contains the code and analysis for interpreting machine learning models trained on the **UCI Adult Income Dataset**, focusing on **feature importance** and **fairness** through **SHAP** and **LIME** explanations.

🔗 **Colab Notebook**: [Open Code in Google Colab](https://colab.research.google.com/drive/1Z7W0qFivWZGqpEkidjocdFj1qNFtqiYG?usp=sharing)

---

## 📁 Dataset Overview

- **Total Features**: 12
- **Target Variable**: `income`
  - `1` → Income > $50K
  - `0` → Income ≤ $50K

| Feature             | Type        | Description                                      |
|---------------------|-------------|--------------------------------------------------|
| Age                 | Continuous  | Age in years                                     |
| Workclass           | Categorical | Type of employment                               |
| Education-Num       | Continuous  | Education level (numeric)                        |
| Marital Status      | Categorical | Marital status                                   |
| Occupation          | Categorical | Job type                                         |
| Relationship        | Categorical | Household relationship                           |
| Race                | Categorical | Ethnicity                                        |
| Sex                 | Categorical | Gender                                           |
| Capital Gain        | Continuous  | Income from investments                          |
| Capital Loss        | Continuous  | Capital lost                                     |
| Hours per week      | Continuous  | Weekly working hours                             |
| Country             | Categorical | Country of residence                             |

---

## 🧪 Model Training

Two classifiers were trained to predict income:
- **Gradient Boosting**
  - Train Accuracy: `~0.869`
  - Test Accuracy: `~0.868`
- **Multi-Layer Perceptron (MLP)**
  - Train Accuracy: `~0.837`
  - Test Accuracy: `~0.840`

---

## 🔍 Interpretability with SHAP and LIME

### 🔹 Global Feature Insights

| Feature          | Gradient Boosting (SHAP)             | MLP (SHAP)                         |
|------------------|--------------------------------------|------------------------------------|
| Age              | Strongest around age 50              | Decreasing importance with age     |
| Hours per week   | Positive correlation with >$50K      | Inverse trend due to NN structure  |

| Feature          | Gradient Boosting (LIME)             | MLP (LIME)                         |
|------------------|--------------------------------------|------------------------------------|
| Age              | Varies by age group                  | Mixed influence                    |
| Hours per week   | >45 hours → positive contribution    | More hours → positive contribution |

> SHAP shows smooth, non-linear feature effects, while LIME gives interpretable local approximations by breaking continuous features into intervals.

---

### 🔸 Single Sample Explanation

| Model                | Positive Contributor  | Negative Contributor      |
|----------------------|-----------------------|----------------------------|
| SHAP - GB            | Hours per week        | Relationship               |
| SHAP - MLP           | Hours per week        | Relationship               |
| LIME - GB            | Hours per week        | Capital Gain               |
| LIME - MLP           | Country               | Capital Gain               |

> **Hours per week** is a consistent positive contributor. Features like **Relationship** and **Capital Gain** often appear as negative influencers, varying across models and methods.

---

## ✨ Key Takeaways

- **Feature importance** changes between models (GB vs. MLP) and interpretability tools (SHAP vs. LIME).
- **SHAP** provides global and nuanced insight; **LIME** offers local and sample-specific explanation.
- The choice of model architecture influences how features are internally represented and interpreted.
- **"Hours per week"** is a dominant feature for predicting high income across both methods and models.

---

## 🛠 Technologies Used

- Python
- scikit-learn
- SHAP
- LIME
- Google Colab

---

## 📬 Contact

For questions, suggestions, or collaborations, feel free to reach out or fork the repo!

