# 🎢 Disneyland Reviews Sentiment Analysis 🎯

## 📖 Project Overview
This project explores sentiment analysis on a dataset of Disneyland reviews. The goal was to build a machine learning pipeline capable of accurately identifying negative feedback, which is crucial for service improvement.

---

## 📊 Phase 1: Initial Analysis & Baseline
We started by analyzing the distribution of ratings. The data showed a significant **class imbalance**:
* **Positive Reviews (4-5 stars):** ~90% 🟢
* **Negative Reviews (1-2 stars):** ~10% 🔴

### The Baseline Model: Naive Bayes
I initially implemented a **Multinomial Naive Bayes** model. While it achieved a high overall **Accuracy (92%)**, the **Confusion Matrix** revealed a major flaw:
* **Recall for Negative Class:** Only **16%** 📉
* **Problem:** The model was "lazy"—it achieved high accuracy simply by predicting "Positive" most of the time, missing 84% of actual customer complaints.

---

## 🔬 Phase 2: The "Balanced" Approach
To solve the recall issue, I pivoted to **Logistic Regression** with a critical adjustment.

### The Rationale
Unlike Naive Bayes, Logistic Regression in `scikit-learn` allows us to use the `class_weight='balanced'` parameter. This tells the model to give more "importance" to the minority class (Negative reviews) during training.

### Comparison of Results
The shift produced a dramatic improvement in the model's ability to catch complaints:

| Metric | Naive Bayes | Logistic Regression (Balanced) |
| :--- | :--- | :--- |
| **Overall Accuracy** | 92% | **93%** |
| **Negative Recall** | 16% | **91%** 🚀 |

---

## 🏆 Conclusion
By choosing a model that supports class weighting, we transformed a model that ignored complaints into a highly sensitive tool that captures **91% of negative sentiment** without sacrificing overall accuracy.
