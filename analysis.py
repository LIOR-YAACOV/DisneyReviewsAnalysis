import pandas as pd
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils import get_model, plot_distribution

if __name__ == "__main__":
    # The dataset isn't UTF-8; fall back to a Windows-friendly encoding.
    df = pd.read_csv("data/DisneylandReviews.csv", encoding="cp1252")
    
    #Analyze the distribution of ratings
    rating_counter = Counter(df["Rating"])
    plot_distribution(
        x_axis=list(rating_counter.keys()),
        y_axis=list(rating_counter.values()),
        x_label="Rating",
        y_label="Count",
        title="Distribution of Ratings",
        filename="ratings_distribution.png"
    )
    
    # Filter out neutral reviews (Rating = 3) and create a binary label for positive (1) and negative (0) reviews.
    df_filtered = df[df["Rating"] != 3].copy()
    
    df_filtered['label'] = df_filtered['Rating'].apply(lambda x: 1 if x >= 4 else 0)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    X = df_filtered['Review_Text']
    y = df_filtered['label']
    
    # Analyze the class distribution before splitting
    positive_negative_counts = Counter(y)
    labels = ["Negative (0)", "Positive (1)"]
    counts = [positive_negative_counts[0], positive_negative_counts[1]]
    plot_distribution(
        x_axis=labels,
        y_axis=counts,
        x_label="Class",
        y_label="Count",
        title="Distribution of Classes",
        filename="class_distribution.png"
    )
    
    #Initialize the TfidfVectorizer and Multinomial Naive Bayes model
    vectorizer = TfidfVectorizer(max_features=5000)
    model_name = input("Enter the model to use (naive_bayes/logistic_regression): ")
    model = get_model(model_name)
    
    fold_accuracies = []
    
    all_y_test = []
    all_y_pred = []
    
    for train_index, test_index in tqdm(skf.split(X, y), total=5, desc="Processing Folds"):
    # Split the data into training and testing sets for the current fold
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Vectorize the text data
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train the model
        model.fit(X_train_vec, y_train)
        
        # Predict and evaluate the model
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(acc)
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)

    print(f"Average Accuracy across Folds: {sum(fold_accuracies)/5:.2f}")
    
    # Generate and save the confusion matrix
    cm_normalized = confusion_matrix(all_y_test, all_y_pred, normalize='true')

    # Create a confusion matrix display with the normalized values and save it as an image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=["Negative", "Positive"])
    disp.plot(cmap='Blues', values_format='.2f')
    plt.title("Normalized Confusion Matrix (Percentages)")
    plt.savefig(f"confusion_matrix_normalized_{model_name}.png")
    
    