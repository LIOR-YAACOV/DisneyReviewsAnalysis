from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

def get_model(model_name):
    if model_name == 'naive_bayes':
        return MultinomialNB()
    elif model_name == 'logistic_regression':
        return LogisticRegression(class_weight='balanced', max_iter=1000)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

def plot_distribution(x_axis, y_axis, x_label, y_label, title, filename):
    plt.bar(x_axis, y_axis)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename)
    plt.clf()