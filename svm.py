import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None

    def fit(self, X, y):
        # m = number of rows (samples), n = number of columns (features)
        self.m, self.n = X.shape 

        self.X = X
        self.Y = y
        
        # Initialize weights and bias to zeros
        self.w = np.zeros(self.n)
        self.b = 0

        # Label encoding target values
        y_labeled = np.where(self.Y <= 0, -1, 1);
        
        # Implement gradient descent to update weights and bias
        for i in range(self.n_iters) :
            for index, xi in enumerate(X) :
                if y_labeled[index] * (np.dot(xi, self.w) - self.b) >= 1 :
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else :
                    dw = 2 * self.lambda_param * self.w - np.dot(xi, y_labeled[index])
                    db = y_labeled[index]
                
                 # w = (w - learningRate * dw) => (w - learningRate * (2*lambda*
                self.w = self.w - self.lr * dw
                
                # b = (b - learningRate * db)
                self.b = self.b - self.lr * db


    def predict(self, X):
        # Return class labels based on the sign of the linear combination
        result = self.decision_fun(X)
        return np.sign(result)

    def decision_fun(self, X):
        # Compute the linear combination of weights and features plus bias
        return np.dot(X, self.w) - self.b

# Plotting the decision boundary and save the figure.
def plot_decision_boundaries(X, y, classifiers, num_classes, plot_directory="plot"):
    X_transformed = X
    
    x_min, x_max = X_transformed[:, 0].min() - 1, X_transformed[:, 0].max() + 1
    y_min, y_max = X_transformed[:, 1].min() - 1, X_transformed[:, 1].max() + 1
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    confidence_scores = np.array([classifiers[i].decision_fun(mesh_points) for i in classifiers]).T
    Z = np.argmax(confidence_scores, axis=1)
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    colors = ['red', 'blue', 'lightgreen']
    labels = ['Setosa', 'Versicolor', 'Virginica']
    for i, color, label in zip(np.unique(y), colors, labels):
        plt.scatter(X_transformed[y == i, 0], X_transformed[y == i, 1], c=color, edgecolors='k', label=label, cmap=plt.cm.Paired)

    # After dimensionalitty reduction all 4 features are tuned into two features naming -> (feature 1 and feature 2)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.legend(loc='upper left')

    # Check if directory exists, if not, create it
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    plt.savefig(f"{plot_directory}/svm_decision_boundary.png")
    plt.show()

# Calculating Confidence Scores for each class, predicting the best
def svm_one_vs_rest(X_test, classifiers):
    """Predicting using the one-vs-rest strategy for multi-class classification."""
    confidence_scores = np.array([classifiers[i].decision_fun(X_test) for i in classifiers]).T
    return np.argmax(confidence_scores, axis=1)

def main():
    # Load and preprocess dataset
    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    # Dimensionality reduction for plotting the decision boundary
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_pca)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=100)

    # Training the model
    classifiers_ovr = {}
    num_classes = len(np.unique(y_train))

    for i in range(num_classes):
        # Creating labels for the current class vs. rest of teh classes
        y_train_binary = np.where(y_train == i, 1, -1)
        
        svm = LinearSVM()
        svm.fit(X_train, y_train_binary)
        classifiers_ovr[i] = svm

    y_predicted = svm_one_vs_rest(X_test, classifiers_ovr)
    
    # Calculate the metrics using scikit-learn
    cm = metrics.confusion_matrix(y_test, y_predicted)
    precision = metrics.precision_score(y_test, y_predicted, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_predicted)
    recall = metrics.recall_score(y_test, y_predicted, average='weighted')
    f1_score = metrics.f1_score(y_test, y_predicted, average='weighted')

    print("Confusion matrix on test data")
    print(cm)
    print("Precision on test data: ", precision)
    print("Accuracy on test data : ", accuracy)
    print("Recall on test data: ", recall)
    print("F1_score on test data: ", f1_score)

    plot_decision_boundaries(X_train, y_train, classifiers_ovr, num_classes)


if __name__ == "__main__":
    main()
