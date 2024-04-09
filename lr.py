import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        """
        Initializes the LogisticRegression model.

        Parameters:
        - learning_rate (float): The step size at each iteration.
        - n_iterations (int): Number of iterations over the training dataset.

        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None  # To be initialized in fit method
        self.bias = None  # To be initialized in fit method

        # Since we are using one vs rest approach initializing lists for different weights and bias used in different classfiers
        self.weights_list = []
        self.bias_list = []
        self.models = []

    # Calculating sigmoid value based on the formula
    def _sigmoid(self, z):
        sigmoid_value = 1/ (1 + np.exp(-z))
        return sigmoid_value


    def fit(self, X, y):
        self.X = X
        self.y = y
        
        # m -> number of samples (rows), n -> number of features (columns) 
        self.m, self.n = X.shape
        
        # Initialize weights and bias to zeros
        self.weights = np.zeros(self.n)
        self.bias = 0
        for i in range(self.n_iterations) :
            # Gradient descent to update weights and bias
            # Implement the Gradient descent method to update the cost function
            z = self.X.dot(self.weights) + self.bias

            # Based on logistic regression defrential formulas for weights and bias
            dw = (1/self.m) * np.dot(self.X.T, (self._sigmoid(z) - self.y))
            db = (1/self.m) * np.sum(self._sigmoid(z) - self.y)

            # Updating the weights using formula w = w - dw, b = b - db
            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
            
        self.weights_list.append(self.weights)
        self.bias_list.append(self.bias)
        

    def predict_proba(self, X):
        
        # Calculate probability using the sigmoid function for each sample
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)
    

    def predict(self, X):
        if not self.models:
            raise ValueError("No models have been trained. Call one_vs_rest first.")
            
        # Initialize a predictions array
        predictions = np.zeros((X.shape[0], len(self.models)))
        
        # Fill the predictions array with probabilities from each model
        for idx, model in enumerate(self.models):
            predictions[:, idx] = model.predict_proba(X).ravel()
        
        # Return the class with the highest probability for each instance
        return np.argmax(predictions, axis=1)

    # As there are more than two target classes we are following one vs rest approach to fit the models
    def one_vs_rest(self, X, y):
        self.models = []
        classes = np.unique(y)
        for c in classes:
            y_binary = np.where(y == c, 1, 0)
            model = LogisticRegression(self.learning_rate, self.n_iterations)
            model.fit(X, y_binary)
            self.models.append(model)

def plot_decision_boundary(X, y, classifier, resolution=0.02):
    # Setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = plt.cm.RdYlBu
    labels = ['Setosa', 'Versicolor', 'Virginica']

    # Plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Increase figure size
    plt.figure(figsize=(10, 8))
    
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class samples with unique colors and labels
    for idx, (cl, label) in enumerate(zip(np.unique(y), labels)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx],
                    marker=markers[idx], label=label, edgecolor='black')

    # Create plot directory and save the figure
    plot_dir = "plot"
    if not os.path.isdir(plot_dir):
        os.makedirs(plot_dir)

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    
    # Adjust the legend positioning
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Classes")

    plt.tight_layout()  # Adjust the layout
    plt.savefig(f"{plot_dir}/decision_boundary_lr.png")
    plt.show()



def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the dataset only once
    X_train, X_test, y_train, y_test = train_test_split(X_scaled[:, :2], y, test_size=0.2, random_state=42)

    logistic_reg = LogisticRegression()
    # Train the model using one-vs-rest approach
    logistic_reg.one_vs_rest(X_train, y_train)
    # Predict on the test set
    y_predicted_test = logistic_reg.predict(X_test)

    # For plotting, use the scaled features but only the first two for visualization
    X_plot = X_scaled[:, :2]

    # Plotting the decision boundary using the first two features
    plot_decision_boundary(X_scaled[:, :2], y, logistic_reg)

    # Evaluation
    print("Evaluation on Test data : ")
    cm = metrics.confusion_matrix(y_test, y_predicted_test)
    precision = metrics.precision_score(y_test, y_predicted_test, average='weighted')
    accuracy = metrics.accuracy_score(y_test, y_predicted_test)
    recall = metrics.recall_score(y_test, y_predicted_test, average='weighted')
    f1_score = metrics.f1_score(y_test, y_predicted_test, average='weighted')
    
    print("confusion_matrix")
    print(cm)
    print("Precision : ", precision)
    print("Accuracy : ", accuracy)
    print("Recall : ", recall)
    print("F1_score : ", f1_score)


if __name__ == "__main__":
    main()

