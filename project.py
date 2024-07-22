import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def load_dataset(path):
    data = []
    labels = []
    for i in range(1, 41):
        for j in range(1, 11):
            img_path = f"{path}/s{i}/{j}.pgm"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            data.append(img.flatten())
            labels.append(i)
    return np.array(data), np.array(labels)

labels = load_dataset(r"C:\Users\jeffr\Desktop\dataset")
num_classes = len(np.unique(labels))

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.4, random_state=42)

mean_face = np.mean(X_train, axis=0)
delta_X_train = X_train - mean_face
covariance_matrix = np.cov(delta_X_train, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

k_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
accuracies = []

for k in k_values:
    feature_vector = eigenvectors[:, :k]
    X_train_pca = np.dot(delta_X_train, feature_vector)

    ann_classifier = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
    ann_classifier.fit(X_train_pca, y_train)

    X_test_pca = np.dot(X_test - mean_face, feature_vector)
    accuracy = ann_classifier.score(X_test_pca, y_test)
    accuracies.append(accuracy)

plt.plot(k_values, accuracies, marker='o')
plt.title('Accuracy vs. Number of Principal Components')
plt.xlabel('Number of Principal Components (k)')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()
