# python_assignment
# Project Title
SVM confusion matrix and navies bayes confusion matrix

A brief description of the project and what it does.
it helps to evaluate the SVM model and Naives bayes model to calculate the actual and predicted values taking macro average, weighted average and accuracy.


## Getting Started
you need to install jyupiter notebook and install all the libraries as given below.
you need all the required libraries such as,
import pandas as pd,
import numpy as np,
import matplotlib.pyplot as plt,
import seaborn as sns,
from sklearn.model_selection import train_test_split,
from sklearn.preprocessing import StandardScaler,
from sklearn.svm import SVC,
from sklearn.naive_bayes import GaussianNB,
from sklearn.metrics import classification_report, confusion_matrix.

## dataset
Load the dataset where you saved your file

## encoding
Encode the target variable by
df['Class'] = df['Class'].map({'Kecimen': 0, 'Besni': 1})

## split the data into features and target
X = df.drop(columns=['Class'])
y = df['Class']


## Running the Tests

check for the missing values and split the data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

then train both the models and evaluate them.
## SVM constructs a hyperplane to separate classes
svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
## Naïve Bayes assumes feature independence and follows Bayes' Theorem
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)

## code for the SVM evaluation model
print("SVM Classification Report:")
print(classification_report(y_test, svm_predictions))
print("SVM Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, svm_predictions), annot=True, fmt='d', cmap='Blues')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

## code for naives bayes model
print("Naïve Bayes Classification Report:")
print(classification_report(y_test, nb_predictions))
print("Naïve Bayes Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, nb_predictions), annot=True, fmt='d', cmap='Oranges')
plt.title("Naïve Bayes Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


## Deployment

by using all the above instructions and code give above you can run SVM and bayes model successfully.

## Author

Akhil Karra

## License

This project is licensed under the MIT License.

## Acknowledgments

https://www.baeldung.com/cs/naive-bayes-vs-svm
https://www.geeksforgeeks.org/naive-bayes-vs-svm-for-text-classification/
https://www.analyticsvidhya.com/blog/2020/11/understanding-naive-bayes-svm-and-its-implementation-on-spam-sms/

take help of these sites to understand how they can be performed accurately






