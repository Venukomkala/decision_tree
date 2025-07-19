import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay

# load the dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='target')
target_names = iris.target_names

# quick look at the data
print("Shape:", X.shape)
print(X.head())

# add labels to the dataframe
df = X.copy()
df['species'] = y.map({i: name for i, name in enumerate(target_names)})

# class count plot
sns.countplot(data=df, x='species', palette='Set2')
plt.title("Class Count")
plt.show()

# correlation heatmap
sns.heatmap(df.drop('species', axis=1).corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# train a decision tree
model = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=42)
model.fit(X_train, y_train)

# plot feature importances
sns.barplot(x=model.feature_importances_, y=iris.feature_names, palette='rocket')
plt.title("Feature Importance")
plt.show()

# predictions and evaluation
y_pred = model.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print(classification_report(y_test, y_pred, target_names=target_names))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_names).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# plot the decision tree
plt.figure(figsize=(16, 10))
plot_tree(model, filled=True, rounded=True, feature_names=iris.feature_names, class_names=target_names)
plt.title("Decision Tree")
plt.show()

# try predicting a new sample
sample = np.array([[5.1, 3.5, 1.4, 0.2]])
pred_class = model.predict(sample)[0]
print("Prediction for [5.1, 3.5, 1.4, 0.2]:", target_names[pred_class])
