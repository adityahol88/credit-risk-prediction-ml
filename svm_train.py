from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

X_train = train_data.drop(columns=["loan_status"]).values
y_train = train_data["loan_status"].values

X_test = test_data.drop(columns=["loan_status"]).values
y_test = test_data["loan_status"].values

# param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1]}
# grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
# grid_search.fit(X_train, y_train)

# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)

# best_model = grid_search.best_estimator_

model = SVC(C=10, gamma=0.1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

cm_normalized = conf_matrix.astype("float") / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"],
)
plt.title("SVM")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("./plots/SVM.png")
plt.close()
