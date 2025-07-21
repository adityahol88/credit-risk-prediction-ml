from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    confusion_matrix,
    classification_report,
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Custom scorer for grid search to check for best 1 label classifier
def accuracy_label(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tp = cm[1, 1]
    fn = cm[1, 0]
    accuracy_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    return accuracy_1


label_1_scorer = make_scorer(accuracy_label, greater_is_better=True)
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

X_train = train_data.drop(columns=["loan_status"]).values
y_train = train_data["loan_status"].values

X_test = test_data.drop(columns=["loan_status"]).values
y_test = test_data["loan_status"].values

param_grid = {
    "max_depth": [3, 5, 10, 20, None],
    "n_estimators": [10, 30, 50, 70, 90, 110],
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(
    rf, param_grid, cv=5, scoring=label_1_scorer, verbose=1, n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best accuracy:", grid_search.best_score_)

best_params = grid_search.best_params_
best_rf = RandomForestClassifier(
    max_depth=best_params["max_depth"],
    n_estimators=best_params["n_estimators"],
    random_state=42,
)

best_rf.fit(X_train, y_train)
# tree = best_rf.estimators_[0]
y_pred = best_rf.predict(X_test)
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
plt.title("Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("./plots/RandomForest.png")
plt.close()

# feature_names = train_data.drop(columns=["loan_status"]).columns.tolist()
# plt.figure(figsize=(20, 10))
# plot_tree(
#     tree,
#     feature_names=feature_names,
#     class_names=["Class 0", "Class 1"],
#     filled=True,
#     rounded=True,
#     fontsize=10,
# )
# plt.title("Decision Tree Visualization")
# plt.savefig("./plots/DecisionTree.svg", format="svg", bbox_inches="tight")
# plt.close()
