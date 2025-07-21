from models.svms import SVM
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

# For linear kernel
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.93      0.80      0.86      5072
#            1       0.52      0.78      0.63      1445

#     accuracy                           0.79      6517
#    macro avg       0.73      0.79      0.74      6517
# weighted avg       0.84      0.79      0.81      6517

# For Polynomial kernel
# C: 0.1, Degree: 3, Accuracy: 0.7575571582016265
# C: 0.1, Degree: 5, Accuracy: 0.7606260549332515
# C: 1.0, Degree: 3, Accuracy: 0.8103421819855762
# C: 1.0, Degree: 5, Accuracy: 0.7604726100966702
# C: 10, Degree: 3, Accuracy: 0.8040509436857449
# C: 10, Degree: 5, Accuracy: 0.7570968236918828
# C: 100, Degree: 3, Accuracy: 0.8072732852539513
# C: 100, Degree: 5, Accuracy: 0.7601657204235077
# Accuracy: 0.8103421819855762
# Confusion Matrix:
#  [[4194  878]
#  [ 358 1087]]
# Classification Report:
#                precision    recall  f1-score   support

#            0       0.92      0.83      0.87      5072
#            1       0.55      0.75      0.64      1445

#     accuracy                           0.81      6517
#    macro avg       0.74      0.79      0.75      6517
# weighted avg       0.84      0.81      0.82      6517

prev_accuracy = 0
rc = [0.1, 1.0, 10, 100]
degrees = [3, 5]
for c in rc:
    for degree in degrees:
        model = SVM(kernel="poly", C=c, degree=degree)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"C: {c}, Degree: {degree}, Accuracy: {accuracy}")
        if accuracy > prev_accuracy:
            best_model = model
            prev_accuracy = accuracy

y_pred = best_model.predict(X_test)
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
plt.savefig("./plots/SVM_custom.png")
plt.close()
