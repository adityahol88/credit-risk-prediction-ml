from tensorflow import keras
from tensorflow.keras import layers, callbacks
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")

X_train = train_data.drop(columns=["loan_status"]).values
y_train = train_data["loan_status"].values

X_test = test_data.drop(columns=["loan_status"]).values
y_test = test_data["loan_status"].values

print(X_train.shape)

model = keras.Sequential(
    [
        layers.Dense(128, input_shape=[22]),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=10,
    restore_best_weights=True,
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Validation Accuracy: {accuracy:.4f}")

plt.plot(history.history["accuracy"], label="train accuracy")
plt.plot(history.history["val_accuracy"], label="validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("./plots/Accuracy.png")

plt.plot(history.history["loss"], label="train loss")
plt.plot(history.history["val_loss"], label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.savefig("./plots/Loss.png")

y_pred = (model.predict(X_test) > 0.5).astype("int32")

cm = confusion_matrix(y_test, y_pred)
cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    xticklabels=["Class 0", "Class 1"],
    yticklabels=["Class 0", "Class 1"],
)
plt.title("Neural Network")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.savefig("./plots/NeuralNetwork.png")
