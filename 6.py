# S7AH Niranjan J
# Weight initializations and regularizers
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np
import time

# Load and preprocess dataset for all parts of the experiment
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)

print("=" * 50)
print("1. BASE MODEL")
print("=" * 50)
base_model = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
base_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
st = time.time()
base_model.fit(x_train, y_train_cat, epochs=5, batch_size=64, validation_data=(x_test, y_test_cat))
sp = time.time()
base_test_loss, base_test_acc = base_model.evaluate(x_test, y_test_cat)
base_time_taken = round(sp - st, 2)
base_accuracy = base_test_acc
base_loss = base_test_loss
print(f'Base Model Results:\nTest Loss: {round(base_test_loss, 4)}\nTest Accuracy: {round(base_test_acc * 100, 2)}%\nTime Taken: {base_time_taken} seconds\n')


print("=" * 50)
print("2. WEIGHT INITIALIZERS (Xavier & Kaiming)")
print("=" * 50)
def create_model(initializer=None):
    model_with_kernel = models.Sequential([
        layers.Flatten(input_shape=(32, 32, 3)),
        layers.Dense(256, activation='relu', kernel_initializer=initializer),
        layers.Dense(128, activation='relu', kernel_initializer=initializer),
        layers.Dense(64, activation='relu', kernel_initializer=initializer),
        layers.Dense(10, activation='softmax')
    ])
    return model_with_kernel

res = []
initialization_results = {}
name_map = {
    'glorot_uniform': 'Xavier (Glorot)',
    'he_normal': 'Kaiming (He)'
}
weight_initializers = ['glorot_uniform', 'he_normal']

for init in weight_initializers:
    print(f"Training with initializer: {init}\n")
    model = create_model(initializer=init)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    wei_st = time.time()
    model.fit(x_train, y_train_cat, epochs=5)
    wei_sp = time.time()
    test_loss, test_accuracy = model.evaluate(x_test, y_test_cat)
    time_taken = wei_sp - wei_st
    res.append((time_taken, test_accuracy))
    initialization_results[name_map[init]] = (test_loss, test_accuracy)
    print(f"\nTest Accuracy (using '{init}' weight initialization): {round(test_accuracy * 100, 2)}%\n\nTime Taken: {round(time_taken, 2)} seconds\n")
    print("--" * 40)


print("=" * 50)
print("3. COMPARISON CALCULATIONS (Glorot / He vs Base)")
print("=" * 50)
base_glorot_acc = round(base_test_acc / res[0][1], 2)
base_glorot_time = round(base_time_taken / res[0][0], 3)
base_he_acc = round(base_test_acc / res[1][1])
base_he_time = round(base_time_taken / res[1][0], 3)

print(f"Baseline Results Comparison:\n\nAccuracy comparison:\n{weight_initializers[0]}: {base_glorot_acc}% accurate than base model\n{weight_initializers[1]}: {base_he_acc}% accurate than base model")
print(f"\nTime comparison:\n{weight_initializers[0]}: {base_glorot_time} seconds faster than base model\n\n{weight_initializers[1]}: {base_he_time} seconds faster than base model\n")


print("=" * 50)
print("4. DROPOUT REGULARIZATION")
print("=" * 50)
model_with_dropout = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])
model_with_dropout.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_with_dropout.fit(x_train, y_train, epochs=5)
dropout_test_loss, dropout_test_accuracy = model_with_dropout.evaluate(x_test, y_test)
dropout_accuracy = dropout_test_accuracy
dropout_loss = dropout_test_loss
print(f"\nTest Accuracy: {round(dropout_test_accuracy * 100, 4)}%")
print("--" * 40)
base_dropout_acc = round(base_test_acc / dropout_test_accuracy, 2)
print(f"Baseline Results Comparison:\nAccuracy: {base_dropout_acc}% faster than base model\n")


print("=" * 50)
print("5. L1 REGULARIZATION")
print("=" * 50)
model_with_l1 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.01)),
    layers.Dense(10, activation='softmax')
])
model_with_l1.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_with_l1.fit(x_train, y_train, epochs=5)
reg_test_loss_l1, reg_test_accuracy_l1 = model_with_l1.evaluate(x_test, y_test)
l1_accuracy = reg_test_accuracy_l1
l1_loss = reg_test_loss_l1
print(f"\nTest Accuracy: {round(reg_test_accuracy_l1 * 100, 4)}%")
print("--" * 40)
base_reg_acc_l1 = round(base_test_acc / reg_test_accuracy_l1, 2)
print(f"Baseline Results Comparison:\nAccuracy: {base_reg_acc_l1}% faster than base model\n")


print("=" * 50)
print("6. L2 REGULARIZATION")
print("=" * 50)
model_with_l2 = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dense(10, activation='softmax')
])
model_with_l2.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_with_l2.fit(x_train, y_train, epochs=5)
reg_test_loss_l2, reg_test_accuracy_l2 = model_with_l2.evaluate(x_test, y_test)
l2_accuracy = reg_test_accuracy_l2
l2_loss = reg_test_loss_l2
print(f"\nTest Accuracy: {round(reg_test_accuracy_l2 * 100, 4)}%")
print("--" * 40)
base_reg_acc_l2 = round(base_test_acc / reg_test_accuracy_l2, 2)
print(f"Baseline Results Comparison:\nAccuracy: {base_reg_acc_l2}% faster than base model")


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print("\nModel                          Accuracy         Loss")
print("-" * 55)

print(
    f"Baseline                     {base_accuracy * 100:.2f}%        {base_loss:.4f}"
)

print(
    f"Xavier (Glorot)              "
    f"{initialization_results['Xavier (Glorot)'][1] * 100:.2f}%        "
    f"{initialization_results['Xavier (Glorot)'][0]:.4f}"
)

print(
    f"Kaiming (He)                 "
    f"{initialization_results['Kaiming (He)'][1] * 100:.2f}%        "
    f"{initialization_results['Kaiming (He)'][0]:.4f}"
)

print(
    f"Dropout                      {dropout_accuracy * 100:.2f}%        "
    f"{dropout_loss:.4f}"
)

print(
    f"L1 Regularization            {l1_accuracy * 100:.2f}%        "
    f"{l1_loss:.4f}"
)

print(
    f"L2 Regularization            {l2_accuracy * 100:.2f}%        "
    f"{l2_loss:.4f}"
)

print("-" * 55)

# Find best model
results = {
    "Baseline": base_accuracy,
    "Xavier (Glorot)": initialization_results["Xavier (Glorot)"][1],
    "Kaiming (He)": initialization_results["Kaiming (He)"][1],
    "Dropout": dropout_accuracy,
    "L1 Regularization": l1_accuracy,
    "L2 Regularization": l2_accuracy
}

best_model = max(results, key=results.get)

print("\nBest Model:", best_model)
print("Best Test Accuracy:", round(results[best_model] * 100, 2), "%")
