import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = (x_train / 255.0)[..., None]
x_test = (x_test / 255.0)[..., None]

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%\n")

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test.flatten()

for i in range(10):
    confidence = predictions[i][predicted_labels[i]] * 100
    print(f"Index {i:02d} | Predicted: {predicted_labels[i]} ({confidence:5.1f}%) | Actual: {true_labels[i]}")

plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    plt.title(
        f"Pred: {predicted_labels[i]}\nTrue: {true_labels[i]}",
        color=color,
        fontsize=10
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
