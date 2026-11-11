import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

pad_spec = ((0, 0), (10, 10), (10, 10))
x_train = np.pad(x_train / 255.0, pad_spec, mode='constant', constant_values=0)
x_test = np.pad(x_test / 255.0, pad_spec, mode='constant', constant_values=0)

x_train = np.repeat(x_train[..., None], 3, axis=-1)
x_test = np.repeat(x_test[..., None], 3, axis=-1)

base_vgg = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(48, 48, 3)
)
base_vgg.trainable = False

model = tf.keras.Sequential([
    base_vgg,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test.flatten()

for i in range(10):
    confidence = predictions[i][predicted_labels[i]] * 100
    print(f"Index {i:02d} | Predicted: {predicted_labels[i]} ({confidence:5.1f}%) | Actual: {true_labels[i]}")

plt.figure(figsize=(12, 8))
for i in range(3):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i])
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    plt.title(
        f"Pred: {predicted_labels[i]}\nTrue: {true_labels[i]}",
        color=color,
        fontsize=10
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
