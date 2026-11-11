import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

base_vgg = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(32, 32, 3)
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

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
true_labels = y_test.flatten()

for i in range(10):
    pred_class = class_names[predicted_labels[i]]
    true_class = class_names[true_labels[i]]
    confidence = predictions[i][predicted_labels[i]] * 100
    print(f"Index {i:02d} | Predicted: {pred_class:<10} ({confidence:5.1f}%) | Actual: {true_class}")

plt.figure(figsize=(12, 8))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[i])
    color = 'green' if predicted_labels[i] == true_labels[i] else 'red'
    plt.title(
        f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[true_labels[i]]}",
        color=color,
        fontsize=10
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
