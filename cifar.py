import tensorflow as tf
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

hidden_units = [512, 256, 128]
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    *[tf.keras.layers.Dense(units, activation='relu') for units in hidden_units],
    tf.keras.layers.Dense(10, activation='softmax')
])


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))


_, acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {acc * 100:.2f}%")


labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
preds = model.predict(x_test[:3])

fig, axes = plt.subplots(3, 2, figsize=(8, 6))
for i in range(3):
    axes[i, 0].imshow(x_test[i])
    axes[i, 0].axis('off')
    axes[i, 1].barh(labels, preds[i], color='steelblue')
    axes[i, 1].set_xlim(0, 1)
plt.tight_layout()
plt.show()
