import numpy as np
import tensorflow as tf

# 1. Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 2. Vectorized 0-pad to 48x48 and replicate across 3 color channels
pad_spec = ((0, 0), (10, 10), (10, 10))  # 28 + 10 + 10 = 48
x_train = np.pad(x_train / 255.0, pad_spec, mode='constant', constant_values=0)
x_test = np.pad(x_test / 255.0, pad_spec, mode='constant', constant_values=0)

x_train = np.repeat(x_train[..., None], 3, axis=-1)
x_test = np.repeat(x_test[..., None], 3, axis=-1)

# 3. Instantiate Pretrained Backbone (VGG-16)
base_vgg = tf.keras.applications.VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(48, 48, 3)
)
base_vgg.trainable = False  # Freeze pretrained weights

# 4. Append Custom Classification Head
model = tf.keras.Sequential([
    base_vgg,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 5. Compile and Fit
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=5, batch_size=128, validation_data=(x_test, y_test))

# 6. Evaluation
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc * 100:.2f}%")
