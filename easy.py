import tensorflow as tf
from tensorflow.keras import layers, models, initializers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = to_categorical(y_train), to_categorical(y_test)


# Model
def create_model(initializer=None, dropout=False, regularizer=None):
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Flatten(),

        layers.Dense(256, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizer),

        layers.Dropout(0.2) if dropout else layers.Activation('linear'),

        layers.Dense(128, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizer),

        layers.Dropout(0.2) if dropout else layers.Activation('linear'),

        layers.Dense(64, activation='relu',
                     kernel_initializer=initializer,
                     kernel_regularizer=regularizer),

        layers.Dropout(0.2) if dropout else layers.Activation('linear'),

        layers.Dense(10, activation='softmax')
    ])

    return model


# Models
baseline_model = create_model()

xavier_model = create_model(
    initializer=initializers.glorot_uniform()
)

kaiming_model = create_model(
    initializer=initializers.he_normal()
)

dropout_model = create_model(
    dropout=True
)

l1_model = create_model(
    regularizer=regularizers.l1(0.01)
)

l2_model = create_model(
    regularizer=regularizers.l2(0.01)
)


# Compile
models_list = [
    baseline_model,
    xavier_model,
    kaiming_model,
    dropout_model,
    l1_model,
    l2_model
]

for model in models_list:
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )


# Train
histories = []

for model in models_list:
    history = model.fit(
        x_train, y_train,
        epochs=5,
        validation_data=(x_test, y_test)
    )
    histories.append(history)


# Plot validation accuracy
names = ['Baseline', 'Xavier', 'Kaiming', 'Dropout', 'L1', 'L2']

for history, name in zip(histories, names):
    plt.plot(history.history['val_accuracy'], label=name)

plt.xlabel('Epoch')
plt.ylabel('Validation Accuracy')
plt.legend()
plt.show()