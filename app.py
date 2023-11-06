import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

for i in range(10):
    img = Image.fromarray((x_train[i] * 255).astype('uint8'))
    plt.imshow(img)
    plt.title(f'True Label: {y_train[i]}')
    plt.show()

# dnn model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(50, activation='relu'),
  tf.keras.layers.Dense(30,activation='relu'),
  tf.keras.layers.Dense(10,activation='softmax')
])

print(len(x_train[0]))

predictions = model(x_train[:1]).numpy()  #logits- vector of raw predictions
print(np.round(predictions,2))


# tf.nn.softmax(predictions).numpy()   #converts logits to probabilities

#finding the loss 
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)   #returns scalar loss for each example
print(loss_fn(y_train[:1], predictions).numpy())  # near 2.3


#hidden layer
model.compile(optimizer='adam',                     # adam- adaptive moment estimation
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)              # hidden layer with 5 epochs

model.evaluate(x_test,  y_test, verbose=2)         # verbose 0-silent, 1-progress bar, 2- one line per epoch
model.save('model_mnist.h5')

predictions = model.predict(x_train[:10])

# Get the predicted classes
predicted_classes = np.argmax(predictions, axis=1)

# Print the predicted classes
for i in range(10):
    print(f'True Label: {y_train[i]}, Predicted Label: {predicted_classes[i]}')

exit()

