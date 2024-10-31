import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data_processing as dp
import feature_extraction as fe

sampling = 3
(x_train, y_train), (x_test, y_test) = dp.load_data_keras("../Data")
(x_test, y_test) = fe.ResnetPreprocess(x_test=x_test, y_test=y_test, sampling=sampling, test=True)

model = tf.keras.models.load_model('./model/federate_learning_model.keras')

def create_adversarial_example(model, image, label, epsilon=0.03, alpha=0.01, num_steps=10):
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor(label)

    original_image = image

    for step in range(num_steps):
        with tf.GradientTape() as tape:
            tape.watch(image)
            prediction = model(image)
            loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)

        gradient = tape.gradient(loss, image)

        image = image + alpha * tf.sign(gradient)

        image = tf.clip_by_value(image, original_image - epsilon, original_image + epsilon)
        image = tf.clip_by_value(image, 0, 1)  

    return image

image_index = 0  
image = x_test[image_index:image_index+1]
label = y_test[image_index:image_index+1]

adversarial_image = create_adversarial_example(model, image, label)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image[0])
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Adversarial Image")
plt.imshow(adversarial_image[0].numpy())
plt.axis('off')

plt.show()

original_prediction = np.argmax(model.predict(image))
adversarial_prediction = np.argmax(model.predict(adversarial_image))

print("Dự đoán ban đầu:", original_prediction)
print("Dự đoán với ví dụ đối kháng:", adversarial_prediction)
