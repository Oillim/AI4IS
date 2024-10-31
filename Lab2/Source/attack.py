import tensorflow as tf
import numpy as np
import data_processing as dp
import feature_extraction as fe
import matplotlib.pyplot as plt

# Tải dữ liệu
(x_train, y_train), (x_test, y_test) = dp.load_data_keras("../Data")
(x_test, y_test) = fe.HogPreprocess(x_val=x_test, y_val=y_test)
print("Shape của tập kiểm tra:", x_test.shape)

# Tải mô hình
model = tf.keras.models.load_model('./model/hog_model.keras')

def create_adversarial_example(model, image, label, epsilon=0.003, alpha=0.01, num_steps=10):
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

epsilon_values = [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02]  # Các giá trị epsilon để thử nghiệm
adversarial_accuracies = []

total_images = len(x_test)

for epsilon in epsilon_values:
    correct_adversarial_predictions = 0

    for i in range(50):  
        image = x_test[i:i+1]
        label = y_test[i]

        adversarial_image = create_adversarial_example(model, image, label, epsilon=epsilon)

        adversarial_prediction = np.argmax(model.predict(adversarial_image))

        if adversarial_prediction == label:
            correct_adversarial_predictions += 1

    adversarial_accuracy = correct_adversarial_predictions / 100  
    adversarial_accuracies.append(adversarial_accuracy)

plt.figure(figsize=(10, 5))
plt.plot(epsilon_values, adversarial_accuracies, marker='o')
plt.title('Tỷ lệ chính xác theo epsilon')
plt.xlabel('Epsilon')
plt.ylabel('Tỷ lệ ảnh dự đoán chính xác sau khi tạo đối kháng')
plt.xticks(epsilon_values)  
plt.grid()
plt.show()
