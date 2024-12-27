import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_data, epochs=10, validation_data=val_data)

# Evaluate the model
loss, accuracy = model.evaluate(test_data)
print('Test accuracy:', accuracy)

# Make a prediction
new_image = load_and_preprocess_image('path/to/new_image.jpg')
prediction = model.predict(new_image)
if prediction > 0.5:
    print('Predicted class: Dog')
else:
    print('Predicted class: Cat')
