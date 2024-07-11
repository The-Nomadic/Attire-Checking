import keras
from keras.preprocessing import image
import numpy as np

# Step 1: Load the model from the .h5 file
model_path = r'C:\Programs\GCP\ml-deployment\Attire-Checking\Attire_model1 1.h5'
model = keras.models.load_model(model_path)

# Step 2: Prepare input data (example with a single image)
image_path = r'C:\Programs\GCP\ml-deployment\Attire-Checking\pics\test4.webp'
img = image.load_img(image_path, color_mode='rgb', target_size=(128, 128))  # Load as RGB

# Convert the image to a numpy array
x = image.img_to_array(img)

# Expand dimensions to match the model's input shape (batch size, height, width, channels)
x = np.expand_dims(x, axis=0)

# Normalize the image data to the range [0, 1]
x = x / 255.0

# Step 3: Make predictions
predictions = model.predict(x)

# Step 4: Interpret the predictions (specific to your model and task)
predicted_class = np.argmax(predictions)
print("Predicted class:", predicted_class)
