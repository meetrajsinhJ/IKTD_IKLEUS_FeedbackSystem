from tensorflow.keras.models import load_model

# Load the model
model = load_model('/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/EfficientNet.h5')

#evaluation on a image

from PIL import Image
import numpy as np

image = Image.open("/Users/jadeja19/Documents/Hiwi iktd/data set/allgemein1-5/01Schraubedrawcorrected/page_1_7.jpg")
image = image.resize((260, 260))
image_array = np.array(image)

# Normalize the image
image_array = image_array / 255.0

# If the model expects a 4D array (including batch size), add an extra dimension
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array)

predicted_class = np.argmax(prediction, axis=-1)

print(prediction)
