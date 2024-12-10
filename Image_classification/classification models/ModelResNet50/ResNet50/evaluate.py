from tensorflow.keras.models import load_model

# Load the model
model = load_model('/Users/jadeja19/Documents/Hiwi iktd/IKTD ML PROJECT/ModelResNet50')

#evaluation on a image

from PIL import Image
import numpy as np

image = Image.open("/Users/jadeja19/Documents/Pycharm/IKTD_MLProject/Dataset/allgemein1-5/05uncorrected/566.jpg")
image = image.resize((224, 224))
image_array = np.array(image)

# Normalize the image
image_array = image_array / 255.0

# If the model expects a 4D array (including batch size), add an extra dimension
image_array = np.expand_dims(image_array, axis=0)

prediction = model.predict(image_array)

predicted_class = np.argmax(prediction, axis=-1)

print(prediction)



