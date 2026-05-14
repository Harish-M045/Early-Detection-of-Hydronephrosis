
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("models/hydronephrosis_model.h5")

classes = ["Normal","Hydronephrosis"]

img = Image.open("test_image.jpg")
img = img.resize((224,224))

img_array = np.array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)

print("Prediction:", classes[np.argmax(prediction)])
