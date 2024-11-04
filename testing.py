import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
import numpy as np
import cv2
from matplotlib import pyplot as plt

import glob
import random
import os
from PIL import Image

folder_path = '/path/to/images'
image_extensions = ['*.png', '*.jpg', '*.jpeg']

# Use glob to get a list of images
images = glob.glob(os.path.join('testing-images', image_extensions[1]))

# Select a random image
random_image = random.choice(images)
#print(random_image)


#img = cv2.imread('testing-images/cat-test.jpg')
img = cv2.imread(random_image)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show(block=False)
plt.pause(2)
plt.close()

resize = tf.image.resize(img, (256,256))
new_model = tf.keras.models.load_model('D:\deeplearningtesting\models\model1.h5')
yhatnew = new_model.predict(np.expand_dims(resize/255, 0))
if yhatnew > 0.5:
    print(f'Predicted class is a Puppy')
else:
    print(f'Predicted class is a Cat')
