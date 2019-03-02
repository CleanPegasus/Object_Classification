import numpy as np
import cv2

from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array




def get_square(image):

    height,width,channel=image.shape
    if(height>width):
      differ=height
    else:
      differ=width
    differ+=4

    mask = np.zeros((differ,differ,channel), dtype="uint8")
    x_pos=int((differ-width)/2)
    y_pos=int((differ-height)/2)
    mask[y_pos:y_pos+height,x_pos:x_pos+width]=image[0:height,0:width]
    mask=cv2.resize(mask,(differ,differ),interpolation=cv2.INTER_AREA)

    return mask



img = cv2.imread('images/office.png')
#height, width = img.shape



sq = get_square(img)
print(sq.shape)
#print(diff)
image = cv2.imread('images/office.png')
#image = cv2.resize(image, (224,224))
input_shape = (224, 224)
preprocess = imagenet_utils.preprocess_input

model = VGG16(weights = "imagenet")

img1 = cv2.resize(sq, (224,224))
img = img_to_array(img1)
img = preprocess(img)
img = np.expand_dims(img, axis=0)


pred = model.predict(img)
P = imagenet_utils.decode_predictions(pred)
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

#print(pred)

(imagenetID, label, prob) = P[0][0]
cv2.putText(image, "Label: {}, {:.2f}%".format(label, prob * 100),
	(20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

cv2.imshow('Original', img1)

cv2.imshow('Result', image)
cv2.waitKey(0)