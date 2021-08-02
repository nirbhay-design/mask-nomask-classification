import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from prediction_class import Predict_Model_class
import numpy as np
# # print(torch.cuda.get_device_properties(0).total_memory/1000000000)

# from PIL import Image
# img = Image.open('real_life_images/a.jpeg')

# img = np.array(img)
# print(img.shape)
# # print(img.size)
# # plt.imshow(img)
# # plt.show()
# # img = img.resize((224,224))
# # print(img.size)
# # plt.imshow(img)
# # plt.show()
# torchvision.transforms.ToPILImage(),
transformation = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),
                                                torchvision.transforms.Resize((256,256)),
                                                torchvision.transforms.CenterCrop((224,224)),
                                                torchvision.transforms.ToTensor()])


transformation1 = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
# img1 = transformation(img).unsqueeze(0)
# print(img1.shape)
# print(img1.max())
# # plt.imshow(img1[0][0])
# # plt.show()

mob_v2 = torchvision.models.mobilenet_v2(pretrained=True)
mob_v2.classifier[0] = nn.Dropout(p=0.3)
mob_v2.classifier[1] = nn.Linear(in_features=1280,out_features=2)

checkpoint = torch.load('mask_classification_v1.pth',map_location='cpu')
print(mob_v2.load_state_dict(checkpoint['model']))

model = Predict_Model_class(mob_v2)
# value = model.evaluate_img(img1)

# print(value)

# plt.imshow(img1[0][0])
# plt.show()




import cv2
import numpy as np
import pandas as pd

cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
l = []
cam = cv2.VideoCapture(0)
cam.set(4, 4800)  # set video widht
cam.set(4, 480)  # set video height
font = cv2.FONT_HERSHEY_SIMPLEX
count = 1
while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(50, 50),
    )
    for(x, y, w, h) in faces:
        
        count += 1
        test = img[y:y+h, x:x+w]
        img1 = cv2.resize(test, (512, 512))
        test1 = []
        test1.append(np.array(img1))
        x_test = np.array([i for i in test1])
        # plt.imshow(x_test[0])
        # plt.show()
        # print(x_test.shape)

        x_test = x_test[0]
        # print(x_test.shape)
        l.append(x_test)
        img1 = transformation(x_test).unsqueeze(0)
        print(img1.shape)
        value = model.evaluate_img(img1)
        color = (0,255,0) if value == 'masked' else (0,0,255)

        # cv2.rectangle(img, (x, y), (x+w, y+h), (0,225,0), 2)
        cv2.rectangle(img, (x-7, y-10), (x+w+7, y+h+12), color, 2)
        """prediction= model.predict(x_test)
        score = tf.nn.softmax(prediction)"""
    cv2.imshow('Video', img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cam.release()
cv2.destroyAllWindows()

# for i,value in enumerate(l):
#     value = transformation1(value)
#     value.save(f'my_images/img{i}.jpg')
#     if (i == 10):
#         break
