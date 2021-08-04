import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from prediction_class import Predict_Model_class
import numpy as np
# # print(torch.cuda.get_device_properties(0).total_memory/1000000000)

from PIL import Image
img = Image.open('images/amirkhan.jpg')

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
transformation = torchvision.transforms.Compose([torchvision.transforms.Resize((256,256)),
                                                torchvision.transforms.CenterCrop((224,224)),
                                                torchvision.transforms.ToTensor()])


transformation1 = torchvision.transforms.Compose([torchvision.transforms.ToPILImage()])
img1 = transformation(img).unsqueeze(0)
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
_,value = model.evaluate_img(img1)

print(value)

plt.imshow(img1[0][0])
plt.show()