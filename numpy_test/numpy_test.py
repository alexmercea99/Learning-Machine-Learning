import numpy as np
import matplotlib.pyplot as plt

from matplotlib.image import imread
a = np.array([1.,1.,1.,1.])
# print(np.all(a))
img = imread('object_B.png')
print(img.shape)
img_big = imread('input_img_A.png')
print(img_big.shape)
# print(img[1][1])
# for i in range(len(img[0])-1):
#     for j in range(len(img[1])-1):
#         comparison = img[i][j] == a
#         equal_arrays = comparison.all()
#         if(equal_arrays != True):
#             img[i][j]= np.array([0.705,1,1,1])

plt.imshow(img_big)
plt.show()