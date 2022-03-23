#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import io
import cv2
import matplotlib.pyplot as plt 
import numpy as np
from PIL import Image


# In[2]:


# conda install opencv


# In[47]:


img = cv2.imread('sample.png')
img2 = io.imread('sample.png')


# In[5]:


print(type(img))


# In[6]:


print(img.shape)
print(img2.shape)


# In[7]:


img


# 

# In[8]:


img2


# In[4]:


plt.title('Sample')  # RGB
plt.imshow(img)
plt.show()
# plt.imshow(img2)
# plt.show()


# In[33]:


cv2.imshow("hello",img)
cv2.waitKey(0)


# In[46]:


hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
plt.title('Sample')  # RGB
plt.imshow(hsv)
plt.show()


# In[27]:


# plt.imshow(hsv)
cv2.imshow("hello",hsv)
cv2.waitKey(0)


# In[48]:


# #120,120,188
# #62,62,123
lower_blue = np.array([180,0,0])
upper_blue = np.array([255,140,100])

# lower_blue = np.array([ 96, 176, 160])
# upper_blue = np.array([116, 196, 255])


# In[49]:


#black & white mask
# image_mask = cv2.inRange(hsv,lower_blue,upper_blue)
image_mask = cv2.inRange(img,lower_blue,upper_blue)


# In[50]:


image_mask=255-image_mask


# In[51]:


cv2.imshow("hello",image_mask)
cv2.waitKey(0)


# In[ ]:


#bright color mask
# hsvOutput = cv2.bitwise_and(img, img, mask=image_mask)


# In[ ]:


cv2.imshow("hello",img)
cv2.waitKey(0)


# In[ ]:


cv2. imwrite("result.png",image_mask)


# In[42]:


ratio_blue = cv2.countNonZero(image_mask)/(img.size/3)
panel_uncovered = np.round(100-ratio_blue*100, 2)
print(' ratio of panel', ratio_blue )
print(' pixel percentage:', panel_uncovered )


# In[43]:


per_biofouling = 100 - panel_uncovered
print('percentage of biofouling', per_biofouling )


# In[32]:


plt.imshow(image_mask)


# In[ ]:




