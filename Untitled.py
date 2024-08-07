#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
def basic_sigmoid(x):
    s = 1/ (1+math.exp(-x))
    return s


# In[5]:


basic_sigmoid(-1)


# In[6]:


import numpy as np
x= np.array([1,2,3])
print(np.exp(x))


# In[7]:


#example for vector operation
x= np.array([1,2,3])
print(x+3)


# In[8]:


#sigmoid function
import numpy as np
def sigmoid(x):
    s = 1/ (1+np.exp(-x))
    return s
    


# In[12]:


x= np.array([1,2,3])
sigmoid(x)


# In[13]:


#sigmoid gradient
def sigmoid_derivation(x):
    s = 1/(1+np.exp(-x))
    ds = s*(1-s)
    return ds


# In[15]:


x= np.array([1,2,3])
print("sigmoid_derivation(x) = "+str(sigmoid_derivation(x)))


# In[28]:


#image2vector
#image2vector
def image2vector(image):
  v = image.reshape((v.shape[0]*v.shape[1],v.shape[2], 1))
  return v


# In[30]:


import numpy as np

image = np.array([[[0.67826139, 0.29380381],
                   [0.90714982, 0.52835647],
                   [0.4215251, 0.45017551]],
                  
                  [[0.92814219, 0.96677647],
                   [0.85304703, 0.52351845],
                   [0.19981397, 0.27417313]],
                  
                  [[0.60659855, 0.00533165],
                   [0.10820313, 0.49978937],
                   [0.34144279, 0.94630077]]])

def image2vector(image):
    return image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))

print("image2vector(image) = " + str(image2vector(image)))


# In[31]:


#normalizedRow
def normalized_row(x):
  x_norm = np.linalg.norm(x,axis=1,keepdims=True)
  x = x / x_norm
  return x


# In[33]:


x=np.array([
   [0,1000,4],
    [2,6,4]])
print("normalized_row(x) = " + str(normalized_row(x)))


# In[34]:


#softmax
def softmax(x):
  x_exp = np.exp(x)
  x_sum = np.sum(x_exp,axis=1,keepdims=True)
  s = x_exp / x_sum
  return s


# In[36]:


x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])
print("softmax(x) = " + str(softmax(x)))


# In[37]:


#vectorization
from typing import TypedDict
import time 
x1 = [9, 2, 5, 0, 0,7,5,0,0,0,9,2,5,0,0]
x2 = [9, 2, 2, 9, 0,9,2,5,0,0,9,2,5,0,0]
#dot product of vectors
tic = time.process_time()
dot = 0
for i in range(len(x1)):
  dot += x1[i] * x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# In[38]:


import time
import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

# Outer product of vectors
tic = time.process_time()
outer_product = np.zeros((len(x1), len(x2)))
for i in range(len(x1)):
    for j in range(len(x2)):
        outer_product[i, j] = x1[i] * x2[j]
toc = time.process_time()

print("outer_product = \n" + str(outer_product))
print("----- Computation time = " + str(1000 * (toc - tic)) + "ms")


# In[39]:


import numpy as np
import time

x1 = np.array([9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0])
x2 = np.array([9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0])

# Elementwise multiplication and sum (dot product)
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()

print("dot = " + str(dot) + "\n----- Computation time = " + str(1000 * (toc - tic)) + "ms")


# In[40]:


import numpy as np
import time

x1 = np.array([9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0])
x2 = np.array([9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0])

# Vectorized dot product
tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()

print("dot = " + str(dot) + "\n----- Computation time = " + str(1000 * (toc - tic)) + "ms")


# In[ ]:




