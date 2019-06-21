#!/usr/bin/env python
# coding: utf-8

# In[193]:


import pandas as pd
import numpy as np
import os
pd.options.display.max_rows = 15


# In[194]:


# read csv:
# save_csv = pd.read_csv("results_val.csv")


# In[195]:


import pickle
ImageID=pickle.load(open("ImageID.pkl","rb"))
true = pickle.load(open("true.pkl","rb"))

# in format of [probability xmin ymin xmax ymax]
ours = pickle.load(open("ours.pkl","rb"))


# In[196]:


import copy
def cal_area(a_target, b):  # returns None if rectangles don't intersect
    dx = min(a_target[2], b[2]) - max(a_target[0],b[0])
    dy = min(a_target[3], b[3]) - max(a_target[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
def zero_division(n,d):
    return n/d if d else None

def calculate_recall_precision(predict,data):
    precision = []  
    recall = []
    
    if len(predict)!=len(data):
        print("Length doesn't match")
        return False,False
    
    N = len(predict)
    for i in range(N):
        if i%10000==0:
            print("Doing %d of %d"%(i,N))
        p_i = np.array(predict[i])
        data_i = np.array(data[i])

        
        # All positive
        N_positive = p_i.shape[0]
        N_true = data_i.shape[0]
        
        # calculate TP:
        count=0
        if N_positive*N_true==0:
                pass
        else:
            for j in range(N_true):
                

                area_j = []
                for k in range(p_i.shape[0]):
                    #print(p_i)
                    area_k = cal_area(a_target=data_i[j],b=p_i[k,1:])
                    if area_k:
                        
                        area_all = (data_i[j][2]-data_i[j][0])*(p_i[k,4]-p_i[k,2])+(p_i[k,3]-p_i[k,1])*(data_i[j][3]-data_i[j][1])-area_k
                        #print(area_k/area_all)
                        # loU
                        if area_k/area_all>0.5:
                            count+=1
                            p_i = np.delete(p_i, (k), axis=0)
                            p_i = np.atleast_2d(p_i)
                            k=0
                            break
                            
                            
        TP= count
        precision.append(zero_division(TP,N_positive))
        recall.append(zero_division(TP,N_true))
        
    
    return precision,recall
    
    
precision,recall = calculate_recall_precision(predict=ours,data=true)


# In[197]:


## calculate mean precision vs recall
recall = np.array(recall,dtype=float)

precision = np.array(precision,dtype=float)

x_target = sorted(list(set(recall)))
y_mean = []
for i in range(len(x_target)):
    mask = recall==x_target[i]
    y_mean.append(np.mean(precision[mask]))
y_mean = np.array(y_mean)
x_target = np.array(x_target)


# In[198]:



from scipy import integrate
import matplotlib.pyplot as plt
mask_finite = np.isfinite(x_target+y_mean)
poly = np.poly1d(np.polyfit(x_target[mask_finite],y_mean[mask_finite],5))
mAP = integrate.quad(lambda x: poly(x),0,1)[0]

plt.plot(x_target,y_mean,"k",label="mAP = %.2f"%(mAP))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




