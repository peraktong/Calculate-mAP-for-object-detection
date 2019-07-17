#!/usr/bin/env python
# coding: utf-8

# In[7]:



import os

import matplotlib.pyplot as plt
import numpy as np
## read data:
import pandas as pd
import copy

import os
folder_names = sorted(os.listdir("../Data/ILSVRC/Data/CLS-LOC/train/"))

folder_names =sorted([i for i in folder_names if "n" in i])
# print(len(folder_names))

# match class based on alphabet:
label_to_index = dict((name, index) for index, name in enumerate(folder_names))



val_csv = pd.read_csv("../Data/LOC_val_solution.csv")
data_path = "../Data/ILSVRC/Data/CLS-LOC/test/"


outs_csv = pd.read_csv("output_val.csv")


# In[6]:


val_csv[2:12]


# In[5]:


outs_csv[2:12]


# In[ ]:


box_val = []
box_ours = []
image_id = []
count=0
count_val,count_ours = 0,0

for i in range(len(val_csv)):
    if count%10000==0:
        print("Doing %d of %d"%(count,len(val_csv)))
    # submit_csv_mod
    temp = outs_csv["PredictionString"][outs_csv["ImageId"]==val_csv["ImageId"][i]+".JPEG"].tolist()
    temp_val = val_csv["PredictionString"][i]
    temp_val = temp_val.split(" ")[:-1]

    box_val_i = []
    for k in range(int(len(temp_val)//5)):
        box_val_i.append([float(temp_val[1+5*k]),float(temp_val[2+5*k]),float(temp_val[3+5*k]),float(temp_val[4+5*k])])
    box_val_i = np.array(box_val_i)
    
    
    
    try:
        temp = temp[0]
        temp = temp.split(" ")[:-1]
        # print(temp)
        n_box = len(temp)//6
        
        score = []
        for j in range(n_box):

            score.append(temp[1+j*6])
        score = np.array(score,dtype=float)
        #print(score)
        
        index_max = np.argsort(score)[::-1]
        #print(index_max)
        
        line=""
        count_j = 0
        box_ours_i = []
        temp = np.array(temp,dtype=float)
        for j in range(n_box):
            box_ours_i.append([temp[6*j+2],temp[6*j+3],temp[6*j+4],temp[6*j+5]])
            
            count_j+=1
            if count_j>5:
                break
        box_ours_i = np.array(box_ours_i)

        

            
    except:
        box_ours_i = []
        
    #print("ours")
    image_id.append(val_csv["ImageId"][i])
    count_ours+=1
    box_ours.append(box_ours_i)
    #print("val")
    count_val+=1
    box_val.append(box_val_i)
        

        
        
    
    count+=1

    
    


# In[ ]:


# save:
image_id = np.array(image_id)
import pickle
pickle.dump(image_id,open("ImageID.pkl","wb"))
pickle.dump(box_val,open("box_val.pkl","wb"))
pickle.dump(box_ours,open("box_ours.pkl","wb"))


# In[ ]:


# calculate:

def cal_area(a_target, b):  # returns None if rectangles don't intersect
    dx = min(a_target[2], b[2]) - max(a_target[0],b[0])
    dy = min(a_target[3], b[3]) - max(a_target[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy
def zero_division(n,d):
    return n/d if d else np.nan

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
                p_i = np.atleast_2d(p_i)
                data_i = np.atleast_2d(data_i)
                # print(p_i,data_i)
                

                area_j = []
                for k in range(p_i.shape[0]):
                    #print(p_i)
                    area_k = cal_area(a_target=data_i[j],b=p_i[k,:])
                    if area_k:
                        
                        area_all = (p_i[k,2]-p_i[k,0])*(p_i[k,3]-p_i[k,1])+(data_i[j][2]-data_i[j][0])*(data_i[j][3]-data_i[j][1])-area_k
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
    
    
precision,recall = calculate_recall_precision(predict=box_ours,data=box_val)


# In[ ]:



## calculate mean precision vs recall
recall = np.array(recall)

precision = np.array(precision)

x_target = sorted(list(set(recall)))
y_mean = []
weight = []
for i in range(len(x_target)):
    mask = recall==x_target[i]
    weight.append(len(precision[mask]))
    y_mean.append(np.mean(precision[mask]))
y_mean = np.array(y_mean)
x_target = np.array(x_target)
weight = np.array(weight)

# In[ ]:



from scipy import integrate
import matplotlib.pyplot as plt
mask_finite = np.isfinite(x_target+y_mean)
poly = np.poly1d(np.polyfit(x_target[mask_finite],y_mean[mask_finite],5))
mAP = integrate.quad(lambda x: poly(x),0,1)[0]
mAP = np.nansum(weight*y_mean/(np.nansum(weight)))

plt.plot(x_target,y_mean,"k",label="mAP = %.2f"%(mAP))
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()



# In[ ]:





# In[ ]:





# In[ ]:




