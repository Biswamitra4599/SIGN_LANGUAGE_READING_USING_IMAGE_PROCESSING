#Detects Outliers from the data pickeled values.

#Importing  libraries.
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#Loading Data From Pickle File.
data_dict = pickle.load(open('./d2.pickle', 'rb'))
count=0
# while 1:
#     #print(data_dict['data'][count])
#     if(len(data_dict['data'][count])!=42 ):
#         print(len(data_dict['data'][count]), "--",count)
#     # print(data_dict['labels'][count])
#     # print(count)
#     count+=1

data = (data_dict['data'])
labels =(data_dict['labels'])

d2=[]
l2=[]
print(len(data))
for i in range (len(data)):
    if (len(data[i])==42):
       d2.append(data[i])
       l2.append(labels[i])

# Saveing Data Files in pickled File
f = open('d3.pickle', 'wb')
pickle.dump({'data': d2, 'labels': l2}, f)
f.close()