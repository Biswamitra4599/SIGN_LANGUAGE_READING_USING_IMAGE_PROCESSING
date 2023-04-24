#Traning Using Random Forest Classifier
#Importing Libraries.

import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

#Loading picle(bit) File
data_dict = pickle.load(open('./d3.pickle', 'rb'))
# # print(data_dict.keys())
# count=-0
# while 1:
#     #print(data_dict['data'][count])
#     if(len(data_dict['data'][count])!=42 ):
#         print(len(data_dict['data'][count]), "--",count)
#     # print(data_dict['labels'][count])
#     # print(count)
#     count+=1

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
# print(data.shape)
# d2=[]
# l2=[]
# for d in data:
#     tt= np.array(d)
#     d2.append(tt)

# for l in labels:
#     l2.append(int(l));


# print(len(d2))
# print(len(l2))
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
print("kk")
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were classified correctly !'.format(score * 100))
print("jj")
f = open('model.p', 'wb')
print("ii")
pickle.dump({'model': model}, f)
print("hh")
f.close()
print("gg")


