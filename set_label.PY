#Setting  Labels corresponding to each Class.


import pickle


data_dict = pickle.load(open('./data.pickle', 'rb'))

# print(len(data_dict['labels']))
# print(len(data_dict['data']))

labels2=[]

for temp in data_dict['labels']:
    if temp=='A':
        tt=0
    elif temp=='B':
        tt=1
    elif temp=='C':
        tt=2
    elif temp=='D':
        tt=3
    elif temp=='E':
        tt=4
    elif temp=='F':
        tt=5
    elif temp=='G':
        tt=6
    elif temp=='H':
        tt=7
    elif temp=='I':
        tt=8
    elif temp=='J':
        tt=9
    elif temp=='K':
        tt=10
    elif temp=='L':
        tt=11
    elif temp=='M':
        tt=12
    elif temp=='N':
        tt=13
    elif temp=='O':
        tt=14
    elif temp=='P':
        tt=15
    elif temp=='Q':
        tt=16
    elif temp=='R':
        tt=17
    elif temp=='S':
        tt=18
    elif temp=='T':
        tt=19
    elif temp=='U':
        tt=20
    elif temp=='V':
        tt=21
    elif temp=='W':
        tt=22
    elif temp=='X':
        tt=23
    elif temp=='Y':
        tt=24
    elif temp=='Z':
        tt=25
    elif temp=='del':
        tt=26
    elif temp=='space':
        tt=27

    labels2.append(tt)

f = open('d2.pickle', 'wb')
pickle.dump({'data': data_dict['data'], 'labels': labels2}, f)
f.close()