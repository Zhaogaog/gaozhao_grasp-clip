import numpy as np
a = np.ones((2,3))
b = np.ones((3,3))
list1 = [a,b]
list2 = [a,a,b]
list3 = [list1,list2]
list4 = [list3, list3]
a = np.zeros((0,3))
# d = np.array(list3)
for scene in list3:
    for ann in scene:
        a = np.vstack((a,ann))
with open ('list.txt','w') as f:
    for list_ in list4:
        f.write(str(list_))
        f.write('\n')
f.close()
list = []
for i in list:
    print(i)
a= []
for i in range(5):
    if len(a) ==
    print(len(a))

