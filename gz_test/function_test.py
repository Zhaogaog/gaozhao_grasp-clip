import numpy as np
a = np.array([-1,2,4,5
])
b= a>0
print(b)
c = np.array([[[-1,2,3],
              [4,5,6],
              [0,5,7],
              [8,5,2]
],
              [[-1, 2, 3],
               [4, 5, 6],
               [0, 5, 7],
               [8, 5, 2]
               ]
              ])
g= np.array([[1,2,3,4],
             [-1,2,3,4]

])
# print(c[a>0])
print(a[b])
print(c[:,0]>1)
d=c>0
print(d)
e = np.array([1,2,3,4,5,6,7,8,9])
f=np.vstack(e)
print(f)
print(c[g>0])#二维
print(g[g>0])#一维
h= np.array([[1,2,3,4,5,6,7,8,9],
            [1,2,3,4,5,6,7,8,9]
             ])
i = [h,h,h]
j = np.vstack(i)
print(j)
k = np.zeros((2,1))
k[:,:]=int(5)
print(k)
l = np.load('/home/gaozhao/graspnet/graspness/scene_0000/kinect/object_idx_0000.npy')
print(l)
# for i in range(5):
#     e[2*i:2*(i+1)] = [i,i]
e = np.array([1,2,3,4,5,6,7,8,9])
print(e)
print(e[4:100].shape)
m = c[:,:,2]
print(m)

print(g[g>0])#一维
n=np.array([[1],[2],[-1]])
index=[0,1]
print(n[index])
o = g[g>0]
print(o[index])