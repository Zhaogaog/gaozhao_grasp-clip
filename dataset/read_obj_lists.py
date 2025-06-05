import json
num = sum(1 for line in open('/home/gaozhao/graspnet/obj_lists/kinect/obj_lists.json'))
lists = []
with open('/home/gaozhao/graspnet/obj_lists/kinect/obj_lists.json','r') as f:
    for line in f.readlines():
        dic = json.loads(line)
        lists.append(dic)
print(num)
print(len(lists))
print(lists[0])