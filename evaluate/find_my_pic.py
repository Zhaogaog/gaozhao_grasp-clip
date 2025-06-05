import json
scene_set = set()
scene_dict = {}
with open('/media/ama/data0/gz/graspnet/graspnet/obj_lists/kinect/obj_lists_test.json','r')as f:
    for line in f.readlines():
        dic = json.loads(line)
        if dic['scene_id'] not in scene_dict.keys():

            scene_dict[dic['scene_id']] = set([dic['obj_id']])
        else:
            print( scene_dict[dic['scene_id']])
            scene_dict[dic['scene_id']].add(dic['obj_id'])
print(scene_dict)
for key in scene_dict.keys():
    if {7,38} < scene_dict[key]:
    # if {9,41,11} < scene_dict[key]:
        print(key)