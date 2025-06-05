import json
obj_ids_train = []
with open('/media/ama/data0/gz/graspnet/graspnet/obj_lists/kinect/obj_lists_test.json', 'r') as f:
    for line in f.readlines():
        # obj_id = 'scene_{}'.format(str(json.loads(line)['obj_id']).zfill(4))
        obj_id = json.loads(line)['obj_id']
        if obj_id not in obj_ids_train:
            obj_ids_train.append(obj_id)
print(obj_ids_train)
print(len(obj_ids_train))
        # self.obj_lists.append(json.loads(line))