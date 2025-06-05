import json
import os
thing_classes = []

scene_list = []
for scene_id in range(135,142):
    scene_list.append(scene_id)
# for scene_id in range(300,3)
for scene_id in scene_list:
    input_classes = []
    scene_path = '/media/ama/data0/gz/graspnet/graspnet_sim/scenes/scene_'+ str(scene_id).zfill(4)
    with open(os.path.join(scene_path, 'obj_id_list.txt'), 'r') as f:
        for line in f.readlines():
            if line != '\n':
                if 'picture_frame' in line:
                    # thing_classes.append('black_picture_frame')
                    pass
                else:
                    # thing_classes.append(line.strip().replace('_', ' '))
                    input_classes.append(line.strip().replace('_', ' '))

                    # thing_classes.append(line.strip())

    thing_classes.append(input_classes)
print(thing_classes)