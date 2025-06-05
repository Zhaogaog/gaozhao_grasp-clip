import os
sdfpath = './SDFGen/bin/SDFGen'
modelpath = './models'
modelpath = '/media/ama/data0/gz/OBJ/Similar'
modelpath = '/media/ama/data0/gz/graspnet/graspnet_sim/models/test_similar'
modelpath = '/media/ama/data0/gz/OBJ/test_similar/scenes/'
# obj_names = os.listdir(modelpath)
# obj_names = sorted(obj_names)
# print(obj_names)
for scene_name in range(8,10):
    # scene_name = 0
    split = 'test_similar'
    modelpath = '/media/ama/data0/gz/OBJ/' + split + '/scenes/' + 'scene_' + str(300+scene_name).zfill(4)

    categories = []
    with open('/media/ama/data0/gz/graspnet/graspnet_sim/scenes/scene_' + str(300+scene_name).zfill(4)+ '/obj_id_list.txt') as f:
        for line in f.readlines():
            if line != '\n':
                categories.append(line.strip())
                print(line.strip())
    # categories = ['red_bell_pepper']

    for obj_name in categories:
        print("Processing %s" % obj_name)
        # objpath = os.path.join(modelpath, obj_name + '.obj')
        print(modelpath)
        if not os.path.exists(os.path.join(modelpath, obj_name + '.sdf')):
            os.system('%s %s/%s.obj %d %d' % (sdfpath, modelpath, obj_name,  100, 5))
