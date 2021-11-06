import os
classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

if os.path.exists('trainpaths.txt'):
    os.remove('trainpaths.txt')

root_dir = 'data/seg_train/seg_train'
for folder in os.listdir(root_dir):
    for filename in os.listdir(os.path.join(root_dir, folder)):
        path = os.path.join(root_dir, folder, filename)
        label = classes.index(folder)
        with open('trainpaths.txt', 'a') as f:
            f.writelines(str(label)+','+path+'\n')


if os.path.exists('testpaths.txt'):
    os.remove('testpaths.txt')

root_dir = 'data/seg_test/seg_test'
for folder in os.listdir(root_dir):
    for filename in os.listdir(os.path.join(root_dir, folder)):
        path = os.path.join(root_dir, folder, filename)
        label = classes.index(folder)
        with open('testpaths.txt', 'a') as f:
            f.writelines(str(label)+','+path+'\n')

