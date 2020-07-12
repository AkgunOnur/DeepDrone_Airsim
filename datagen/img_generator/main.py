import time
from pose_sampler import *

num_samples = 10000
dataset_path = '/home/kca/Desktop/all_files/airsim_dataset/' #'/home/rb/all_files/airsim_datasets/soccer_test'

# check if output folder exists
if not os.path.isdir(dataset_path):
    os.makedirs(dataset_path)
    img_dir = os.path.join(dataset_path, 'images')
    os.makedirs(img_dir)
else:
    print('Error: path already exists')

pose_sampler = PoseSampler(num_samples, dataset_path)


for idx in range(pose_sampler.num_samples):
    
    pose_sampler.update()
    

    if idx % 1 == 0:
        # Crash durumunu yazacak
        print('Num samples: {}'.format(idx))
    # time.sleep(0.3)   #comment this out once you like your ranges of values
