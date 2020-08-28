import time
from pose_sampler import *

num_samples = 100
dataset_path = '/home/merkez/Downloads/DeepDrone_Airsim/files/'
random_range = 0.3

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
