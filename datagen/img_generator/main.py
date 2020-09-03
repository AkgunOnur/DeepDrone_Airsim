from pose_sampler import *

base_path = '/home/merkez/Downloads/DeepDrone_Airsim/'

num_samples = 1
mode = 1 # If 1, drone flies in test mode, otherwise it flies data collection mode
flight_log = True
# check if output folder exists
if not os.path.isdir(base_path):
    os.makedirs(base_path)
    img_dir = os.path.join(base_path, 'images')
    os.makedirs(img_dir)

pose_sampler = PoseSampler(num_samples, base_path, flight_log)

for idx in range(pose_sampler.num_samples):
	pose_sampler.update(mode)
