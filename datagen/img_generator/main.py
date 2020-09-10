from pose_sampler import *

base_path = '/home/merkez/Downloads/DeepDrone_Airsim/'
num_iterations = 5

mode = "DATA_COLLECTION" # If 1, drone flies in test mode, otherwise it flies data collection mode
flight_log = True
# check if output folder exists
if not os.path.isdir(base_path):
    os.makedirs(base_path)
    img_dir = os.path.join(base_path, 'images')
    os.makedirs(img_dir)

pose_sampler = PoseSampler(base_path, flight_log)
for i in range(num_iterations):
	print "Iteration: {0}/{1}".format(i+1, num_iterations)
	pose_sampler.update(mode)
