from pose_sampler import *
import os

base_path = '/home/merkez/Downloads/DeepDrone_Airsim/'
num_iterations = 100
mode = "DATA_COLLECTION"
flight_log = True

def summarize_results(list_all = True):
	result_file = os.path.join(base_path, "files/results_102.pkl")
	cost_list, time_list = pickle.load(open(result_file, "rb"))

	method_counter = []
	for i in range(len(time_list)):
		method_counter.append(len(time_list[i]))
		time_list[i] = sorted(time_list[i].items(), key=lambda x: x[1], reverse=False)

	if list_all:
		for i in range(len(time_list)):
			print "\nThere are {0} successful flights are in test_variables_{1}.pkl file".format(method_counter[i], i)
			for element in time_list[i]:
				method, arrival_time = element
				print"Method: {0}, Time of arrival: {1:.5}".format(method,arrival_time)

	else:
		ind = np.argmax(np.array(method_counter))
		print"\nThere are {0} successful flights are in test_variables_{1}.pkl file".format(np.max(np.array(method_counter)), ind)

		for element in time_list[ind]:
			method, arrival_time = element
			print"Method: {0}, Time of arrival: {1:.5}".format(method,arrival_time)


def main():
	
	# check if output folder exists
	if not os.path.isdir(base_path):
	    os.makedirs(base_path)
	    img_dir = os.path.join(base_path, 'images')
	    os.makedirs(img_dir)

	if mode == "DATA_COLLECTION":
		pose_sampler = PoseSampler(base_path, flight_log)
		for i in range(num_iterations):
			print"PHASE: " + mode + " Iteration: {0}/{1}".format(i+1, num_iterations)
			pose_sampler.update(mode)
	elif mode == "SUMMARY":
		summarize_results()
	else:
		pose_sampler = PoseSampler(base_path, flight_log)
		print"PHASE: " + mode
		pose_sampler.update(mode)


if __name__ == '__main__':
	main()

