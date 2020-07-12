from __future__ import division
import random
import math
import time
import numpy as np
import threading
import os,sys
import airsimdroneracingvae
import airsimdroneracingvae.types
import airsimdroneracingvae.utils
import cv2
from scipy.spatial.transform import Rotation


# add the path to the folder that contains the AirSimClient module
sys.path += ["/home/regen/anaconda3/envs/Airsim/lib/python2.7/site-packages"]

# now this import should succeed
import airsim


# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
#import_path = "/home/dogukanyy/catkin_ws/src/AirSim-Drone-Racing-VAE-Imitation"#racing_utils/"
print(import_path)
sys.path.insert(0, import_path)
import racing_utils

random.seed()

# DEFINE DATA GENERATION META PARAMETERS
num_gates_track = 8
race_course_radius = 21
gate_displacement_noise = 1.3
viz_traj = False
direction = 0  # 0 for clockwise, 1 for counter-clockwise
perpendicular = False  # if True, then move with velocity constraint
vel_max = 5.0
acc_max = 3.0
radius_noise = gate_displacement_noise
height_range = [0, -gate_displacement_noise]

class DroneRacingDataGenerator(object):
    def __init__(self, 
                drone_name,
                gate_passed_thresh,
                race_course_radius,
                radius_noise,
                height_range,
                direction,
                perpendicular,
                odom_loop_rate_sec,
                vel_max,
                acc_max):
        self.base_path = '/home/regen/Desktop/all_files/airsim_dataset' #'/home/rb/all_files/airsim_datasets/soccer_test'
        self.csv_path = os.path.join(self.base_path, 'gate_training_data.csv')
        self.csv_path2 = os.path.join(self.base_path, 'imu.csv')
        self.csv_path3 = os.path.join(self.base_path, 'ground_truth.csv')

        self.curr_track_gate_poses = None
        self.next_track_gate_poses = None
        self.gate_object_names_sorted = None
        self.num_training_laps = None
        self.imu_data = None

        # gate idx trackers
        self.gate_passed_thresh = gate_passed_thresh
        self.last_gate_passed_idx = -1
        self.last_gate_idx_moveOnSpline_was_called_on = -1
        self.next_gate_idx = 0
        self.next_next_gate_idx = 1
        self.next_next_next_gate_idx = 2
        self.train_lap_idx = 0
        #EKLEDIM - PoseSampler
        self.curr_idx = 0
        self.file = open(self.csv_path, "a")
        self.file2 = open(self.csv_path2, "a")
        self.file3 = open(self.csv_path3, "a")

        # should be same as settings.json
        self.drone_name = drone_name
        # training params
        self.race_course_radius = race_course_radius
        self.radius_noise = radius_noise
        self.height_range = height_range
        self.direction = direction
        self.perpendicular=perpendicular

        self.vel_max = vel_max
        self.acc_max = acc_max

        # todo encapsulate in function
        self.client = airsimdroneracingvae.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.05)

        # threading stuff
        self.got_odom = False
        self.is_expert_planner_controller_thread_active = False
        self.expert_planner_controller_thread = threading.Thread(target=self.repeat_timer_expert, args=(self.expert_planner_controller_callback, odom_loop_rate_sec))
        # self.image_loop = threading.Thread(target=self.repeat_timer, args=(self.image_callback, 0.05))

    # def image_callback(self):
    #     self.client.()
    # EKLEDIM
    # write image to file
    def writeImgToFile(self, image_response):
        if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            cv2.imwrite(os.path.join(self.base_path, 'images', str(self.curr_idx).zfill(len(str(100000))) + '.png'), img_rgb)  # write to png
            print("Timestamp- IMG",self.curr_multirotor_state.timestamp)
        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3} \n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)

    def writeImuToFile(self,x,y,z,q,w,e,f):
        data_string2 = '{0} {1} {2} {3} {4} {5} {6} \n'.format(x,y,z,q,w,e,f)
        self.file2.write(data_string2)

    def writeGTToFile(self, x,y,z,e1,e2,e3,e4):
        data_string3 = '{0} {1} {2} {3} {4} {5} {6} \n'.format(x,y,z,e1,e2,e3,e4)
        self.file3.write(data_string3)

    def convert_position_world_2_body(self, x, y, z):

        dx = self.curr_multirotor_state.kinematics_estimated.orientation.x_val
        dy = self.curr_multirotor_state.kinematics_estimated.orientation.y_val
        dz = self.curr_multirotor_state.kinematics_estimated.orientation.z_val
        dw = self.curr_multirotor_state.kinematics_estimated.orientation.w_val
        
        q = np.array([dw, dx, dy, dz], dtype=np.float64)
        n = np.dot(q, q)

        q *= math.sqrt(2.0 / n)
        q = np.outer(q, q)
        rotation_matrix = np.array([[1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0]],
                                    [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0]],
                                    [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2]]])
        
        t_body_np = [x,y,z]
        t_world_np = np.dot(np.transpose(rotation_matrix), t_body_np)
        #print("t_world:", t_world_np, t_body_np)
        t_world = t_world_np.reshape(-1,1)
        return t_world
        
    def Car_to_Spherical(self):
        #drone_position = self.curr_xyz
        #gate_pos = self.curr_track_gate_poses[self.next_gate_idx].position
        #print((self.curr_track_gate_poses[self.next_gate_idx].position).type())
        x_dif = - self.curr_xyz[0] + self.curr_track_gate_poses[self.next_gate_idx].position.x_val
        y_dif = - self.curr_xyz[1] + self.curr_track_gate_poses[self.next_gate_idx].position.y_val
        z_dif = - self.curr_xyz[2] + self.curr_track_gate_poses[self.next_gate_idx].position.z_val

        x_dif, y_dif, z_dif = self.convert_position_world_2_body(x_dif,y_dif,z_dif)
        
        r = math.sqrt(x_dif*x_dif + y_dif*y_dif + z_dif*z_dif)
        psi = math.atan2(y_dif,x_dif)
        theta = math.acos(z_dif/r)                          
        return r, psi, theta  

    def phi_relative_calculation(self):
        gx = self.curr_track_gate_poses[self.next_gate_idx].orientation.x_val
        gy = self.curr_track_gate_poses[self.next_gate_idx].orientation.y_val
        gz = self.curr_track_gate_poses[self.next_gate_idx].orientation.z_val
        gw = self.curr_track_gate_poses[self.next_gate_idx].orientation.w_val
        gate_orientation = Rotation.from_quat([gx, gy, gz, gw])
        gate_orientation_euler = gate_orientation.as_euler('zyx', degrees=True)

        dx = self.curr_multirotor_state.kinematics_estimated.orientation.x_val
        dy = self.curr_multirotor_state.kinematics_estimated.orientation.y_val
        dz = self.curr_multirotor_state.kinematics_estimated.orientation.z_val
        dw = self.curr_multirotor_state.kinematics_estimated.orientation.w_val
        drone_orientation = Rotation.from_quat([dx, dy, dz, dw])
        drone_orientation_euler = drone_orientation.as_euler('zyx', degrees=True)

        phi_rel = gate_orientation_euler[0] - drone_orientation_euler[0]
        #print("donusumden once: ",phi_rel)
        if phi_rel < -180:
            phi_rel += 180
            if phi_rel < -180:
                phi_rel += 180

        if phi_rel > 0:
            phi_rel -= 180
            if phi_rel > 0:
                phi_rel -= 180

        phi_rel = phi_rel*np.pi/180
        #print("donusumden sonra: ",phi_rel)        
        return phi_rel

     

    def repeat_timer_expert(self, task, period):
        while self.is_expert_planner_controller_thread_active:
            task()
            time.sleep(period)

    # def repeat_timer_image_cb(self, task, period):
    #     while self.is_expert_planner_controller_thread_active:
    #         task()
            # time.sleep(period)

    def load_level(self, level_name='Soccer_Field_Easy'):
        self.client.simLoadLevel(level_name)
        time.sleep(2)

        self.set_current_track_gate_poses_from_default_track_in_binary()
        self.next_track_gate_poses = self.get_next_generated_track()

        for gate_idx in range(len(self.gate_object_names_sorted)):
            print(self.next_track_gate_poses[gate_idx].position.x_val, self.next_track_gate_poses[gate_idx].position.y_val, self.next_track_gate_poses[gate_idx].position.z_val)
            self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx], self.next_track_gate_poses[gate_idx])
            time.sleep(0.05)

        self.set_current_track_gate_poses_from_default_track_in_binary()
        self.next_track_gate_poses = self.get_next_generated_track()

    def set_current_track_gate_poses_from_default_track_in_binary(self):
        gate_names_sorted_bad = sorted(self.client.simListSceneObjects("Gate.*"))
        # gate_names_sorted_bad is ['Gate0', 'Gate10_21', 'Gate11_23', 'Gate1_3', 'Gate2_5', 'Gate3_7', 'Gate4_9', 'Gate5_11', 'Gate6_13', 'Gate7_15', 'Gate8_17', 'Gate9_19']
        # number after underscore is unreal garbage. also leading zeros are not there. 
        gate_indices_bad = [int(gate_name.split('_')[0][4:]) for gate_name in gate_names_sorted_bad]
        gate_indices_correct = sorted(range(len(gate_indices_bad)), key=lambda k:gate_indices_bad[k])
        self.gate_object_names_sorted = [gate_names_sorted_bad[gate_idx] for gate_idx in gate_indices_correct]

        # limit the number of gates in the track
        self.gate_object_names_sorted = self.gate_object_names_sorted[:num_gates_track]

        self.curr_track_gate_poses = [self.client.simGetObjectPose(gate_name) for gate_name in self.gate_object_names_sorted]
        # destroy all previous gates in map
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)

        # generate track with correct number of gates
        self.next_track_gate_poses = self.get_next_generated_track()
        self.curr_track_gate_poses = self.next_track_gate_poses

        # create red gates in their places
        for idx in range(len(self.gate_object_names_sorted)):
            self.client.simSpawnObject(self.gate_object_names_sorted[idx], "RedGate16x16", self.next_track_gate_poses[idx], 0.75)
            time.sleep(0.05)

        # for gate_pose in self.curr_track_gate_poses:
        #     print(gate_pose.position.x_val, gate_pose.position.y_val,gate_pose.position.z_val)

    def takeoff_with_moveOnSpline(self, takeoff_height, vel_max, acc_max):
        self.client.moveOnSplineAsync(path=[airsimdroneracingvae.Vector3r(4, -2, takeoff_height)],
                                      vel_max=vel_max, acc_max=acc_max,
                                      add_curr_odom_position_constraint=True,
                                      add_curr_odom_velocity_constraint=True,
                                      viz_traj=viz_traj,
                                      vehicle_name=self.drone_name).join()

    def expert_planner_controller_callback(self):
        self.curr_multirotor_state = self.client.getMultirotorState()
        airsim_xyz = self.curr_multirotor_state.kinematics_estimated.position
        self.curr_xyz = [airsim_xyz.x_val, airsim_xyz.y_val, airsim_xyz.z_val]
        self.got_odom = True

        if ((self.train_lap_idx == 0) and (self.last_gate_passed_idx == -1)):
            if (self.last_gate_idx_moveOnSpline_was_called_on == -1):
                self.fly_to_next_gate_with_moveOnSpline()
                self.last_gate_idx_moveOnSpline_was_called_on = 0
                return                
            
        # todo transcribe hackathon shitshow of lists to np arrays
        # todo this NOT foolproof. future self: check for passing inside or outside of gate.
        # BEN EKLEDIM
        #anim = airsim.MultirotorClient().simDisableActor('Gate.*') # work perfectly
        if (self.curr_track_gate_poses is not None):
            # request quad img from AirSim,

            print("mainloop",self.curr_idx)
            image_response = self.client.simGetImages([airsimdroneracingvae.ImageRequest('0', airsimdroneracingvae.ImageType.Scene, False, False)])[0]
            # save all the necessary information to file

            #print(self.curr_xyz[0], self.curr_xyz[1], self.curr_xyz[2], "\n", self.curr_track_gate_poses[self.next_gate_idx].position.x_val, self.curr_track_gate_poses[self.next_gate_idx].position.y_val, self.curr_track_gate_poses[self.next_gate_idx].position.z_val)
            dist_from_next_gate, psi, theta = self.Car_to_Spherical()
            phi_rel = self.phi_relative_calculation()
            #print("bizim fonk:",dist_from_next_gate, phi_rel)
            self.imu_data = self.client.getImuData()            

            #rint(self.client.getImuData.angular_velocity())
            self.writeImgToFile(image_response)
            self.writePosToFile(dist_from_next_gate, psi, theta, phi_rel)
            self.writeGTToFile(self.curr_multirotor_state.kinematics_estimated.position.x_val, self.curr_multirotor_state.kinematics_estimated.position.y_val,
                self.curr_multirotor_state.kinematics_estimated.position.z_val,
                self.curr_multirotor_state.kinematics_estimated.orientation.w_val,self.curr_multirotor_state.kinematics_estimated.orientation.x_val,
                self.curr_multirotor_state.kinematics_estimated.orientation.y_val,self.curr_multirotor_state.kinematics_estimated.orientation.z_val)
                           #print(np.array(self.imu_data.angular_velocity.x_val)).type) #,self.imu_data.angular_velocity.y_val,self.imu_data.angular_velocity.z_val, self.imu_data.linear_acceleration.x_val, self.imu_data.linear_acceleration.y_val, self.imu_data.linear_acceleration.z_val)
            a1 = np.array(self.imu_data.linear_acceleration.x_val)
            a2 = np.array(self.imu_data.linear_acceleration.y_val)
            a3 = np.array(self.imu_data.linear_acceleration.z_val)
            b1 = np.array(self.imu_data.orientation.w_val)
            b2 = np.array(self.imu_data.orientation.x_val)
            b3 = np.array(self.imu_data.orientation.y_val)
            b4 = np.array(self.imu_data.orientation.z_val)

            self.writeImuToFile(a1,a2,a3,b1,b2,b3,b4)
            #print("Relative Phi value = ", phi_rel
            self.curr_idx += 1


            # print(self.last_gate_passed_idx, self.next_gate_idx, dist_from_next_gate)
            self.client.simDestroyObject(self.gate_object_names_sorted[self.next_next_gate_idx])
            self.client.simDestroyObject(self.gate_object_names_sorted[self.next_next_next_gate_idx])
            
            if dist_from_next_gate < self.gate_passed_thresh:                
                self.last_gate_passed_idx += 1
                self.next_gate_idx += 1
                self.next_next_gate_idx += 1
                self.next_next_next_gate_idx +=1 
                # self.set_pose_of_gate_just_passed()
                self.set_pose_of_gate_passed_before_the_last_one()

                if self.next_next_gate_idx >= len(self.curr_track_gate_poses):
                    self.next_next_gate_idx = 0

                if self.next_next_next_gate_idx >= len(self.curr_track_gate_poses):
                    self.next_next_next_gate_idx = 0
                # if current lap is complete, generate next track
                if (self.last_gate_passed_idx == len(self.curr_track_gate_poses)-1):
                    print("Generating next track")
                    self.last_gate_passed_idx = -1
                    self.next_gate_idx = 0
                    self.curr_track_gate_poses = self.next_track_gate_poses 
                    self.next_track_gate_poses = self.get_next_generated_track()
                    self.train_lap_idx += 1

                    # if last gate of last training lap was just passed, chill out and stop the expert thread!
                    # todo stopping thread from callback seems pretty stupid. watchdog?
                    if (self.train_lap_idx == self.num_training_laps-1):
                        self.stop_expert_planner_controller_thread()

                # todo this is pretty ugly
                if (not(self.last_gate_idx_moveOnSpline_was_called_on == self.next_gate_idx)):
                    self.fly_to_next_gate_with_moveOnSpline()
                    self.last_gate_idx_moveOnSpline_was_called_on = self.next_gate_idx
                # self.fly_to_next_gate_with_learner()
                # self.fly_to_next_gate_with_moveToPostion()
                self.client.simSpawnObject(self.gate_object_names_sorted[self.next_next_gate_idx-1], "RedGate16x16", self.curr_track_gate_poses[self.next_next_gate_idx-1], 0.75)

    def fly_to_next_gate_with_moveOnSpline(self):
        # print(self.curr_track_gate_poses[self.next_gate_idx].position)
        # print(self.curr_track_gate_poses[self.next_next_gate_idx].position)
        if not self.perpendicular:
            self.last_future = self.client.moveOnSplineAsync([self.curr_track_gate_poses[self.next_gate_idx].position],
                                                             vel_max=self.vel_max, acc_max=self.acc_max,
                                                             add_curr_odom_position_constraint=True,
                                                             add_curr_odom_velocity_constraint=True,
                                                             viz_traj=viz_traj,
                                                             vehicle_name=self.drone_name)
        else:
            gate_vector = racing_utils.geom_utils.get_gate_facing_vector_from_quaternion(self.curr_track_gate_poses[self.next_gate_idx].orientation, self.direction, scale=vel_max/1.5)
            self.last_future = self.client.moveOnSplineVelConstraintsAsync([self.curr_track_gate_poses[self.next_gate_idx].position],
                                                                           [gate_vector],
                                                                           vel_max=self.vel_max, acc_max=self.acc_max,
                                                                           add_curr_odom_position_constraint=True,
                                                                           add_curr_odom_velocity_constraint=True,
                                                                           viz_traj=viz_traj,
                                                                           vehicle_name=self.drone_name)

    # maybe maintain a list of futures, or else unreal binary will crash if join() is not called at the end of script
    def join_all_pending_futures(self):
        self.last_future.join()

    def get_next_generated_track(self):
        # todo enable gate spawning in neurips environments for variable number of gates in training laps
        # self.next_track_gate_poses = self.track_generator.generate_gate_poses(num_gates=random.randint(6,10), race_course_radius=30.0, type_of_segment = "circle")
        return racing_utils.trajectory_utils.generate_gate_poses(num_gates=len(self.curr_track_gate_poses),
                                                                 race_course_radius=self.race_course_radius,
                                                                 radius_noise=self.radius_noise,
                                                                 height_range=self.height_range,
                                                                 direction=self.direction,
                                                                 type_of_segment="circle")

    def set_pose_of_gate_just_passed(self):
        if (self.last_gate_passed_idx == -1):
            return
        self.client.simSetObjectPose(self.gate_object_names_sorted[self.last_gate_passed_idx], self.next_track_gate_poses[self.last_gate_passed_idx])
        # todo unhardcode 100+, ensure unique object ids or just set all non-gate objects to 0, and gates to range(self.next_track_gate_poses)... not needed for hackathon
        # self.client.simSetSegmentationObjectID(self.gate_object_names_sorted[self.last_gate_passed_idx], 100+self.last_gate_passed_idx);
        # todo do we really need this sleep
        time.sleep(0.05)

    def set_pose_of_gate_passed_before_the_last_one(self):
        gate_idx_to_move = self.last_gate_passed_idx - 1

        # if last_gate passed was -1 or 0, it means the "next" track is already the "current" track. 

        if (self.train_lap_idx > 0):
            if (self.last_gate_passed_idx in [-1,0]):
                print("last_gate_passed_idx", self.last_gate_passed_idx, "moving gate idx from CURRENT track", gate_idx_to_move)
                self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move], self.curr_track_gate_poses[gate_idx_to_move])
                return
            else:
                print("last_gate_passed_idx", self.last_gate_passed_idx, "moving gate idx from NEXT track", gate_idx_to_move)
                self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move], self.next_track_gate_poses[gate_idx_to_move])
                return

        if (self.train_lap_idx == 0):
            if (self.last_gate_passed_idx in [-1,0]):
                return
            else:
                print("last_gate_passed_idx", self.last_gate_passed_idx, "moving gate idx from NEXT track", gate_idx_to_move)
                self.client.simSetObjectPose(self.gate_object_names_sorted[gate_idx_to_move], self.next_track_gate_poses[gate_idx_to_move])

        # todo unhardcode 100+, ensure unique object ids or just set all non-gate objects to 0, and gates to range(self.next_track_gate_poses)... not needed for hackathon
        # self.client.simSetSegmentationObjectID(self.gate_object_names_sorted[self.last_gate_passed_idx], 100+self.last_gate_passed_idx);
        # todo do we really need this sleep
        time.sleep(0.05)

    def start_expert_planner_controller_thread(self):
        if not self.is_expert_planner_controller_thread_active:
            self.is_expert_planner_controller_thread_active = True
            self.expert_planner_controller_thread.start()
            print("Started expert_planner_controller thread")

    def stop_expert_planner_controller_thread(self):
        if self.is_expert_planner_controller_thread_active:
            self.is_expert_planner_controller_thread_active = False
            self.expert_planner_controller_thread.join()
            print("Stopped expert_planner_controller thread")

    def set_num_training_laps(self, num_training_laps):
        self.num_training_laps = num_training_laps

    def start_training_data_generator(self, num_training_laps=100, level_name='Soccer_Field_Easy'):
        self.load_level(level_name)
        # todo encapsulate in functions
        self.client.enableApiControl(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.armDisarm(True, vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.client.setTrajectoryTrackerGains(airsimdroneracingvae.TrajectoryTrackerGains().to_list(), vehicle_name=self.drone_name)
        time.sleep(0.01)
        self.takeoff_with_moveOnSpline(takeoff_height=-2, vel_max=self.vel_max, acc_max=self.acc_max)
        self.set_num_training_laps(num_training_laps)
        self.start_expert_planner_controller_thread()



if __name__ == "__main__":
    drone_racing_datagenerator = DroneRacingDataGenerator(drone_name='drone_0',
                                                          gate_passed_thresh=0.5,
                                                          race_course_radius=race_course_radius,
                                                          radius_noise=radius_noise,
                                                          height_range=height_range,
                                                          direction=direction,
                                                          perpendicular=perpendicular,
                                                          odom_loop_rate_sec=0.015,
                                                          vel_max=vel_max,
                                                          acc_max=acc_max
                                                          )
    drone_racing_datagenerator.start_training_data_generator()
