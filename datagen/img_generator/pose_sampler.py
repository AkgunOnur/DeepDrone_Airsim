import cv2
import numpy as np
import os
import sys
from os.path import isfile, join

import airsimdroneracingvae as airsim
# print(os.path.abspath(airsim.__file__))
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr
import airsimdroneracingvae.types
import airsimdroneracingvae.utils
from scipy.spatial.transform import Rotation
import time

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)
import racing_utils

# Extras for Perception
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image

import Dronet
import lstmf

#Extras for Trajectory and Control
import pickle
import random
from numpy import zeros
from sklearn.externals.joblib import dump, load
from sklearn.preprocessing import StandardScaler
from quadrotor import *
from mytraj import MyTraj
import geom_utils
from geom_utils import QuadPose
from trajectory import Trajectory
from network import Net, Net_Regressor
from sympy import Point3D, Line3D

GATE_YAW_RANGE = [-np.pi, np.pi]  # world theta gate
# GATE_YAW_RANGE = [-1, 1]  # world theta gate -- this range is btw -pi and +pi
UAV_X_RANGE = [-30, 30] # world x quad
UAV_Y_RANGE = [-30, 30] # world y quad
UAV_Z_RANGE = [-2, -3] # world z quad

UAV_YAW_RANGE = [-np.pi/4, np.pi/4]  #[-eps, eps] [-np.pi/4, np.pi/4]
#eps = np.pi/20.0  # 18 degrees
UAV_PITCH_RANGE = [-np.pi/4, np.pi/4]
UAV_ROLL_RANGE = [-np.pi/4, np.pi/4]

R_RANGE = [2, 10]  # in meters
correction = 0.85
CAM_FOV = 90.0*correction  # in degrees -- needs to be a bit smaller than 90 in fact because of cone vs. square

# MP_list = ["min_vel", "min_acc", "min_jerk", "min_snap", "min_acc_stop", "min_jerk_stop", "min_snap_stop", 
#            "min_jerk_full_stop", "min_snap_full_stop", "pos_waypoint_arrived","pos_waypoint_timed", "pos_waypoint_interp"] 


flight_columns = ['true_init_x','true_init_y','true_init_z', 'noise_coeff', 'var_sum', 'diff_x', 'diff_y', 'diff_z', 'v_x', 'v_y', 'v_z', 'diff_phi', 'diff_theta', 'diff_psi', 
                  'phi_dot', 'theta_dot', 'psi_dot', 'r_var', 'phi_var', 'theta_var', 'psi_var', 'Tf', 'MP_Method', 'Cost', 'Status', 'curr_idx']

flight_filename = 'files/data.csv'

class PoseSampler:
    def __init__(self, dataset_path, flight_log, with_gate=True):
        self.num_samples = 1
        self.base_path = dataset_path
        self.csv_path = os.path.join(self.base_path, 'files/gate_training_data.csv')
        self.curr_idx = 0
        self.current_gate = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(1)        
        #self.client = airsim.MultirotorClient()
        self.configureEnvironment()
        self.log_path = os.path.join(self.base_path, 'files/flight_log.txt')
        self.flight_log = flight_log

        #----- Classifier/Regressor parameters -----------------------------
        self.mp_classifier = Net()
        self.t_or_s_classifier = Net()
        self.speed_regressor = Net_Regressor()
        self.time_regressor = None
        self.mp_scaler = None
        self.t_or_s_scaler = None
        self.speed_scaler = None
        self.time_scaler = None



        #----- Drone parameters -----------------------------
        self.quad = None
        self.dtau = 1e-3
        self.Tf = 0.
        self.xd_ddot_pr = 0.
        self.xd_dddot_pr = 0.
        self.yd_ddot_pr = 0.
        self.yd_dddot_pr = 0.
        self.psid_pr = 0.
        self.psid_dot_pr = 0.
        self.x_dot_pr = 0.
        self.y_dot_pr = 0.
        self.z_dot_pr = 0.
        self.vel_sum = 0.
        self.quadrotor_freq = int(1. / self.dtau)
        self.method = "MAX"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = Net()
        self.model = self.model.to(self.device)
        # self.model.load_state_dict(torch.load('best_model.pt'))
        # self.model.eval()
        self.state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.test_states = {"MAX":[], "DICE" :[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_jerk_full_stop":[]}
        self.test_costs = {"MAX":0., "DICE" :0., "min_vel":0., "min_acc":0., "min_jerk":0., "min_jerk_full_stop":0.}
        self.test_arrival_time = {"MAX":0., "DICE" :0., "min_vel":0., "min_acc":0., "min_jerk":0., "min_jerk_full_stop":0.}
        self.test_modes = ["MAX", "DICE", "min_vel", "min_acc", "min_jerk", "min_jerk_full_stop"]
        self.test_safe_counter = {"MAX":0, "DICE" :0, "min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0}
        # self.test_modes = ["MAX", "min_vel"]
        self.time_coeff = 0.
        self.quad_period = 0.
        self.drone_status = ""

        # pickle.dump(self.test_states, open(self.base_path + "files/test_states.pkl","wb"), protocol=2)

        #----- Motion planning parameters -----------------------------
        self.MP_methods = {"pos_waypoint_timed":1, "pos_waypoint_interp":2, "min_vel":3, "min_acc":4, "min_jerk":5, "min_snap":6,
                              "min_acc_stop":7, "min_jerk_stop":8, "min_snap_stop":9, "min_jerk_full_stop":10, "min_snap_full_stop":11,
                              "pos_waypoint_arrived":12}
        self.MP_names = ["hover", "pos_waypoint_timed", "pos_waypoint_interp", "min_vel", "min_acc", "min_jerk", "min_snap",
                         "min_acc_stop", "min_jerk_stop", "min_snap_stop", "min_jerk_full_stop", "min_snap_full_stop","pos_waypoint_arrived"]
        self.MP_cost = {"pos_waypoint_timed":1e9, "pos_waypoint_interp":1e9, "min_vel":1e9, "min_acc":1e9, "min_jerk":1e9, "min_snap":1e9,
                        "min_acc_stop":1e9, "min_jerk_stop":1e9, "min_snap_stop":1e9, "min_jerk_full_stop":1e9, "min_snap_full_stop":1e9,
                        "pos_waypoint_arrived":1e9}
        self.MP_states = {"pos_waypoint_timed":[], "pos_waypoint_interp":[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_snap":[],
                          "min_acc_stop":[], "min_jerk_stop":[], "min_snap_stop":[], "min_jerk_full_stop":[], "min_snap_full_stop":[],
                          "pos_waypoint_arrived":[]}



        self.trajSelect = np.zeros(3)
        # Select Position Trajectory Type (0: hover,                    1: pos_waypoint_timed,      2: pos_waypoint_interp,    
        #                                  3: minimum velocity          4: minimum accel,           5: minimum jerk,           6: minimum snap
        #                                  7: minimum accel_stop        8: minimum jerk_stop        9: minimum snap_stop
        #                                 10: minimum jerk_full_stop   11: minimum snap_full_stop
        #                                 12: pos_waypoint_arrived
        self.trajSelect[0] = 7
        # Select Yaw Trajectory Type      (0: none                      1: yaw_waypoint_timed,      2: yaw_waypoint_interp     3: follow          4: zero)
        self.trajSelect[1] = 2
        # Select if waypoint time is used, or if average speed is used to calculate waypoint time   (0: waypoint time,   1: average speed)
        self.trajSelect[2] = 1
        
        


        #---- Model import ---------------------------------
        self.device = torch.device("cpu")

        input_size = 4
        output_size = 4
        lstmR_hidden_size = 16
        lstmR_num_layers = 1

        # Dronet
        self.Dronet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(self.device)
        #print("Dronet Model:", self.Dronet)
        self.Dronet.load_state_dict(torch.load(self.base_path + 'weights/Dronet_new.pth'))   
        self.Dronet.eval()

        # LstmR
        self.lstmR = lstmf.LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
        self.lstmR.to(self.device)
        #print("lstmR Model:", self.lstmR)
        self.lstmR.load_state_dict(torch.load(self.base_path + 'weights/R_2.pth'))   
        self.lstmR.eval() 


        self.angle_lim = 0.
        self.pos_lim = 0.
        self.brightness = 0.
        self.contrast = 0.
        self.saturation = 0.
        self.blur_coeff = 0.
        self.blur_range = 0
        self.period_denum = 30.

        self.transformation = transforms.Compose([
                            transforms.Resize([200, 200]),
                            #transforms.Lambda(self.gaussian_blur),
                            #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                            transforms.ToTensor()])

        quat0 = R.from_euler('ZYX',[-90.,0.,0.],degrees=True).as_quat()
        quat1 = R.from_euler('ZYX',[0.,0.,0.],degrees=True).as_quat()
        quat2 = R.from_euler('ZYX',[30.,0.,0.],degrees=True).as_quat()
        quat3 = R.from_euler('ZYX',[45.,0.,0.],degrees=True).as_quat()
        quat4 = R.from_euler('ZYX',[60.,0.,0.],degrees=True).as_quat()
        quat5 = R.from_euler('ZYX',[90.,0.,0.],degrees=True).as_quat()

        self.drone_init = Pose(Vector3r(0.,10.,-2), Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3]))
        self.gate = [Pose(Vector3r(0.,2.,-2.), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                     Pose(Vector3r(2.,-5.,-2.4), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                     Pose(Vector3r(4.,-13.,-3.1), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                     Pose(Vector3r(7.,-20.,-3.75), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3]))]
                     #Pose(Vector3r(9.,-20.,-4.), Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3]))]

        # Previous gates
        # self.gate = [Pose(Vector3r(0.,20.,-2.), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
        #              Pose(Vector3r(1.,10.,-2.5), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
        #              Pose(Vector3r(2.,0.,-3.), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3]))]

        self.drone_init_2 = Pose(Vector3r(0.,0.,-2), Quaternionr(0., 0., -0.70710678, 0.70710678))
        self.gate_2 = [Pose(Vector3r(0.,-5.,-2.), Quaternionr(0., 0., 0., 1.)),
                    Pose(Vector3r(-2.,-10.,-2.5), Quaternionr(0., 0., -0.25881905, 0.96592583))]


        self.race_course_radius = 16
        self.radius_noise = 0.1
        self.height_range = [0, -1.0]
        self.direction = 0
        self.line_list = []
        self.gate_gate_distances = []
        self.gate_gate_edge_lines = []
        self.gate_edge_list = []
        self.gate_edge_distances = []
        self.collision_check_interval = 15

        self.circle_track = racing_utils.trajectory_utils.generate_gate_poses(num_gates=6,
                                                                 race_course_radius=self.race_course_radius,
                                                                 radius_noise=self.radius_noise,
                                                                 height_range=self.height_range,
                                                                 direction=self.direction,
                                                                 type_of_segment="circle")
        self.drone_init_circle = Pose(Vector3r(10.,13.,-0.1), Quaternionr(0., 0., 0.98480775, 0.17364818))


        self.track = self.gate # for circle trajectory change this with circle_track
        self.drone_init = self.drone_init # for circle trajectory change this with drone_init_circle
        #-----------------------------------------------------------------------             


    def find_gate_distances(self):
        gate_1 = self.track[0]
        init_to_gate = np.linalg.norm([gate_1.position.x_val-self.drone_init.position.x_val, gate_1.position.y_val-self.drone_init.position.y_val, gate_1.position.z_val-self.drone_init.position.z_val])
        self.gate_gate_distances.append(init_to_gate)
        for i in range(len(self.track)-1):
            gate_1 = self.track[i]
            gate_2 = self.track[i+1]
            gate_to_gate = np.linalg.norm([gate_1.position.x_val-gate_2.position.x_val, gate_1.position.y_val-gate_2.position.y_val, gate_1.position.z_val-gate_2.position.z_val])
            self.gate_gate_distances.append(gate_to_gate)


    def check_on_road(self):
        gate_drone_distances = []
        for i in range(len(self.track)):
            drone_x, drone_y, drone_z = self.quad.state[0], self.quad.state[1], self.quad.state[2]
            gate_x, gate_y, gate_z = self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val
            drone_to_center = np.linalg.norm([gate_x-drone_x, gate_y-drone_y, gate_z-drone_z])
            gate_drone_distances.append(drone_to_center)
            
        max_gate_to_gate = np.max(self.gate_gate_distances)
        min_drone_to_gate = np.min(gate_drone_distances)

        if min_drone_to_gate > 1.1 * max_gate_to_gate:
            return False

        return True


    def find_gate_edges(self):
        for i in range(len(self.track)):
            rot_matrix = Rotation.from_quat([self.track[i].orientation.x_val, self.track[i].orientation.y_val, 
                                      self.track[i].orientation.z_val, self.track[i].orientation.w_val]).as_dcm().reshape(3,3)
            gate_x_range = [.75, -.75]
            gate_z_range = [.75, -.75]
            edge_ind = 0
            #print "\nGate Ind: {0}, Gate x={1:.3}, y={2:.3}, z={3:.3}".format(i+1, self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val)
            gate_pos = np.array([self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val])
            
            check_list = []
            gate_edge_list = []
            # print ""
            for x_rng in gate_x_range:
                for z_rng in gate_z_range:
                    gate_edge_range = np.array([x_rng, 0., z_rng])
                    gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
                    gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
                    edge_ind += 1
                    # print "Index: {0}, Edge x={1:.3}, y={2:.3}, z={3:.3}".format(edge_ind, gate_edge_point[0], gate_edge_point[1], gate_edge_point[2])
                    # quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], 0., 0., 0.]
                    # self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                    # time.sleep(3)
                    gate_edge_list.append([gate_edge_point[0], gate_edge_point[1], gate_edge_point[2]])

            ind = 0
            # print "\nFor Gate: " + str(i)
            # print "They are on the same line"
            for i in range(len(gate_edge_list)):
                for j in range(len(gate_edge_list)):
                    edge_i = np.array(gate_edge_list[i])
                    edge_j = np.array(gate_edge_list[j])
                    if i != j and (i+j) != 3 and [i,j] not in check_list and [j,i] not in check_list:
                        # print "Index: " + str(ind) + " - " + str(i) + "/" + str(j)
                        # print "edge_i: " + str(edge_i) + " edge_j: " + str(edge_j)
                        u_v = abs(edge_i - edge_j)
                        current_list = [edge_i, edge_j, u_v]
                        self.line_list.append(current_list)
                        check_list.append([i,j])
                        check_list.append([j,i])
                        ind += 1

                        # print "Edge_i"
                        # quad_pose = [edge_i[0], edge_i[1], edge_i[2], -0., -0., 0.]
                        # self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        # time.sleep(3)
                        # print "Edge_j"
                        # quad_pose = [edge_j[0], edge_j[1], edge_j[2], -0., -0., 0.]
                        # self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        # time.sleep(3)



    def gaussian_blur(self, img):
        image = np.array(img)        
        image_blur = cv2.GaussianBlur(image,(65,65),self.blur_coeff)
        return image_blur
        

    def predict(self, X, model, isClassifier=True, method="MAX"):
        model.eval()  # Set model to evaluation mode
        softmax = nn.Softmax(dim=1)
        X_tensor = torch.from_numpy(X).to(self.device)
        output = model(X_tensor.float())
     
        if method=="MAX" and isClassifier:
            _, pred = torch.max(output, 1)
            return pred.item()

        elif method=="DICE" and isClassifier:
            probs = softmax(output).cpu().detach().numpy()[0]
            pred_index = np.random.choice([0, 1, 2, 3, 4], 1, p=probs)[0]
            return pred_index

        return output.item() # if mode is regression

    def check_collision(self, max_distance = 0.15):
        distance_list = []
        for i in range(len(self.track)):
            gate = self.track[i]
            distance = np.linalg.norm([self.quad.state[0] - gate.position.x_val, self.quad.state[1] - gate.position.y_val, self.quad.state[2] - gate.position.z_val])
            distance_list.append(distance)        

        distance_min = np.min(distance_list) # this is the distance of drone's center point to the closest gate 
        #print "Minimum distance: {0:.3}".format(distance_min)

        if distance_min < 1.: # if this value less than threshold, collision check should be done
            drone_x_range = [.1, -.1]
            drone_y_range = [.1, -.1]
            drone_z_range = [.025, -.025]
            rot_matrix = R.from_euler('ZYX',[self.quad.state[5], self.quad.state[4], self.quad.state[3]],degrees=False).as_dcm().reshape(3,3)
            drone_pos = np.array([self.quad.state[0], self.quad.state[1], self.quad.state[2]])
            edge_ind = 0

            #Collision check for drone's centroid
            # for i, line in enumerate(self.line_list):
            #     edge_i, edge_j, u_v = line
            #     # p1, p2, p3 = Point3D(edge_i[0], edge_i[1], edge_i[2]), Point3D(edge_j[0], edge_j[1], edge_j[2]), Point3D(drone_pos[0], drone_pos[1], drone_pos[2])
            #     # l1 = Line3D(p1, p2) 
            #     # distance = l1.distance(p3).evalf()
            #     distance_from_center = edge_i - drone_pos
            #     distance = np.linalg.norm(np.cross(distance_from_center, u_v)) / np.linalg.norm(u_v)
                
            #     #print "Edge: {0}, (Numeric) Distance from the center: {1:.3}".format(i, distance) 
            #     if distance < max_distance:
            #         print "Collision detected!"
            #         print "Index: {0}, Drone center x={1:.3}, y={2:.3}, z={3:.3}".format(i, drone_pos[0], drone_pos[1], drone_pos[2])

            #         return True

            # Collision check for Drone's corner points
            for x_rng in drone_x_range:
                for y_rng in drone_y_range:
                    for z_rng in drone_z_range:
                        drone_range = np.array([x_rng, y_rng, z_rng])
                        drone_range_world = np.dot(rot_matrix.T, drone_range.reshape(-1,1)).ravel()
                        drone_edge_point = np.array([drone_pos[0]+drone_range_world[0], drone_pos[1]+drone_range_world[1], drone_pos[2]+drone_range_world[2]])
                        edge_ind += 1
                        
                        
                        for i, line in enumerate(self.line_list):
                            edge_i, edge_j, u_v = line
                            distance_from_center = edge_i - drone_edge_point
                            distance = np.linalg.norm(np.cross(distance_from_center, u_v)) / np.linalg.norm(u_v)
                            #print "Edge: {0}, (Numeric) Distance from the center: {1:.3}".format(i, distance) 
                            if distance < max_distance:
                                print "Collision detected!"
                                print "Index: {0}, Drone corner x={1:.3}, y={2:.3}, z={3:.3}".format(i, drone_edge_point[0], drone_edge_point[1], drone_edge_point[2])

                                return True
            
            # print "No Collision!"
            return False


        else:
            return False

    def isThereAnyGate(self, img_rgb):
        # loop over the boundaries
        low_red = np.array([161, 155, 84])
        high_red = np.array([179, 255, 255])
        hsv_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        red_mask = cv2.inRange(img_rgb, low_red, high_red)
        red = cv2.bitwise_and(img_rgb, img_rgb, mask=red_mask)

        if red.any():
            #print "there is a gate on the frame!"
            return True

        return False
       

    def test_collision(self, gate_index):
        phi = np.random.uniform(-np.pi/6, np.pi/6)
        theta =  np.random.uniform(-np.pi/6, np.pi/6)
        psi = np.random.uniform(-np.pi/6, np.pi/6)
        print "\nCenter Drone Pos x={0:.3}, y={1:.3}, z={2:.3}".format(self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val)
        quad_pose = [self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val, -phi, -theta, psi]
        self.quad.state = [quad_pose[0], quad_pose[1], quad_pose[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.check_collision()
        time.sleep(5)
        

        rot_matrix = Rotation.from_quat([self.track[gate_index].orientation.x_val, self.track[gate_index].orientation.y_val, 
                                      self.track[gate_index].orientation.z_val, self.track[gate_index].orientation.w_val]).as_dcm().reshape(3,3)
        gate_x_range = [np.random.uniform(0.6, 1.0), -np.random.uniform(0.6, 1.0)]
        gate_z_range = [np.random.uniform(0.6, 1.0), -np.random.uniform(0.6, 1.0)]
        edge_ind = 0
        #print "\nGate Ind: {0}, Gate x={1:.3}, y={2:.3}, z={3:.3}".format(i+1, self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val)
        gate_pos = np.array([self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val])
        gate_edge_list = []
        for x_rng in gate_x_range:
            gate_edge_range = np.array([x_rng/1.5, 0., 0.25*np.random.uniform(-1,1)])
            gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
            gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
            print "\nEdge Drone Pos x={0:.3}, y={1:.3}, z={2:.3}".format(gate_edge_point[0], gate_edge_point[1], gate_edge_point[2])
            self.quad.state = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
            quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], -phi, -theta, psi]
            self.client.simSetVehiclePose(QuadPose(quad_pose), True)
            self.check_collision()
            time.sleep(5)
            

        for z_rng in gate_z_range:
            gate_edge_range = np.array([0.25*np.random.uniform(-1,1), 0., z_rng/1.5])
            gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
            gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
            edge_ind += 1
            print "\nEdge Drone Pos x={0:.3}, y={1:.3}, z={2:.3}".format(gate_edge_point[0], gate_edge_point[1], gate_edge_point[2])
            self.quad.state = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
            quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], -phi, -theta, psi]
            self.client.simSetVehiclePose(QuadPose(quad_pose), True)
            self.check_collision()
            time.sleep(5) 


    def check_completion(self, quad_pose, eps=0.45):
        x, y, z = quad_pose[0], quad_pose[1], quad_pose[2]

        xd = self.track[-1].position.x_val
        yd = self.track[-1].position.y_val
        zd = self.track[-1].position.z_val
        psid = Rotation.from_quat([self.track[-1].orientation.x_val, self.track[-1].orientation.y_val, 
                                   self.track[-1].orientation.z_val, self.track[-1].orientation.w_val]).as_euler('ZYX',degrees=False)[0]

        target = [xd, yd, zd, psid] 
        check_arrival = False


        if ( (abs(abs(xd)-abs(x)) <= eps) and (abs(abs(yd)-abs(y)) <= eps) and (abs(abs(zd)-abs(z)) <= eps)):
            self.quad.calculate_cost(target=target, final_calculation=True)
            check_arrival = True

        return check_arrival


    def test_algorithm(self, method = "MAX", use_model = False):
        pose_prediction = np.zeros((1000,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)
        labels_dict = {0:3, 1:4, 2:5, 3:10, 4:-1}
        gate_target = self.track[0]
        gate_psi = Rotation.from_quat([gate_target.orientation.x_val, gate_target.orientation.y_val, gate_target.orientation.z_val, gate_target.orientation.w_val]).as_euler('ZYX',degrees=False)[0]
        psi_start = gate_psi - np.pi/2  #drone kapi karsisinde olacak sekilde durmali

        #if drone is at initial point
        quad_pose = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, 0., 0., psi_start]
        self.state0 = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, 0., 0., psi_start, 0., 0., 0., 0., 0., 0.]

        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.quad = Quadrotor(self.state0)
        
        self.curr_idx = 0
        self.test_states[method].append(self.quad.state)

        self.xd_ddot_pr = 0.
        self.yd_ddot_pr = 0.
        self.xd_dddot_pr = 0.
        self.yd_dddot_pr = 0.
        self.psid_pr = 0.
        self.psid_dot_pr = 0.

        track_completed = False
        fail_check = False
        collision_check = False
        init_start = True

        covariance_sum = 0.
        prediction_std = [0., 0., 0., 0.]
        sign_coeff = 0. 
        covariance_list = []
        cov_rep_num = 5

        final_target = [self.track[-1].position.x_val, self.track[-1].position.y_val, self.track[-1].position.z_val]


        # To check collision algorithm, comment it out
        # for i in range(100):
        #     gate_index = np.random.randint(0,3)
        #     self.test_collision(gate_index)

        if self.flight_log:
            f=open(self.log_path, "a")



        while((not track_completed) and (not fail_check)):

            # if self.curr_idx % 30 == 0 and self.curr_idx != 0:
            #     noise_on = True
            # elif self.curr_idx % 15 == 0:
            #     noise_on = False

            # sign_coeff = 1.
            # if noise_on:
            #     self.brightness = random.uniform(200.,250.)
            #     self.contrast = random.uniform(200.,250.)
            #     self.saturation = random.uniform(200.,250.)
            #     self.transformation = transforms.Compose([
            #             transforms.Resize([200, 200]),
            #             #transforms.Lambda(self.gaussian_blur),
            #             transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            #             transforms.ToTensor()])
            # else:
            #     self.brightness = 0.
            #     self.contrast = 0.
            #     self.saturation = 0.
            #     self.transformation = transforms.Compose([
            #             transforms.Resize([200, 200]),
            #             #transforms.Lambda(self.gaussian_blur),
            #             #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            #             transforms.ToTensor()])
                
            # noise_coeff = self.brightness + self.contrast + self.saturation


            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            #if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            anyGate = self.isThereAnyGate(img_rgb)
            #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
            img =  Image.fromarray(img_rgb)
            image = self.transformation(img)
            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]

            if self.curr_idx % 30 == 0 and self.curr_idx != 0:
                noise_on = True
            elif self.curr_idx % 15 == 0:
                noise_on = False

            with torch.no_grad():   
                # Determine Gat location with Neural Networks
                pose_gate_body = self.Dronet(image)
                predicted_r = np.copy(pose_gate_body[0][0])


                if predicted_r < 3.0:
                    self.period_denum = 6.0
                elif predicted_r < 5.0:
                    self.period_denum = 20.0
                else:
                    self.period_denum = 30.0


                if noise_on:
                    noise_coeff = np.random.uniform(0.5, 1.5) 
                    sign_coeff = np.random.choice([-1,1])
                else:
                    noise_coeff = 0.


                pose_gate_body[0][0] += (sign_coeff*noise_coeff*pose_gate_body[0][0]) 
                pose_gate_body[0][0] = np.clip(pose_gate_body[0][0], 0.1, pose_gate_body[0][0])


                for i,num in enumerate(pose_gate_body.reshape(-1,1)):
                    #print(num, i , self.curr_idx)
                    pose_prediction[self.curr_idx][i] = num.item()

                if self.curr_idx >= 11:
                    pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                    
                    for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
                        prediction_std[i] = p_g_c.item()

                    prediction_std = np.clip(prediction_std, 0, prediction_std)
                    prediction_std = prediction_std.ravel()
                    covariance_sum = np.sum(prediction_std)

                    covariance_list.append(covariance_sum)
                    if self.curr_idx >= (11 + cov_rep_num):
                        covariance_sum = np.sum(covariance_list[-cov_rep_num:]) / float(cov_rep_num)

                    if covariance_sum > 20.:
                        anyGate = False

                    # Gate ground truth values will be implemented
                    pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
                   
                    # Trajectory generate
                    waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)

                    pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                    vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                    ang_vel0 = [self.quad.state[9], self.quad.state[10], self.quad.state[11]]
                    posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]                    
                    yaw0 = self.quad.state[5]
                    yaw_diff = pose_gate_body[3][0]
                    yawf = (self.quad.state[5]+yaw_diff) + np.pi/2
                    #yawf = Rotation.from_quat([self.track[self.current_gate].orientation.x_val, self.track[self.current_gate].orientation.y_val, 
                    #                   self.track[self.current_gate].orientation.z_val, self.track[self.current_gate].orientation.w_val]).as_euler('ZYX',degrees=False)[0] - np.pi/2
                    
                    print "\nCurrent index: {0}".format(self.curr_idx)
                    print "Final r: {0:.3}, Actual r: {1:.3}, Noise coeff: {2:.4}, Covariance sum: {3:.3}".format(pose_gate_body[0][0], predicted_r, sign_coeff*noise_coeff, covariance_sum)
                    #print "Brightness: {0:.3}, Contast: {1:.3}, Saturation: {2:.3}".format(self.brightness, self.contrast, self.saturation)
                    if self.flight_log:
                        f.write("\nCurrent index: {0}".format(self.curr_idx))
                        f.write("\nFinal r: {0:.3}, Actual r: {1:.3}, Noise coeff: {2:.4}, Covariance sum: {3:.3}".format(pose_gate_body[0][0], predicted_r, sign_coeff*noise_coeff, covariance_sum))
                        #f.write("\nBrightness: {0:.3}, Contast: {1:.3}, Saturation: {2:.3}".format(self.brightness, self.contrast, self.saturation))
                        f.write("\nMP algorithm: " + method)
                        f.write("\nEstimated time of arrival: {0:.3} s.".format(self.Tf))
                        f.write("\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))


                    if use_model:
                        # flight_columns = ['true_init_x','true_init_y','true_init_z', 'noise_coeff', 'var_sum', 'diff_x', 'diff_y', 'diff_z', 'v_x', 'v_y', 'v_z', 'diff_phi', 'diff_theta', 'diff_psi', 
                        #                     'phi_dot', 'theta_dot', 'psi_dot', 'r_var', 'phi_var', 'theta_var', 'psi_var', 'Tf', 'MP_Method', 'Cost', 'Status']

                        X_test = np.array([covariance_sum, posf[0]-pos0[0], posf[1]-pos0[1], posf[2]-pos0[2], self.quad.state[6], self.quad.state[7], self.quad.state[8], 
                                -self.quad.state[3], -self.quad.state[4], yawf-yaw0, self.quad.state[9], self.quad.state[10], self.quad.state[11],
                                 prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]]).reshape(1,-1)

                        # X_mp_test = self.mp_scaler.transform(X_test)
                        # X_time_test = self.time_scaler.transform(X_test)

                        mp_method = self.predict(X_test, model=self.mp_classifier, isClassifier=True, method=method)

                        self.trajSelect[0] = labels_dict[mp_method] 
                        self.trajSelect[1] = 2
                        self.trajSelect[2] = 0

                        if self.trajSelect[0] != -1:
                            print "Predicted MP Algorithm: ", self.MP_names[int(self.trajSelect[0])]

                            #self.Tf = self.time_regressor.predict(X_test)[0]
                            self.Tf = self.time_coeff*abs(pose_gate_body[0][0]) 
                            print "Time based trajectory, T: {0:.3}".format(self.Tf)
                            if self.flight_log:
                                f.write("\nTime based trajectory, T: {0:.3}".format(self.Tf))
                                f.write("\nPredicted Time Length: {0:.3}".format(self.Tf))
                        else:
                            print "Drone is in Safe Mode"
                            self.test_safe_counter[method] += 1
                            if self.flight_log:
                                f.write("\nDrone is in Safe Mode")

                        
                            
                    else:
                        self.trajSelect[0] = self.MP_methods[method]
                        self.trajSelect[1] = 2
                        self.trajSelect[2] = 0
                        self.Tf = self.time_coeff*abs(pose_gate_body[0][0])
                        print "Prediction mode is off. MP algorithm: " + method 
                        print "Estimated time of arrival: " + str(self.Tf) + " s."
                        if self.flight_log:
                            f.write("\nPrediction mode is off. MP algorithm: " + method)
                            f.write("\nEstimated time of arrival: " + str(self.Tf) + " s.")
                            
                    
                    if self.trajSelect[0] != -1:
                        time_list = np.hstack((0., self.Tf)).astype(float)
                        waypoint_list = np.vstack((pos0, posf)).astype(float)
                        yaw_list = np.hstack((yaw0, yawf)).astype(float)

                        self.test_arrival_time[method] += (self.Tf / self.period_denum)

                        newTraj = Trajectory(self.trajSelect, self.quad.state, time_list, waypoint_list, yaw_list) 

                        flight_period = self.Tf / self.period_denum
                        Waypoint_length = flight_period // self.dtau

                        if init_start:
                            t_list = linspace(0, flight_period, num = Waypoint_length)
                            init_start = False
                        else:
                            t_list = linspace(flight_period, 2*flight_period, num = Waypoint_length)
                                            
                        
                        #self.vel_sum = 0.
                        self.quad.costValue = 0.
                        # Call for Controller
                        for ind, t_current in enumerate(t_list): 

                            #self.vel_sum += (self.quad.state[6]**2+self.quad.state[7]**2+self.quad.state[8]**2)

                            pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)

                            xd, yd, zd = pos_des[0], pos_des[1], pos_des[2]
                            xd_dot, yd_dot, zd_dot = vel_des[0], vel_des[1], vel_des[2]
                            xd_ddot, yd_ddot, zd_ddot = acc_des[0], acc_des[1], acc_des[2]

                            xd_dddot = (xd_ddot - self.xd_ddot_pr) / self.dtau
                            yd_dddot = (yd_ddot - self.yd_ddot_pr) / self.dtau
                            xd_ddddot = (xd_dddot - self.xd_dddot_pr) / self.dtau
                            yd_ddddot = (yd_dddot - self.yd_dddot_pr) / self.dtau

                            psid = euler_des[2]

                            psid_dot = (psid - self.psid_pr) / self.dtau
                            psid_ddot = (psid_dot - self.psid_dot_pr) / self.dtau

                            current_traj = [xd, yd, zd, xd_dot, yd_dot, zd_dot, xd_ddot, yd_ddot, zd_ddot,
                                             xd_dddot, yd_dddot, xd_ddddot, yd_ddddot,
                                             psid, psid_dot, psid_ddot]

                            fail_check = self.quad.simulate(self.dtau, current_traj, final_target, prediction_std)

                            if ind % self.collision_check_interval == 0:   
                                collision_check = self.check_collision()
                            

                            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
                            self.test_states[method].append(self.quad.state)
                            self.client.simSetVehiclePose(QuadPose(quad_pose), True)


                            self.xd_ddot_pr = xd_ddot
                            self.yd_ddot_pr = yd_ddot
                            self.xd_dddot_pr = xd_dddot
                            self.yd_dddot_pr = yd_dddot
                            self.psid_pr = psid
                            self.psid_dot_pr = psid_dot

                            self.x_dot_pr = self.quad.state[6]
                            self.y_dot_pr = self.quad.state[7]
                            self.z_dot_pr = self.quad.state[8]

                            if collision_check:
                                self.quad.costValue = 1e12
                                self.test_costs[method] = self.quad.costValue
                                print "Drone has collided with the gate! Current cost: {0:.6}".format(self.test_costs[method])
                                if self.flight_log:
                                    f.write("\nDrone has collided with the gate! Current cost: {0:.6}".format(self.test_costs[method]))
                                break 
                            elif fail_check:
                                self.quad.costValue = 1e12
                                self.test_costs[method] = self.quad.costValue
                                print "Drone has crashed! Current cost: {0:.6}".format(self.test_costs[method])
                                if self.flight_log:
                                    f.write("\nDrone has crashed! Current cost: {0:.6}".format(self.test_costs[method]))
                                break
                            elif not anyGate:
                                self.quad.costValue = 1e12
                                self.test_costs[method] = self.quad.costValue
                                print "Drone has been out of the path! Current cost: {0:.6}".format(self.test_costs[method])
                                if self.flight_log:
                                    f.write("\nDrone has been out of the path! Current cost: {0:.6}".format(self.test_costs[method]))
                                break 

                            check_arrival = self.check_completion(quad_pose)

                            if check_arrival: # drone arrives to the gate
                                track_completed = True
                                #self.vel_sum = self.vel_sum / (ind + 1)
                                self.test_costs[method] += (self.Tf * self.quad.costValue / self.period_denum)
                                print "Drone has finished the lap. Current cost: {0:.6}".format(self.Tf * self.quad.costValue / self.period_denum) 
                                if self.flight_log:
                                    f.write("\nDrone has finished the lap. Current cost: {0:.6}".format(self.Tf * self.quad.costValue / self.period_denum))
                                break        
                            


                        if (not track_completed) and (not fail_check) and (not collision_check) and (anyGate): # drone didn't arrive or crash
                            #self.vel_sum = self.vel_sum / Waypoint_length
                            #print "Velocity Sum (Normalized): ", self.vel_sum
                            self.test_costs[method] += (self.Tf * self.quad.costValue / self.period_denum)
                            print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.Tf * self.quad.costValue / self.period_denum)
                            if self.flight_log:
                                f.write("\nDrone hasn't reached the gate yet. Current cost: {0:.6}".format(self.Tf * self.quad.costValue / self.period_denum))


                        if track_completed or fail_check or collision_check or not anyGate: # drone arrived to the gate or crashed or collided                      
                            break

            self.curr_idx += 1

        if self.flight_log:
            f.close()


    def fly_drone(self, f, method, pos_offset, angle_start, max_iteration = 300):
        x_offset, y_offset, z_offset = pos_offset
        phi_start, theta_start, gate_psi, psi_start = angle_start

        pose_prediction = np.zeros((1000,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)

        #if drone is at initial point
        quad_pose = [self.drone_init.position.x_val+x_offset, self.drone_init.position.y_val+y_offset, self.drone_init.position.z_val+z_offset, -phi_start, -theta_start, psi_start]
        self.state0 = [self.drone_init.position.x_val+x_offset, self.drone_init.position.y_val+y_offset, self.drone_init.position.z_val+z_offset, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]

        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.quad = Quadrotor(self.state0)
        
        self.curr_idx = 0

        self.xd_ddot_pr = 0.
        self.yd_ddot_pr = 0.
        self.xd_dddot_pr = 0.
        self.yd_dddot_pr = 0.
        self.psid_pr = 0.
        self.psid_dot_pr = 0.

        track_completed = False
        fail_check = False
        init_start = True
        collision_check = False

        noise_on = False

        covariance_sum = 0.
        prediction_std = [0., 0., 0., 0.]
        sign_coeff = 0.

        final_target = [self.track[-1].position.x_val, self.track[-1].position.y_val, self.track[-1].position.z_val]

        covariance_list = []
        cov_rep_num = 5

        while((not track_completed) and (not fail_check)):

            # if self.curr_idx % 30 == 0 and self.curr_idx != 0:
            #     noise_on = True
            # elif self.curr_idx % 15 == 0:
            #     noise_on = False

            # sign_coeff = 1.
            # if noise_on:
            #     self.brightness = random.uniform(200.,250.)
            #     self.contrast = random.uniform(200.,250.)
            #     self.saturation = random.uniform(200.,250.)
            #     self.transformation = transforms.Compose([
            #             transforms.Resize([200, 200]),
            #             #transforms.Lambda(self.gaussian_blur),
            #             transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            #             transforms.ToTensor()])
            # else:
            #     self.brightness = 0.
            #     self.contrast = 0.
            #     self.saturation = 0.
            #     self.transformation = transforms.Compose([
            #             transforms.Resize([200, 200]),
            #             #transforms.Lambda(self.gaussian_blur),
            #             #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            #             transforms.ToTensor()])
                
            # noise_coeff = self.brightness + self.contrast + self.saturation

            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            #if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            anyGate = self.isThereAnyGate(img_rgb)
            #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
            img =  Image.fromarray(img_rgb)
            image = self.transformation(img)
            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]

            if self.curr_idx % 30 == 0 and self.curr_idx != 0:
                noise_on = True
            elif self.curr_idx % 15 == 0:
                noise_on = False

            with torch.no_grad():   
                # Determine Gat location with Neural Networks
                pose_gate_body = self.Dronet(image)
                predicted_r = np.copy(pose_gate_body[0][0])


                if predicted_r < 3.0:
                    self.period_denum = 6.0
                elif predicted_r < 5.0:
                    self.period_denum = 20.0
                else:
                    self.period_denum = 30.0


                if noise_on:
                    noise_coeff = np.random.uniform(0.5, 1.5) 
                    sign_coeff = np.random.choice([-1,1])
                else:
                    noise_coeff = 0.


                pose_gate_body[0][0] += (sign_coeff*noise_coeff*pose_gate_body[0][0]) 
                pose_gate_body[0][0] = np.clip(pose_gate_body[0][0], 0.1, pose_gate_body[0][0])

                
                for i,num in enumerate(pose_gate_body.reshape(-1,1)):
                    #print(num, i , self.curr_idx)
                    pose_prediction[self.curr_idx][i] = num.item()

                if self.curr_idx >= 11:
                    pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                    
                    for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
                        prediction_std[i] = p_g_c.item()

                    prediction_std = np.clip(prediction_std, 0, prediction_std)
                    prediction_std = prediction_std.ravel()
                    covariance_sum = np.sum(prediction_std)

                    covariance_list.append(covariance_sum)
                    if self.curr_idx >= (11 + cov_rep_num):
                        covariance_sum = np.sum(covariance_list[-cov_rep_num:]) / float(cov_rep_num)

                    if covariance_sum > 20.:
                        anyGate = False

                    # Gate ground truth values will be implemented
                    pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
                    
                    self.trajSelect[0] = self.MP_methods[method]
                    self.trajSelect[1] = 2
                    self.trajSelect[2] = 0
                    self.Tf = self.time_coeff*abs(pose_gate_body[0][0]) + 0.1
                    
                    # Trajectory generate
                    waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
                    pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                    vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                    ang_vel0 = [self.quad.state[9], self.quad.state[10], self.quad.state[11]]
                    acc0 = [0., 0., 0.]
                    yaw0 = self.quad.state[5]

                    yaw_diff = pose_gate_body[3][0]
                    posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
                    yawf = (self.quad.state[5]+yaw_diff) + np.pi/2


                    print "\nCurrent index: {0}".format(self.curr_idx)
                    print "Final r: {0:.3}, Actual r: {1:.3}, Noise coeff: {2:.4}, Covariance sum: {3:.3}".format(pose_gate_body[0][0], predicted_r, sign_coeff*noise_coeff, covariance_sum)
                    #print "Brightness: {0:.3}, Contast: {1:.3}, Saturation: {2:.3}".format(self.brightness, self.contrast, self.saturation)
                    print "MP algorithm: " + method 
                    print "Estimated time of arrival: {0:.3} s.".format(self.Tf)                       
                    print "Gate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)            
                    #print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])
                    if self.flight_log:
                        f.write("\nCurrent index: {0}".format(self.curr_idx))
                        f.write("\nFinal r: {0:.3}, Actual r: {1:.3}, Noise coeff: {2:.4}, Covariance sum: {3:.3}".format(pose_gate_body[0][0], predicted_r, sign_coeff*noise_coeff, covariance_sum))
                        #f.write("\nBrightness: {0:.3}, Contast: {1:.3}, Saturation: {2:.3}".format(self.brightness, self.contrast, self.saturation))
                        f.write("\nMP algorithm: " + method)
                        f.write("\nEstimated time of arrival: {0:.3} s.".format(self.Tf))
                        f.write("\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))
                        

                    time_list = np.hstack((0., self.Tf)).astype(float)
                    waypoint_list = np.vstack((pos0, posf)).astype(float)
                    yaw_list = np.hstack((yaw0, yawf)).astype(float)

                    newTraj = Trajectory(self.trajSelect, self.quad.state, time_list, waypoint_list, yaw_list) 

                    flight_period = self.Tf / self.period_denum
                    Waypoint_length = flight_period // self.dtau

                    if init_start:
                        t_list = linspace(0, flight_period, num = Waypoint_length)
                        init_start = False
                    else:
                        t_list = linspace(flight_period, 2*flight_period, num = Waypoint_length)


                    #self.vel_sum = 0.
                    self.quad.costValue = 0.
                    self.drone_status = "SUCCESS"
                    # Call for Controller
                    for ind, t_current in enumerate(t_list): 
                        pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)

                        self.vel_sum += (self.quad.state[6]**2+self.quad.state[7]**2+self.quad.state[8]**2)

                        xd, yd, zd = pos_des[0], pos_des[1], pos_des[2]
                        xd_dot, yd_dot, zd_dot = vel_des[0], vel_des[1], vel_des[2]
                        xd_ddot, yd_ddot, zd_ddot = acc_des[0], acc_des[1], acc_des[2]

                        xd_dddot = (xd_ddot - self.xd_ddot_pr) / self.dtau
                        yd_dddot = (yd_ddot - self.yd_ddot_pr) / self.dtau
                        xd_ddddot = (xd_dddot - self.xd_dddot_pr) / self.dtau
                        yd_ddddot = (yd_dddot - self.yd_dddot_pr) / self.dtau

                        psid = euler_des[2]

                        psid_dot = (psid - self.psid_pr) / self.dtau
                        psid_ddot = (psid_dot - self.psid_dot_pr) / self.dtau

                        current_traj = [xd, yd, zd, xd_dot, yd_dot, zd_dot, xd_ddot, yd_ddot, zd_ddot,
                                         xd_dddot, yd_dddot, xd_ddddot, yd_ddddot,
                                         psid, psid_dot, psid_ddot]

                        fail_check = self.quad.simulate(self.dtau, current_traj, final_target, prediction_std)
                        if ind % self.collision_check_interval == 0:   
                            collision_check = self.check_collision()

                        quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
                        self.client.simSetVehiclePose(QuadPose(quad_pose), True)

                        self.xd_ddot_pr = xd_ddot
                        self.yd_ddot_pr = yd_ddot
                        self.xd_dddot_pr = xd_dddot
                        self.yd_dddot_pr = yd_dddot
                        self.psid_pr = psid
                        self.psid_dot_pr = psid_dot

                        self.x_dot_pr = self.quad.state[6]
                        self.y_dot_pr = self.quad.state[7]
                        self.z_dot_pr = self.quad.state[8]

                        if collision_check:
                            self.drone_status = "COLLISION"
                            self.quad.costValue = 1e12
                            self.test_cost = self.quad.costValue
                            print "Drone has collided with the gate! Current cost: {0:.6}".format(self.test_cost)
                            if self.flight_log:
                                f.write("\nDrone has collided with the gate! Current cost: {0:.6}".format(self.test_cost))
                            break 
                        elif fail_check:
                            self.drone_status = "CRASH"
                            self.quad.costValue = 1e12
                            self.test_cost = self.quad.costValue
                            print "Drone has crashed! Current cost: {0:.6}".format(self.test_cost)
                            if self.flight_log:
                                f.write("\nDrone has crashed! Current cost: {0:.6}".format(self.test_cost))
                            break 
                        elif not anyGate:
                            self.drone_status = "OFF_ROAD"
                            self.quad.costValue = 1e12
                            self.test_cost = self.quad.costValue
                            print "Drone has been out of the path! Current cost: {0:.6}".format(self.test_cost)
                            if self.flight_log:
                                f.write("\nDrone has been out of the path! Current cost: {0:.6}".format(self.test_cost))
                            break 

                        check_arrival = self.check_completion(quad_pose)

                        if check_arrival: # drone arrives to the gate
                            track_completed = True
                            #self.vel_sum = self.vel_sum / (ind + 1)
                            #print "Velocity Sum (Normalized): ", self.vel_sum
                            self.test_cost = self.Tf * self.quad.costValue / self.period_denum # time * cost
                            print "Drone has finished the lap. Current cost: {0:.6}".format(self.test_cost) 
                            if self.flight_log:
                                f.write("\nDrone has finished the lap. Current cost: {0:.6}".format(self.test_cost))
                            break        


                    if (not track_completed) and (not fail_check) and (not collision_check) and anyGate: # drone didn't arrive or crash or collide
                        #self.vel_sum = self.vel_sum / Waypoint_length
                        #print "Velocity Sum (Normalized): ", self.vel_sum
                        self.test_cost = self.Tf * self.quad.costValue / self.period_denum # time * cost 
                        print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost)
                        if self.flight_log:
                            f.write("\nDrone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost))


                    # flight_columns = ['true_init_x','true_init_y','true_init_z', 'noise_coeff', 'var_sum', 'diff_x', 'diff_y', 'diff_z', 'v_x', 'v_y', 'v_z', 'diff_phi', 'diff_theta', 'diff_psi', 
                    #                 'phi_dot', 'theta_dot', 'psi_dot', 'r_var', 'phi_var', 'theta_var', 'psi_var', 'Tf', 'MP_Method', 'Cost', 'Status', 'curr_idx']


                    self.write_stats(flight_columns,
                        [pos0[0], pos0[1], pos0[2], noise_coeff, covariance_sum, posf[0]-pos0[0], posf[1]-pos0[1], posf[2]-pos0[2], 
                        vel0[0], vel0[1], vel0[2], -phi_start, -theta_start, yawf-yaw0, ang_vel0[0], ang_vel0[1], ang_vel0[2],
                        prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3], self.Tf, method, self.test_cost, self.drone_status, self.curr_idx], flight_filename)

                    if track_completed or fail_check or collision_check or not anyGate: # drone arrived to the gate or crashed or collided                      
                        break

            self.curr_idx += 1

            if self.curr_idx >= max_iteration:
                break


    def collect_data(self, MP_list):
        path = self.base_path + 'images'
        t_coeff_upper = 1.2
        t_coeff_lower = 0.6
        phi_theta_range = 5.0
        psi_range = 5.0
        pos_range = 0.3
        gate_index = 0
        
        if self.flight_log:
            f=open(self.log_path, "a")

        for method in MP_list:
            #self.time_coeff = random.uniform(t_coeff_lower, t_coeff_upper)
            self.time_coeff = 0.7
            x_offset = pos_range*random.uniform(-1.0, 1.0)
            y_offset = pos_range*random.uniform(-1.0, 1.0)
            z_offset = pos_range*random.uniform(-1.0, 1.0)

            phi_start = phi_theta_range*random.uniform(-1.0,1.0)*np.pi/180
            theta_start = phi_theta_range*random.uniform(-1.0,1.0)*np.pi/180
            gate_target = self.track[gate_index]
            gate_psi = Rotation.from_quat([gate_target.orientation.x_val, gate_target.orientation.y_val, gate_target.orientation.z_val, gate_target.orientation.w_val]).as_euler('ZYX',degrees=False)[0]
            psi_start = psi_range*random.uniform(-1.0,1.0)*np.pi/180 + gate_psi - np.pi/2  #drone kapi karsisinde olacak sekilde durmali

            pos_offset = [x_offset, y_offset, z_offset]
            angle_start = [phi_start, theta_start, gate_psi, psi_start]
            self.fly_drone(f, method, pos_offset, angle_start)
            
        if self.flight_log:
            f.close()
                




    def update(self, mode):
        '''
        convetion of names:
        p_a_b: pose of frame b relative to frame a
        t_a_b: translation vector from a to b
        q_a_b: rotation quaternion from a to b
        o: origin
        b: UAV body frame
        g: gate frame
        '''

        # create and set pose for the quad
        #p_o_b, phi_base = racing_utils.geom_utils.randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)
        
        # create and set gate pose relative to the quad
        #p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV, correction)
        #self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        #min_vel, min_acc, min_jerk, pos_waypoint_interp, min_acc_stop, min_jerk_full_stop
        MP_list = ["min_vel", "min_acc", "min_jerk", "min_jerk_full_stop"]
        #MP_list = ["min_vel"]

        if self.with_gate:
            # gate_name = "gate_0"
            # self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            # self.client.simSetObjectPose(self.tgt_name, self.track[0], True)
            for i, gate in enumerate(self.track):
                #print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
                self.client.simSetObjectPose(self.tgt_name, gate, True)
        # request quad img from AirSim
        time.sleep(0.001)

        self.find_gate_edges()
        self.find_gate_distances()

        if self.flight_log:
            f=open(self.log_path, "w")
            f.write("\nMode %s \n" % mode)
            f.close()

        if mode == "DATA_COLLECTION":
            self.collect_data(MP_list)
        elif mode == "TEST":    
            self.mp_classifier.load_state_dict(torch.load(self.base_path + 'classifier_files/classifier_best.pt'))
            self.time_regressor = load(self.base_path + 'classifier_files/dt_regressor.sav')
            self.time_coeff = 0.6
            # self.mp_scaler = load(self.base_path + 'classifier_files/mp_scaler.bin')
            # self.time_scaler = load(self.base_path + 'classifier_files/time_scaler.bin')
            print "\n>>> PREDICTION MODE: DICE"
            self.test_algorithm(use_model=True, method="DICE")
            print "\n>>> PREDICTION MODE: MAX"
            self.test_algorithm(use_model=True, method="MAX")
            
            for method in MP_list:
                print "\n>>> TEST MODE: " + method
                self.test_algorithm(method = method)

            pickle.dump([self.test_states,self.test_arrival_time,self.test_costs, self.test_safe_counter], open(self.base_path + "files/test_variables.pkl","wb"), protocol=2)
        elif mode == "VISUALIZATION":
            self.visualize_drone()
        else:
            print "There is no such a mode called " + "'" + mode + "'"



    def visualize_drone(self):
        test_states, test_arrival_time, test_costs, test_safe_counter = pickle.load(open(self.base_path + "files/test_variables.pkl", "rb"))
        for mode in self.test_modes:
            print "\nDrone flies using the algorithm, ", mode
            self.client.simSetVehiclePose(self.drone_init, True)
            state_list = test_states[mode]
            for state in state_list:
                quad_pose = [state[0], state[1], state[2], -state[3], -state[4], state[5]]
                self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                time.sleep(0.001)
            print "Time of arrival is {0:.6} s.".format(test_arrival_time[mode])
            print "Total cost is {0:.6}".format(test_costs[mode])
            print "How many times has the drone been in safe mode: " + str(test_safe_counter[mode])
                    
        

    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
        if self.with_gate:
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
        else:
            self.tgt_name = "empty_target"

        if os.path.exists(self.csv_path):
            self.file = open(self.csv_path, "a")
        else:
            self.file = open(self.csv_path, "w")

    # write image to file
    def writeImgToFile(self, image_response):
        if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            print(image_response.height, image_response.width)
            cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)  # write to png
        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)

    def write_stats(self, stats_columns, stats, filename): #testno,stats_columns
        filename = os.path.join(self.base_path, filename)
        df_stats = pd.DataFrame([stats], columns=stats_columns)
        df_stats.to_csv(filename, mode='a', index=False, header=not os.path.isfile(filename))
