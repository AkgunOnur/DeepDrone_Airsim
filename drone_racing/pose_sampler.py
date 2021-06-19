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
import_path = os.path.join(curr_dir, '..')
sys.path.insert(0, import_path)
import trajectory_utils

# Extras for Perception
import torch
from torchvision import models, transforms
from torchvision.transforms.functional import normalize, resize, to_tensor
from PIL import Image


#Extras for Trajectory and Control
import pickle
import random
from numpy import zeros
from sklearn.externals.joblib import dump, load
from sklearn.preprocessing import StandardScaler
from quadrotor import *
import geom_utils
from geom_utils import QuadPose
from trajectory import Trajectory
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
                  'phi_dot', 'theta_dot', 'psi_dot', 'r_var', 'phi_var', 'theta_var', 'psi_var', 'Tf', 'v_average', 'MP_Method', 'Cost', 'Status', 'curr_idx']

class PoseSampler:
    def __init__(self, with_gate=True):
        self.num_samples = 1
        self.curr_idx = 0
        self.current_gate = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(1)        
        #self.client = airsim.MultirotorClient()
        self.configureEnvironment()

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

        self.state0 = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        self.test_states = {"MAX_SAFE":[], "MAX_NO_SAFE":[], "DICE_SAFE" :[], "DICE_NO_SAFE" :[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_jerk_full_stop":[]}
        self.test_costs = {"MAX_SAFE":0., "MAX_NO_SAFE":0., "DICE_SAFE" :0., "DICE_NO_SAFE" :0., "min_vel":0., "min_acc":0., "min_jerk":0., "min_jerk_full_stop":0.}
        self.test_arrival_time = {"MAX_SAFE":0., "MAX_NO_SAFE":0., "DICE_SAFE" :0., "DICE_NO_SAFE" :0., "min_vel":0., "min_acc":0., "min_jerk":0., "min_jerk_full_stop":0.}
        self.test_modes = ["MAX_SAFE", "MAX_NO_SAFE", "DICE_SAFE", "DICE_NO_SAFE", "min_vel", "min_acc", "min_jerk", "min_jerk_full_stop"]
        self.test_safe_counter = {"MAX_SAFE":0., "MAX_NO_SAFE":0., "DICE_SAFE" :0., "DICE_NO_SAFE" :0., "min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0}
        # self.test_modes = ["MAX", "min_vel"]
        self.test_covariances = {"MAX_SAFE":[], "MAX_NO_SAFE":[], "DICE_SAFE" :[], "DICE_NO_SAFE" :[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_jerk_full_stop":[]}
        self.test_methods = {"MAX_SAFE":[], "MAX_NO_SAFE":[], "DICE_SAFE" :[], "DICE_NO_SAFE" :[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_jerk_full_stop":[]}
        self.test_distribution_on_noise = {"MAX_SAFE":{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0, "safe_mode":0}, 
                                           "MAX_NO_SAFE":{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0}, 
                                           "DICE_SAFE" :{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0, "safe_mode":0}, 
                                           "DICE_NO_SAFE" :{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0}}
        self.test_distribution_off_noise = {"MAX_SAFE":{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0, "safe_mode":0}, 
                                           "MAX_NO_SAFE":{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0}, 
                                           "DICE_SAFE" :{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0, "safe_mode":0}, 
                                           "DICE_NO_SAFE" :{"min_vel":0, "min_acc":0, "min_jerk":0, "min_jerk_full_stop":0}}
        self.test_number = "0_0"
        self.time_coeff = 0.
        self.quad_period = 0.
        self.drone_status = ""


        #----- Motion planning parameters -----------------------------
        self.MP_methods = {"pos_waypoint_timed":1, "pos_waypoint_interp":2, "min_vel":3, "min_acc":4, "min_jerk":5, "min_snap":6,
                              "min_acc_stop":7, "min_jerk_stop":8, "min_snap_stop":9, "min_jerk_full_stop":10, "min_snap_full_stop":11,
                              "pos_waypoint_arrived":12}
        self.MP_names = ["hover", "pos_waypoint_timed", "pos_waypoint_interp", "min_vel", "min_acc", "min_jerk", "min_snap",
                         "min_acc_stop", "min_jerk_stop", "min_snap_stop", "min_jerk_full_stop", "min_snap_full_stop","safe_mode"]
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

        self.drone_init = Pose(Vector3r(0.,30.,-2), Quaternionr(quat0[0],quat0[1],quat0[2],quat0[3]))
        self.gate = [Pose(Vector3r(0.,2.,-2.), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
                     Pose(Vector3r(2.,-5.,-2.4), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
                     Pose(Vector3r(4.,-13.,-3.1), Quaternionr(quat3[0],quat3[1],quat3[2],quat3[3])),
                     Pose(Vector3r(7.,-20.,-3.75), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3]))]
                     # Pose(Vector3r(9.,-20.,-4.), Quaternionr(quat5[0],quat5[1],quat5[2],quat5[3]))]

        # Previous gates
        # self.gate = [Pose(Vector3r(0.,20.,-2.), Quaternionr(quat1[0],quat1[1],quat1[2],quat1[3])),
        #              Pose(Vector3r(4.,10.,-1), Quaternionr(quat2[0],quat2[1],quat2[2],quat2[3])),
        #              Pose(Vector3r(10.,0.,-1.5), Quaternionr(quat4[0],quat4[1],quat4[2],quat4[3]))]

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

        self.circle_track = trajectory_utils.generate_gate_poses(num_gates=6,
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


    def check_completion(self, index, quad_pose, eps=0.45):
        x, y, z = quad_pose[0], quad_pose[1], quad_pose[2]

        xd = self.track[index].position.x_val
        yd = self.track[index].position.y_val
        zd = self.track[index].position.z_val
        psid = Rotation.from_quat([self.track[index].orientation.x_val, self.track[index].orientation.y_val, 
                                   self.track[index].orientation.z_val, self.track[index].orientation.w_val]).as_euler('ZYX',degrees=False)[0]

        target = [xd, yd, zd, psid] 
        check_arrival = False


        if ( (abs(abs(xd)-abs(x)) <= eps) and (abs(abs(yd)-abs(y)) <= eps) and (abs(abs(zd)-abs(z)) <= eps)):
            self.quad.calculate_cost(target=target, final_calculation=True)
            check_arrival = True

        return check_arrival

    def fly_through_gates(self):
        x_offset, y_offset, z_offset = 0, 0, 0
        phi_start, theta_start, gate_psi, psi_start = 0, 0, 0, 0

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
        gate_completed = False
        fail_check = False
        init_start = True
        collision_check = False

        noise_on = False

        covariance_sum = 0.
        prediction_std = [0., 0., 0., 0.]
        sign_coeff = 0.
        previous_idx = 0
        index = 0
        v_average = 5.0
        method = "min_vel"

        final_target = [self.track[-1].position.x_val, self.track[-1].position.y_val, self.track[-1].position.z_val]

        covariance_list = []

        while((not track_completed) and (not fail_check)):

            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]

            with torch.no_grad():                   
                self.trajSelect[0] = self.MP_methods[method]
                self.trajSelect[1] = 2
                self.trajSelect[2] = 1
                # self.Tf = self.time_coeff*abs(pose_gate_body[0][0]) + 0.1
                
                # Trajectory generate
                waypoint_world = np.array([self.track[index].position.x_val, self.track[index].position.y_val, self.track[index].position.z_val])
                pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                ang_vel0 = [self.quad.state[9], self.quad.state[10], self.quad.state[11]]
                #acc0 = [0., 0., 0.]
                yaw0 = self.quad.state[5]

                yaw_diff = self.track[index].orientation.z_val
                posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
                yawf = self.track[index].orientation.z_val - np.pi/2 #(self.quad.state[5]+yaw_diff) + np.pi/2

                time_list = np.hstack((0., self.Tf)).astype(float)
                waypoint_list = np.vstack((pos0, posf)).astype(float)
                yaw_list = np.hstack((yaw0, yawf)).astype(float)

                newTraj = Trajectory(self.trajSelect, self.quad.state, time_list, waypoint_list, yaw_list, v_average=v_average) 
                self.Tf = newTraj.t_wps[1]

                #print "MP algorithm: " + method 
                #print "V_average: {0:.3} m/s, Estimated time of arrival: {1:.3} s.".format(v_average, self.Tf)                       
                #print "Gate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)            
                #print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])
     

                flight_period = self.Tf
                Waypoint_length = flight_period // self.dtau
                t_list = linspace(0, flight_period, num = Waypoint_length)

                if init_start:
                    t_list = linspace(0, flight_period, num = Waypoint_length)
                    init_start = False
                else:
                    t_list = linspace(0.1*flight_period, 1*flight_period, num = 1*Waypoint_length)


                #self.vel_sum = 0.
                self.quad.costValue = 0.
                self.drone_status = "SUCCESS"
                # Call for Controller
                for ind, t_current in enumerate(t_list): 
                    pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)

                    # self.vel_sum += (self.quad.state[6]**2+self.quad.state[7]**2+self.quad.state[8]**2)
                    
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
                        break 
                    elif fail_check:
                        self.drone_status = "CRASH"
                        self.quad.costValue = 1e12
                        self.test_cost = self.quad.costValue
                        print "Drone has crashed! Current cost: {0:.6}".format(self.test_cost)
                        break 

                    check_arrival = self.check_completion(index, quad_pose)

                    if check_arrival: # drone arrives to the gate
                        gate_completed = True
                        index += 1
                        #self.vel_sum = self.vel_sum / (ind + 1)
                        #print "Velocity Sum (Normalized): ", self.vel_sum
                        self.test_cost = self.quad.costValue / (v_average**2) / 1e3 
                        print "Drone has gone through the {0}. gate.".format(index) 
                        break        


                    if (not gate_completed) and (not fail_check) and (not collision_check): # drone didn't arrive or crash or collide
                        #self.vel_sum = self.vel_sum / Waypoint_length
                        #print "Velocity Sum (Normalized): ", self.vel_sum
                        self.test_cost = self.quad.costValue / (v_average**2) / 1e3 
                        #print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost)


                    # flight_columns = ['true_init_x','true_init_y','true_init_z', 'noise_coeff', 'var_sum', 'diff_x', 'diff_y', 'diff_z', 'v_x', 'v_y', 'v_z', 'diff_phi', 'diff_theta', 'diff_psi', 
                    #                 'phi_dot', 'theta_dot', 'psi_dot', 'r_var', 'phi_var', 'theta_var', 'psi_var', 'Tf', 'MP_Method', 'Cost', 'Status', 'curr_idx']


                    if gate_completed or fail_check or collision_check: # drone arrived to the gate or crashed or collided                      
                        break

                if index == len(self.gate):
                    track_completed = True
                    print "Drone has completed the track!"
                    break



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
        
        #self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        #min_vel, min_acc, min_jerk, pos_waypoint_interp, min_acc_stop, min_jerk_full_stop
        MP_list = ["min_acc", "min_jerk", "min_jerk_full_stop", "min_vel"]
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


        if mode == "FLY":
            self.fly_through_gates()
        else:
            print "There is no such a mode called " + "'" + mode + "'"

                    
    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
        if self.with_gate:
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
        else:
            self.tgt_name = "empty_target"


    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)

