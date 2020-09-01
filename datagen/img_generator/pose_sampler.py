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
#            "min_jerk_full_stop", "min_snap_full_stop", "pos_waypoint_arrived","pos_way_timed", "pos_waypoint_interp"] 


flight_columns = ['true_init_x','true_init_y','true_init_z','diff_x', 'diff_y', 'diff_z', 'diff_phi', 'diff_theta', 'diff_psi', 
                  'r_std', 'phi_std', 'theta_std', 'psi_std', 'Tf', 'Backstepping_method', 'Cost']

mp_list = []

flight_filename = 'data.csv'

class PoseSampler:
    def __init__(self, num_samples, dataset_path, flight_log, with_gate=True):
        self.num_samples = num_samples
        self.base_path = dataset_path
        self.csv_path = os.path.join(self.base_path, 'files/gate_training_data.csv')
        self.curr_idx = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(4)
        self.client = airsim.MultirotorClient()
        self.configureEnvironment()
        self.log_path = os.path.join(self.base_path, 'files/flight_log.txt')
        self.flight_log = flight_log

        #----- Classifier/Regressor parameters -----------------------------
        self.mp_classifier = Net()
        self.t_or_s_classifier = Net()
        self.speed_regressor = Net_Regressor()
        self.time_regressor = Net_Regressor()
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
        self.quadrotor_freq = int(1. / self.dtau)
        self.method = "MAX"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = Net()
        self.model = self.model.to(self.device)
        # self.model.load_state_dict(torch.load('best_model.pt'))
        # self.model.eval()
        self.state0 = [0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0.]
        self.test_states = []
        self.time_coeff = 0.
        self.quad_period = 0.
        self.Controller_states = {"Backstepping_1":[], "Backstepping_2":[], "Backstepping_3":[], "Backstepping_4":[]}
        self.Controller_Cost = {"Backstepping_1": 1e9, "Backstepping_2":1e9, "Backstepping_3":1e9, "Backstepping_4":1e9}


        #----- Motion planning parameters -----------------------------
        self.MP_methods = {"pos_way_timed":1, "pos_waypoint_interp":2, "min_vel":3, "min_acc":4, "min_jerk":5, "min_snap":6,
                              "min_acc_stop":7, "min_jerk_stop":8, "min_snap_stop":9, "min_jerk_full_stop":10, "min_snap_full_stop":11,
                              "pos_waypoint_arrived":12}
        self.MP_names = ["hover", "pos_way_timed", "pos_waypoint_interp", "min_vel", "min_acc", "min_jerk", "min_snap",
                         "min_acc_stop", "min_jerk_stop", "min_snap_stop", "min_jerk_full_stop", "min_snap_full_stop","pos_waypoint_arrived"]
        self.MP_cost = {"pos_way_timed":1e9, "pos_waypoint_interp":1e9, "min_vel":1e9, "min_acc":1e9, "min_jerk":1e9, "min_snap":1e9,
                        "min_acc_stop":1e9, "min_jerk_stop":1e9, "min_snap_stop":1e9, "min_jerk_full_stop":1e9, "min_snap_full_stop":1e9,
                        "pos_waypoint_arrived":1e9}
        self.MP_states = {"pos_way_timed":[], "pos_waypoint_interp":[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_snap":[],
                          "min_acc_stop":[], "min_jerk_stop":[], "min_snap_stop":[], "min_jerk_full_stop":[], "min_snap_full_stop":[],
                          "pos_waypoint_arrived":[]}

        self.MP_types = ["Time", "Speed"]

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
        self.lstmR.load_state_dict(torch.load(self.base_path + 'weights/16_hidden_lstm_R_PG.pth'))   
        self.lstmR.eval() 


        self.angle_lim = 0.
        self.pos_lim = 0.
        self.brightness = random.uniform(0,0.5)
        self.contrast = random.uniform(0,2)
        self.saturation = random.uniform(0,2)
        self.blur_coeff = 0.
        self.blur_range = 0

        self.transformation = transforms.Compose([
                            transforms.Resize([200, 200]),
                            transforms.Lambda(self.gaussian_blur),
                            #transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
                            transforms.ToTensor()])

        self.drone_init = Pose(Vector3r(0.,0.,-2), Quaternionr(0., 0., -0.70710678, 0.70710678))
        self.gate = [Pose(Vector3r(0.,-5.,-2.), Quaternionr(0., 0., 0., 1.)),
                     Pose(Vector3r(2.,-8.,-2.5), Quaternionr(0., 0., 0.25881905, 0.96592583)),
                     Pose(Vector3r(4.,-10.,-3.), Quaternionr(0., 0., 0.38268343, 0.92387953)),
                     Pose(Vector3r(6.,-12.,-3.5), Quaternionr(0., 0., 0.5, 0.8660254)),
                     Pose(Vector3r(8.,-14.,-4), Quaternionr(0., 0., 0.70710678, 0.70710678))]

        self.drone_init_2 = Pose(Vector3r(0.,0.,-2), Quaternionr(0., 0., -0.70710678, 0.70710678))
        self.gate_2 = [Pose(Vector3r(0.,-5.,-2.), Quaternionr(0., 0., 0., 1.)),
                    Pose(Vector3r(-2.,-10.,-2.5), Quaternionr(0., 0., -0.25881905, 0.96592583))]


        self.race_course_radius = 16
        self.radius_noise = 0.1
        self.height_range = [0, -1.0]
        self.direction = 0

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
            return pred[0]

        elif method=="DICE" and isClassifier:
            probs = softmax(outputs).cpu().detach().numpy()[0]
            pred_index = np.random.choice([0, 1, 2, 3], 1, p=probs)[0]
            return pred_index

        return output.item() # if mode is regression

    def test_algorithm(self):
        pose_prediction = np.zeros((1000,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)

        gate_target = self.track[0]
        gate_psi = Rotation.from_quat([gate_target.orientation.x_val, gate_target.orientation.y_val, gate_target.orientation.z_val, gate_target.orientation.w_val]).as_euler('ZYX',degrees=False)[0]
        psi_start = gate_psi - np.pi/2  #drone kapi karsisinde olacak sekilde durmali

        #if drone is at initial point
        quad_pose = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, 0., 0., psi_start]
        self.state0 = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, 0., 0., psi_start, 0., 0., 0., 0., 0., 0.]

        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.quad = Quadrotor(self.state0)
        
        self.curr_idx = 0
        self.test_states.append(self.quad.state)

        self.xd_ddot_pr = 0.
        self.yd_ddot_pr = 0.
        self.xd_dddot_pr = 0.
        self.yd_dddot_pr = 0.
        self.psid_pr = 0.
        self.psid_dot_pr = 0.

        self.blur_range = 0.01
        self.blur_coeff = random.uniform(0, self.blur_range)

        track_completed = False
        fail_check = False

        if self.flight_log:
            f=open(self.log_path, "a")


        while((not track_completed) and (not fail_check)):
            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            #if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
            img =  Image.fromarray(img_rgb)
            image = self.transformation(img)
            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]


            with torch.no_grad():   
                # Determine Gat location with Neural Networks
                pose_gate_body = self.Dronet(image)
                
                for i,num in enumerate(pose_gate_body.reshape(-1,1)):
                    #print(num, i , self.curr_idx)
                    pose_prediction[self.curr_idx][i] = num.item()

                if self.curr_idx >= 11:
                    pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                    
                    for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
                        prediction_std[i] = p_g_c.item()
            
                    # Gate ground truth values will be implemented
                    pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
                    prediction_std = np.clip(prediction_std, 0, prediction_std)
                    prediction_std = prediction_std.ravel()
                    covariance_sum = np.sum(prediction_std)
                    
                    # Trajectory generate
                    waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
                    pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                    posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
                    yaw0 = self.quad.state[5]
                    yaw_diff = pose_gate_body[3][0]
                    yawf = (self.quad.state[5]+yaw_diff) + np.pi/2

                    print "\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)
                    print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])

                    if self.flight_log:
                        f.write("\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))
                        f.write("Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]))


                    
                    # # diff_x, diff_y, diff_z, diff_phi, diff_theta, diff_psi, std_r, std_phi, std_theta, std_psi
                    # X_test = np.array([posf[0]-pos0[0], posf[1]-pos0[1], posf[2]-pos0[2], -self.quad.state[3], -self.quad.state[4], yawf-yaw0,
                    #             prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]]).reshape(1,-1)

                    # X_mp_test = self.mp_scaler.transform(X_test)
                    # X_ts_test = self.t_or_s_scaler.transform(X_test)
                    # X_time_test = self.time_scaler.transform(X_test)
                    # X_speed_test =  self.speed_scaler.transform(X_test)

                    # time_or_speed = self.predict(X_ts_test, model=self.t_or_s_classifier, isClassifier=True)
                    # mp_method = self.predict(X_mp_test, model=self.mp_classifier, isClassifier=True)

                    # self.trajSelect[0] = mp_method + 3 # this is the offset, i.e. 0-min_vel corresponds 3 here
                    # self.trajSelect[1] = 2
                    # self.trajSelect[2] = time_or_speed

                    # print "Predicted MP Algorithm: ", self.MP_names[int(self.trajSelect[0])]
                    # print "Predicted MP Type: ", self.MP_types[int(self.trajSelect[2])]


                    time_or_speed = 0
                    self.Tf = 0.3*pose_gate_body[0][0]
                    newTraj = Trajectory(self.trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf)
                    print "ETA: " + str(self.Tf) + " s."
                    # if time_or_speed == 0:
                    #     self.Tf = self.predict(X_time_test, model=self.time_regressor, isClassifier=False)
                    #     newTraj = Trajectory(self.trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf)
                    #     print "Time based trajectory, T: {0:.3}".format(newTraj.t_wps[1])
                    #     print "Predicted Time Length: {0:.3}".format(self.Tf)
                    #     if self.flight_log:
                    #         f.write("Time based trajectory, T: {0:.3}".format(newTraj.t_wps[1]))
                    # else:
                    #     self.v_average = self.predict(X_speed_test, model=self.speed_regressor, isClassifier=False)
                    #     newTraj = Trajectory(self.trajSelect, self.quad.state, 1.0, pos0, posf, yaw0, yawf, v_average=self.v_average)
                    #     print "Velocity based trajectory, Predicted V average: {0:.3}, T: {1:.3}".format(self.v_average, newTraj.t_wps[1])
                    #     if self.flight_log:
                    #         f.write("Velocity based trajectory, Predicted V average: {0:.3}, T: {1:.3}".format(self.v_average, newTraj.t_wps[1]))

                    Waypoint_length = self.Tf // self.dtau
                    t_list = linspace(0, self.Tf, num = Waypoint_length)

                    vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                    #acc0 = [float(self.quad.state[6]-self.x_dot_pr)/self.dtau, float(self.quad.state[7]-self.y_dot_pr)/self.dtau, float(self.quad.state[8]-self.z_dot_pr)/self.dtau]
                    acc0 = [0., 0., 0.]
                    velf = [float(posf[0]-pos0[0])/self.Tf, float(posf[1]-pos0[1])/self.Tf, float(posf[2]-pos0[2])/self.Tf]
                    accf = [0., 0., 0.]
                    
                    mp_algorithm = MyTraj(gravity = -9.81)
                    traj = mp_algorithm.givemetraj(pos0, vel0, acc0, posf, velf, accf, self.Tf)
                    
                    Waypoint_length = int(self.Tf / self.dtau)
                    t_list = linspace(0., self.Tf, num = Waypoint_length)
                                        
                    
                    # Call for Controller
                    for t_current in t_list: 

                        pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)
                        pos_des, vel_des, acc_des = mp_algorithm.givemepoint(traj, t_current)
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

                        fail_check = self.quad.simulate(self.dtau, current_traj, prediction_std)

                        quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
                        self.test_states.append(self.quad.state)
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


                        if fail_check:
                            self.test_cost = self.quad.costValue
                            print "Drone has crashed! Current cost: {0:.6}".format(self.test_cost)
                            if self.flight_log:
                                f.write("Drone has crashed! Current cost: {0:.6}".format(self.test_cost))
                            break 

                        check_arrival, on_road = self.check_completion(quad_pose, -1, test_control = True)

                        if check_arrival: # drone arrives to the gate
                            track_completed = True
                            self.test_cost = newTraj.t_wps[1] * self.quad.costValue
                            print "Drone has arrived finished the lap. Current cost: {0:.6}".format(self.test_cost) 
                            if self.flight_log:
                                f.write("Drone has arrived finished the lap. Current cost: {0:.6}".format(self.test_cost))
                            break        
                        elif not on_road: #drone can not complete the path, but still loop should be ended
                            track_completed = True
                            self.test_cost = self.quad.costValue
                            print "Drone is out of the path. Current cost: {0:.6}".format(self.test_cost)
                            if self.flight_log:
                                f.write("Drone is out of the path. Current cost: {0:.6}".format(self.test_cost))
                            break


                    if (not track_completed) and (not fail_check): # drone didn't arrive or crash
                        self.test_cost = newTraj.t_wps[1] * self.quad.costValue
                        print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost)
                        if self.flight_log:
                            f.write("Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost))


                    if track_completed or fail_check: # drone arrived to the gate or crashed                        
                        break

            self.curr_idx += 1

        if self.flight_log:
            f.close()


    def check_completion(self, quad_pose, gate_index, eps=0.3, test_control = False):
        x, y, z = quad_pose[0], quad_pose[1], quad_pose[2]

        if test_control:
            xd = self.track[-1].position.x_val
            yd = self.track[-1].position.y_val
            zd = self.track[-1].position.z_val
            psid = Rotation.from_quat([self.track[-1].orientation.x_val, self.track[-1].orientation.y_val, 
                                       self.track[-1].orientation.z_val, self.track[-1].orientation.w_val]).as_euler('ZYX',degrees=False)[0]

            xd_prev = self.drone_init.position.x_val
            yd_prev = self.drone_init.position.y_val
            zd_prev = self.drone_init.position.z_val
        else:
            xd = self.track[gate_index].position.x_val
            yd = self.track[gate_index].position.y_val
            zd = self.track[gate_index].position.z_val
            psid = Rotation.from_quat([self.track[gate_index].orientation.x_val, self.track[gate_index].orientation.y_val, 
                                       self.track[gate_index].orientation.z_val, self.track[gate_index].orientation.w_val]).as_euler('ZYX',degrees=False)[0]

            if gate_index == 0:
                xd_prev = self.drone_init.position.x_val
                yd_prev = self.drone_init.position.y_val
                zd_prev = self.drone_init.position.z_val
            else:
                xd_prev = self.track[gate_index-1].position.x_val
                yd_prev = self.track[gate_index-1].position.y_val
                zd_prev = self.track[gate_index-1].position.z_val


        target = [xd, yd, zd, psid] 
        div_coef = 3
        check_arrival = False

        

        init_distance = np.sqrt((xd_prev-xd)**2 + (yd_prev-yd)**2 + (zd_prev-zd)**2)
        current_distance = np.sqrt((x-xd)**2 + (y-yd)**2 + (z-zd)**2)

        on_road = (abs(xd_prev)-div_coef*eps <= abs(x) <= abs(xd)+div_coef*eps) and (abs(yd_prev)-div_coef*eps <= abs(y) <= abs(yd)+div_coef*eps) and (abs(zd_prev)-div_coef*eps <= abs(z) <= abs(zd)+div_coef*eps)

        if ((abs(xd)-abs(x) <= eps) and (abs(yd)-abs(y) <= eps) and (abs(zd)-abs(z) <= eps)):
            self.quad.calculate_cost(target)
            check_arrival = True

        if not on_road:
            self.quad.calculate_cost(target, off_road = True)


        return check_arrival, on_road


    def fly_drone(self, gate_index, method):
        pose_prediction = np.zeros((2000,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)

        phi_theta_range = 10.0
        psi_range = 10.0
        pos_range = 0.3

        x_range = pos_range*random.uniform(-1.0, 1.0)
        y_range = pos_range*random.uniform(-1.0, 1.0)
        z_range = pos_range*random.uniform(-1.0, 1.0)

        phi_start = phi_theta_range*random.uniform(-1.0,1.0)*np.pi/180
        theta_start = phi_theta_range*random.uniform(-1.0,1.0)*np.pi/180
        gate_target = self.track[gate_index]
        gate_psi = Rotation.from_quat([gate_target.orientation.x_val, gate_target.orientation.y_val, gate_target.orientation.z_val, gate_target.orientation.w_val]).as_euler('ZYX',degrees=False)[0]
        psi_start = psi_range*random.uniform(-1.0,1.0)*np.pi/180 + gate_psi - np.pi/2  #drone kapi karsisinde olacak sekilde durmali

        if gate_index == 0: #if drone is at initial point
            quad_pose = [self.drone_init.position.x_val+x_range, self.drone_init.position.y_val+y_range, self.drone_init.position.z_val+z_range, -phi_start, -theta_start, psi_start]
            self.state0 = [self.drone_init.position.x_val+x_range, self.drone_init.position.y_val+y_range, self.drone_init.position.z_val+z_range, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]
            true_init_pos = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val]
        else:
            quad_pose = [self.track[gate_index-1].position.x_val+x_range, self.track[gate_index-1].position.y_val+y_range, self.track[gate_index-1].position.z_val+z_range, -phi_start, -theta_start, psi_start]
            self.state0 = [self.track[gate_index-1].position.x_val+x_range, self.track[gate_index-1].position.y_val+y_range, self.track[gate_index-1].position.z_val+z_range, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]
            true_init_pos = [self.track[gate_index-1].position.x_val, self.track[gate_index-1].position.y_val, self.track[gate_index-1].position.z_val]


        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.quad = Quadrotor(self.state0)
        # this is only used for yaw motion planning
        self.trajSelect[0] = 3 # minimum velocity
        self.trajSelect[1] = 2 
        self.trajSelect[2] = 0
        self.curr_idx = 0
        self.Controller_states[method].append(self.quad.state)
        #self.MP_states[algorithm].append(self.quad.state)

        self.xd_ddot_pr = 0.
        self.yd_ddot_pr = 0.
        self.xd_dddot_pr = 0.
        self.yd_dddot_pr = 0.
        self.psid_pr = 0.
        self.psid_dot_pr = 0.

        self.blur_range = 10. # blurring coefficient 
        self.blur_coeff = random.uniform(0, self.blur_range)

        initial_start = True

        print "\n>>>Controller Method: ", method
        track_completed = False
        fail_check = False

        if self.flight_log:
            f=open(self.log_path, "a")
            f.write("\n>>>Controller Method: %s " %method)

        while((not track_completed) and (not fail_check)):
            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            #if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
            img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
            img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
            img =  Image.fromarray(img_rgb)
            image = self.transformation(img)
            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]


            with torch.no_grad():   
                # Determine Gat location with Neural Networks
                pose_gate_body = self.Dronet(image)
                
                for i,num in enumerate(pose_gate_body.reshape(-1,1)):
                    #print(num, i , self.curr_idx)
                    pose_prediction[self.curr_idx][i] = num.item()

                if self.curr_idx >= 11:
                    pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                    
                    for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
                        prediction_std[i] = p_g_c.item()
            
                    # Gate ground truth values will be implemented
                    pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
                    prediction_std = np.clip(prediction_std, 0, prediction_std)
                    prediction_std = prediction_std.ravel()
                    covariance_sum = np.sum(prediction_std)
                    
                    # r,theta,psi,phi = pose_gate_body[0][0],pose_gate_body[1][0],pose_gate_body[2][0],pose_gate_body[3][0] # notation is different. In our case, phi equals to psi
                    # q1,q2,q3,q4 = R.from_euler('ZYX',[self.quad.state[5], self.quad.state[4], self.quad.state[3]], degrees=False).as_quat()
                    # quad_pose = Pose(Vector3r(self.quad.state[0], self.quad.state[1],self.quad.state[2]),Quaternionr(q1,q2,q3,q4))
                    # estimation = geom_utils.debugGatePoses(quad_pose , r, theta, psi)
                    
                    # Trajectory generate
                    self.Tf = self.time_coeff*pose_gate_body[0][0] # T=r*0.5
                    waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
                    pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                    vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                    #acc0 = [float(self.quad.state[6]-self.x_dot_pr)/self.dtau, float(self.quad.state[7]-self.y_dot_pr)/self.dtau, float(self.quad.state[8]-self.z_dot_pr)/self.dtau]
                    acc0 = [0., 0., 0.]
                    yaw0 = self.quad.state[5]

                    posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
                    velf = [float(posf[0]-pos0[0])/self.Tf, float(posf[1]-pos0[1])/self.Tf, float(posf[2]-pos0[2])/self.Tf]
                    accf = [0., 0., 0.]
                    yaw_diff = pose_gate_body[3][0]
                    yawf = (self.quad.state[5]+yaw_diff) + np.pi/2

                    newTraj = Trajectory(self.trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf) # this is only used for yaw motion planning

                    print "\nTime of arrival: {0:.3} s., time coefficient: {1:.3}".format(self.Tf, self.time_coeff)
                    print "Gate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)
                    print "Gate Ground truth, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val, (gate_psi-np.pi/2)*180/np.pi)
                    print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])
                    print "Blurring coefficient: {0:.3}/{1:.3}".format(self.blur_coeff,self.blur_range)


                    if self.flight_log:
                        f.write("\nTime of arrival: {0:.3}, time coefficient: {1:.3}".format(self.Tf, self.time_coeff))
                        f.write("Gate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))
                        f.write("Gate Ground truth, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val, (gate_psi-np.pi/2)*180/np.pi))
                        f.write("Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]))
                        f.write("Blurring coefficient: {0:.3}".format(self.blur_coeff))
                    

                    # if time_or_speed == 0:
                    #     newTraj = Trajectory(self.trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf)
                    #     print "Time based trajectory, T: {0:.3}".format(newTraj.t_wps[1])
                    #     if self.flight_log:
                    #         f.write("Time based trajectory, T: {0:.3}".format(newTraj.t_wps[1]))
                    # else:
                    #     newTraj = Trajectory(self.trajSelect, self.quad.state, 1.0, pos0, posf, yaw0, yawf, v_average=self.v_average)
                    #     print "Velocity based trajectory, V_avg: {0:.3}, T: {1:.3}".format(self.v_average, newTraj.t_wps[1])
                    #     if self.flight_log:
                    #         f.write("Velocity based trajectory, V_avg: {0:.3}, T: {1:.3}".format(self.v_average, newTraj.t_wps[1]))

                    # if initial_start:
                    #     init_time = 0.
                    #     initial_start = False
                    # else:
                    #     init_time = newTraj.t_wps[1] / 6.0

                    # Waypoint_length = (newTraj.t_wps[1] - init_time) // self.dtau
                    # t_list = linspace(init_time, newTraj.t_wps[1], num = Waypoint_length)


                    mp_algorithm = MyTraj(gravity = -9.81)
                    traj = mp_algorithm.givemetraj(pos0, vel0, acc0, posf, velf, accf, self.Tf)
                    
                    Waypoint_length = int(self.Tf / self.dtau)
                    t_list = linspace(0., self.Tf, num = Waypoint_length)
                    

                    for t_current in t_list: 
                        pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)

                        pos_des, vel_des, acc_des = mp_algorithm.givemepoint(traj, t_current)

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

                        fail_check = self.quad.simulate(self.dtau, current_traj, method, prediction_std)

                        quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
                        self.Controller_states[method].append(self.quad.state)
                        self.client.simSetVehiclePose(QuadPose(quad_pose), True)


                        self.xd_ddot_pr = xd_ddot
                        self.yd_ddot_pr = yd_ddot
                        self.xd_dddot_pr = xd_dddot
                        self.yd_dddot_pr = yd_dddot
                        self.psid_pr = psid
                        self.psid_dot_pr = psid_dot


                        if fail_check:
                            self.Controller_Cost[method] = self.quad.costValue
                            print "Drone has crashed! Current cost: {0:.6}".format(self.Controller_Cost[method])
                            if self.flight_log:
                                f.write("Drone has crashed! Current cost: {0:.6}".format(self.Controller_Cost[method]))
                            break 

                        check_arrival, on_road = self.check_completion(quad_pose, gate_index=gate_index)

                        if check_arrival: # drone arrives to the gate
                            track_completed = True
                            self.Controller_Cost[method] = self.Tf * self.quad.costValue + 1e3*np.sum(prediction_std)
                            print "Drone has arrived to the {0}. gate. Current cost: {1:.6}".format(gate_index+1, self.Controller_Cost[method]) 
                            if self.flight_log:
                                f.write("Drone has arrived to the {0}. gate. Current cost: {1:.6}".format(gate_index+1, self.Controller_Cost[method]))
                            break        
                        elif not on_road: #drone can not complete the path, but still loop should be ended
                            track_completed = True
                            self.Controller_Cost[method] = self.quad.costValue
                            print "Drone is out of the path. Current cost: {0:.6}".format(self.Controller_Cost[method])
                            if self.flight_log:
                                f.write("Drone is out of the path. Current cost: {0:.6}".format(self.Controller_Cost[method]))
                            break


                    if (not track_completed) and (not fail_check): # drone didn't arrive or crash
                        self.Controller_Cost[method] = self.Tf * self.quad.costValue + 1e3*np.sum(prediction_std)
                        print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.Controller_Cost[method])
                        if self.flight_log:
                            f.write("Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.Controller_Cost[method]))

                    self.write_stats(flight_columns,
                        [true_init_pos[0], true_init_pos[1], true_init_pos[2], posf[0]-pos0[0], posf[1]-pos0[1], posf[2]-pos0[2], -phi_start, -theta_start, yawf-yaw0,
                         prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3], self.Tf, method, self.Controller_Cost[method]], flight_filename)
                    #print "Flight data is written to the file"

                    self.quad.costValue = 0.
                    if track_completed or fail_check: # drone arrived to the gate or crashed                        
                        break

            self.curr_idx += 1

        if self.flight_log:
            f.close()


        


    def collect_data(self, Method_List):
        path = self.base_path + 'images'

        # v_upper = 1.2
        # v_lower = 0.9
        t_coeff_upper = 1.0 
        t_coeff_lower = 0.3
        N_iter = 100

        gate_index = 0
        

        #for gate_index in range(len(self.track)):
        
        for k in range(N_iter):
            print "\nIteration: {0}/{1}".format(k+1, N_iter)
            for method in Method_List:
                self.time_coeff = random.uniform(t_coeff_lower, t_coeff_upper)
                self.fly_drone(gate_index, method) 
                # self.MP_cost[algorithm] = 0.
                # time_or_speed = 1 # a velocity based trajectory
                # self.v_average = random.uniform(v_lower, v_upper)
                # self.fly_drone(gate_index, time_or_speed, algorithm)

                # self.MP_cost[algorithm] = 0.
                # time_or_speed = 0 # a time based trajectory
                # self.time_coeff = random.uniform(t_coeff_lower, t_coeff_upper)
                # self.fly_drone(gate_index, time_or_speed, algorithm)
                




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

        Modes = ["DATA_COLLECTION", "TEST"]
        Method_List = ["Backstepping_1","Backstepping_2","Backstepping_3","Backstepping_4"]
        # MP_list = ["min_vel", "min_acc", "min_jerk", "min_snap"] 
        # MP_list = MP_list[::-1]
        if self.with_gate:
            gate_name = "gate_0" 
            gate = self.track[0]
            self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            self.client.simSetObjectPose(self.tgt_name, gate, True)
            # for i, gate in enumerate(self.track):
            #     #print ("gate: ", gate)
            #     gate_name = "gate_" + str(i)
            #     self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            #     self.client.simSetObjectPose(self.tgt_name, gate, True)
            # self.client.plot_tf([p_o_g], duration=20.0)
        # request quad img from AirSim
        time.sleep(0.001)

        if self.flight_log:
            f=open(self.log_path, "a")
            f.write("\nMode %s \n" % Modes[mode])
            f.close()

        if Modes[mode] == "DATA_COLLECTION":
            self.collect_data(Method_List)
        elif Modes[mode] == "TEST":
            self.mp_classifier.load_state_dict(torch.load(self.base_path + 'classifier_files/mp_classifier_best_model.pt'))
            self.t_or_s_classifier.load_state_dict(torch.load(self.base_path + 'classifier_files/t_or_s_classifier_best_model.pt'))
            self.speed_regressor.load_state_dict(torch.load(self.base_path + 'classifier_files/speed_regressor_best_model.pt'))
            self.time_regressor.load_state_dict(torch.load(self.base_path + 'classifier_files/time_regressor_best_model.pt'))
            self.mp_scaler = load(self.base_path + 'classifier_files/mp_scaler.bin')
            self.t_or_s_scaler = load(self.base_path + 'classifier_files/t_or_s_scaler.bin')
            self.speed_scaler = load(self.base_path + 'classifier_files/speed_scaler.bin')
            self.time_scaler = load(self.base_path + 'classifier_files/time_scaler.bin')

            self.test_algorithm()

            
        
        
        

    def visualize_drone(self, MP_list):
        for algorithm in MP_list:
            print "Drone flies by the algorithm, ", algorithm
            self.client.simSetVehiclePose(self.drone_init, True)
            state_list = self.MP_states[algorithm]
            for state in state_list:
                quad_pose = [state[0], state[1], state[2], -state[3], -state[4], state[5]]
                self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                time.sleep(0.001)

    def get_video(self, algorithm):

        pathIn= self.base_path + 'images/'
        pathOut = self.base_path + algorithm + '_video.avi'
        fps = 0.5
        frame_array = []
        files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]#for sorting the file names properly
        files.sort(key = lambda x: x[5:-4])
        for i in range(len(files)):
            filename=pathIn + files[i]
            #reading each files
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width,height)
            
            #inserting the frames into an image array
            frame_array.append(img)

        out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
        for i in range(len(frame_array)):
            # writing to a image array
            out.write(frame_array[i])
        out.release()
            

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
