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
from geom_utils import QuadPose
from trajectory import Trajectory

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
MP_methods = {"pos_way_timed":1, "pos_waypoint_interp":2, "min_vel":3, "min_acc":4, "min_jerk":5, "min_snap":6,
              "min_acc_stop":7, "min_jerk_stop":8, "min_snap_stop":9, "min_jerk_full_stop":10, "min_snap_full_stop":11,
              "pos_waypoint_arrived":12}


class PoseSampler:
    def __init__(self, num_samples, dataset_path, with_gate=True):
        self.num_samples = num_samples
        self.base_path = dataset_path
        self.csv_path = os.path.join(self.base_path, 'gate_training_data.csv')
        self.curr_idx = 0
        self.with_gate = with_gate
        self.client = airsim.MultirotorClient()

        self.client.confirmConnection()
        self.client.simLoadLevel('Soccer_Field_Easy')
        time.sleep(4)
        self.client = airsim.MultirotorClient()
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
        self.quadrotor_freq = int(1. / self.dtau)
        self.method = "MAX"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = Net()
        self.model = self.model.to(self.device)
        # self.model.load_state_dict(torch.load('best_model.pt'))
        # self.model.eval()
        self.state0 = [0, 0, 0, 0, 0, 0, 0., 0., 0., 0., 0., 0.]


        #----- Motion planning parameters -----------------------------
        self.MP_cost = {"pos_way_timed":1e6, "pos_waypoint_interp":1e6, "min_vel":1e6, "min_acc":1e6, "min_jerk":1e6, "min_snap":1e6,
                        "min_acc_stop":1e6, "min_jerk_stop":1e6, "min_snap_stop":1e6, "min_jerk_full_stop":1e6, "min_snap_full_stop":1e6,
                        "pos_waypoint_arrived":1e6}
        self.MP_states = {"pos_way_timed":[], "pos_waypoint_interp":[], "min_vel":[], "min_acc":[], "min_jerk":[], "min_snap":[],
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
        
        # with open('dataset.pkl', 'r') as f:  # Python 3: open(..., 'wb')
        #     X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

        # scaler = StandardScaler()
        # scaler.fit(X_train)
        # self.scaler = scaler

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
        self.Dronet.load_state_dict(torch.load('/home/merkez/Downloads/kamil_airsim/weights/Dronet_yeni.pth'))   
        self.Dronet.eval()

        # LstmR
        self.lstmR = lstmf.LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
        self.lstmR.to(self.device)
        #print("lstmR Model:", self.lstmR)
        self.lstmR.load_state_dict(torch.load('/home/merkez/Downloads/kamil_airsim/weights/16_hidden_lstm_R_PG.pth'))   
        self.lstmR.eval() 
        
        # Transformation
        self.transformation = transforms.Compose([
                transforms.Resize([200, 200]),
                #transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.ToTensor()]
        )

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
    def polarTranslation(self,r, theta, psi):
        # follow math convention for polar coordinates
        # r: radius
        # theta: azimuth (horizontal)
        # psi: vertical
        x = r * np.cos(theta) * np.sin(psi)
        y = r * np.sin(theta) * np.sin(psi)
        z = r * np.cos(theta)
        return Vector3r(x, y, z)

    def convert_t_body_2_world(self,t_body, q_o_b):
        rotation = Rotation.from_quat([q_o_b.x_val, q_o_b.y_val, q_o_b.z_val, q_o_b.w_val])
        t_body_np = [t_body.x_val, t_body.y_val, t_body.z_val]
        t_world_np = rotation.apply(t_body_np)
        t_world = Vector3r(t_world_np[0], t_world_np[1], t_world_np[2])
        return t_world

    def debugGatePoses(self,p_o_b, r, theta, psi):
        # get relative vector in the base frame coordinates
        t_b_g_body = self.polarTranslation(r, theta, psi)
        # transform relative vector from base frame to the world frame
        t_b_g = self.convert_t_body_2_world(t_b_g_body, p_o_b.orientation)
        # get the gate coord in world coordinates from origin
        t_o_g = p_o_b.position + t_b_g
        # check if gate is at least half outside the ground
        # create rotation of gate
        """phi_quad_ref = np.arctan2(p_o_b.position.y_val, p_o_b.position.x_val)
        phi_rel = np.pi/2
        phi_gate = phi_quad_ref + phi_rel
        rot_gate = Rotation.from_euler('ZYX', [phi_gate, 0, 0])
        q = rot_gate.as_quat()
        p_o_g = Pose(t_o_g, Quaternionr(q[0], q[1], q[2], q[3]))"""
        return t_o_g

    
    def get_trajectory(self, waypoint_body, ground_truth=False):
        #We have waypoint_body_noisy values, which are 
        #r_noisy, phi_noisy, theta_noisy, psi_noisy
        #We need to convert them into the cartesian form in world frame.
        if ground_truth:
            waypoint_world = [waypoint_body.position.x_val, waypoint_body.position.y_val, waypoint_body.position.z_val]
            yaw_diff = 0.
            self.Tf = 5.
        else:
            yaw_diff = waypoint_body[3][0]
            r = waypoint_body[0][0]
            self.Tf = r*0.75
            waypoint_world = spherical_to_cartesian(self.quad.state, waypoint_body)
        #print ("spherical_to_cartesian, x_gate= {0:.3}, y_gate= {1:.3}, z_gate= {2:.3}".format(waypoint_world[0], waypoint_world[1], waypoint_world[2]))


        pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
        vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
        acc0 = [0., 0., 0.]
        posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
        #gate_yaw = self.quad.state[5] + yaw_diff # yaw_gate = drone_yaw + yaw_diff
        yawf = np.pi +(self.quad.state[5]+yaw_diff)-np.pi/2# yaw_gate = drone_yaw + yaw_diff
        if yawf > 0:
            yawf = yawf % np.pi
        else:
            yawf = yawf % -np.pi
        #yawf = - np.pi/2
        velf = [0., 0., 0.]
        accf = [0., 0., 0.]

        traj = MyTraj(gravity = -9.81)
        trajectory = traj.givemetraj(pos0, vel0, acc0, posf, velf, accf, self.Tf)



        N = int(self.Tf/self.dtau)
        t = linspace(0,self.Tf,num = N)

        xd, yd, zd, psid = zeros(t.shape), zeros(t.shape), zeros(t.shape), zeros(t.shape)
        xd_dot, yd_dot, zd_dot, psid_dot = zeros(t.shape), zeros(t.shape), zeros(t.shape), zeros(t.shape)
        xd_ddot, yd_ddot, zd_ddot, psid_ddot = zeros(t.shape), zeros(t.shape), zeros(t.shape), zeros(t.shape)
        xd_dddot, yd_dddot, zd_dddot = zeros(t.shape), zeros(t.shape), zeros(t.shape)
        xd_ddddot, yd_ddddot, zd_ddddot = zeros(t.shape), zeros(t.shape), zeros(t.shape)

        i = 0
        ts = 0

        for i in range(N):
            pos_des, vel_des, acc_des = traj.givemepoint(trajectory, ts)

            xd[i], yd[i], zd[i] = pos_des[0], pos_des[1], pos_des[2]
            xd_dot[i], yd_dot[i], zd_dot[i] = vel_des[0], vel_des[1], vel_des[2]
            xd_ddot[i], yd_ddot[i], zd_ddot[i] = acc_des[0], acc_des[1], acc_des[2]
            psid[i] = yawf
            ts += self.dtau


        ref_traj = np.c_[xd, yd, zd, xd_dot, yd_dot, zd_dot, 
                         xd_ddot, yd_ddot, zd_ddot,
                         xd_dddot, yd_dddot, xd_ddddot, yd_ddddot,
                         psid, psid_dot, psid_ddot]
        
        
        return ref_traj


    def check_arrival(self, quad_pose, gate_index, eps=0.15):
        x, y, z = quad_pose[0], quad_pose[1], quad_pose[2]
        xd = self.track[gate_index].position.x_val
        yd = self.track[gate_index].position.y_val
        zd = self.track[gate_index].position.z_val

        if (abs(xd)-abs(x) <= eps) and (abs(yd)-abs(y) <= eps) and (abs(zd)-abs(z) <= 2*eps):
            return True

        return False


    def test_algorithm():
        newTraj = Trajectory(trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf, v_average=1.0)
        Waypoint_length = int(self.Tf/self.dtau)
        N = np.minimum(self.quadrotor_freq, Waypoint_length)
        t = linspace(0,self.Tf,num = Waypoint_length)
        prediction_std = prediction_std.ravel()


        # Call for Controller
        for i in range(N):  #Kontrolcu frekansi kadar itera edecek
            t_current = t[i]
            time_rate = float(t_current / self.Tf)

            pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)
            xd, yd, zd = pos_des[0], pos_des[1], pos_des[2]
            xd_dot, yd_dot, zd_dot = vel_des[0], vel_des[1], vel_des[2]
            xd_ddot, yd_ddot, zd_ddot = acc_des[0], acc_des[1], acc_des[2]

            xd_dddot = (xd_ddot - xd_ddot_pr) / self.dtau
            yd_dddot = (yd_ddot - yd_ddot_pr) / self.dtau
            xd_ddddot = (xd_dddot - xd_dddot_pr) / self.dtau
            yd_ddddot = (yd_dddot - yd_dddot_pr) / self.dtau

            psid = euler_des[2]

            psid_dot = (psid - psid_pr) / self.dtau
            psid_ddot = (psid_dot - psid_dot_pr) / self.dtau

            current_traj = [xd, yd, zd, xd_dot, yd_dot, zd_dot, xd_ddot, yd_ddot, zd_ddot,
                         xd_dddot, yd_dddot, xd_ddddot, yd_ddddot,
                         psid, psid_dot, psid_ddot]

            fail_check = self.quad.simulate(self.Tf, self.dtau, i, current_traj, prev_traj, prediction_std, scaler=self.scaler, model=self.model, device=self.device, method=self.method)
            quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
            self.client.simSetVehiclePose(QuadPose(quad_pose), True)

            print ("acc_x:{0:.2}-jerk_x:{1:.2}-snap_x:{2:.2}, acc_y:{3:.2}-jerk_y:{4:.2}-snap_y:{5:.2}, psid:{6:.2}-psid_dot:{7:.2}-psid_ddot:{8:.2}"
                .format(xd_ddot,xd_dddot,xd_ddddot, yd_ddot,yd_dddot,yd_ddddot, psid,psid_dot,psid_ddot))

            if fail_check:
                return 

            prev_traj = np.copy(current_traj)
            xd_ddot_pr = xd_ddot
            yd_ddot_pr = yd_ddot
            xd_dddot_pr = xd_dddot
            yd_dddot_pr = yd_dddot
            psid_pr = psid
            psid_dot_pr = psid_dot


    def collect_data(self, MP_list):
        
        
        path = '/home/merkez/Downloads/kamil_airsim/images'

        angle_lim = 10.0

        for gate_index in range(len(self.track)):
            phi_start = angle_lim*random.uniform(-1.0,1.0)*np.pi/180
            theta_start = angle_lim*random.uniform(-1.0,1.0)*np.pi/180
            gate_target = self.track[gate_index]
            gate_psi = Rotation.from_quat([gate_target.orientation.x_val, gate_target.orientation.y_val, gate_target.orientation.z_val, gate_target.orientation.w_val]).as_euler('ZYX',degrees=False)[0]
            psi_start = angle_lim*random.uniform(-1.0,1.0)*np.pi/180 + gate_psi - 2*np.pi/3  #drone kapi karsisinde olacak sekilde durmali
            time_or_speed = random.randint(0,1)


            for algorithm in MP_list:
                pose_prediction = np.zeros((1000,4),dtype=np.float32)
                prediction_std = np.zeros((4,1),dtype=np.float32)

                if gate_index == 0: #if drone is at initial point
                    quad_pose = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, -phi_start, -theta_start, psi_start]
                    self.state0 = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]
                else:
                    quad_pose = [self.track[gate_index-1].position.x_val, self.track[gate_index-1].position.y_val, self.track[gate_index-1].position.z_val, -phi_start, -theta_start, psi_start]
                    self.state0 = [self.track[gate_index-1].position.x_val, self.track[gate_index-1].position.y_val, self.track[gate_index-1].position.z_val, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]


                self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                self.quad = Quadrotor(self.state0)
                self.trajSelect[0] = MP_methods[algorithm]
                self.trajSelect[1] = 2
                self.trajSelect[2] = time_or_speed
                self.curr_idx = 0
                self.MP_states[algorithm].append(self.quad.state)

                print "\nMP Method: ", algorithm
                track_completed = False
                fail_check = False
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

                    # if self.check_arrival(quad_pose):
                    #     track_completed = True
                    #     print ("The track is completed!")
                    #     break

                    with torch.no_grad():   
                        # Determine Gat location with Neural Networks
                        pose_gate_body = self.Dronet(image)

                        if self.check_arrival(quad_pose, gate_index=gate_index):
                            track_completed = True
                            print "Drone has arrived to the {0}. Gate for {1}".format(gate_index,algorithm) 
                            break
                        #r,theta,psi,phi = np.asarray(pose_gate_body[0])
                        #q1,q2,q3,q4 = R.from_euler('zyx',[self.quad.state[5], self.quad.state[4], self.quad.state[3]], degrees=False).as_quat()

                        # print("Initial value of Gate:", self.gate.position.x_val, self.gate.position.y_val, self.gate.position.z_val)
                        # for_estimation = Pose(Vector3r(self.quad.state[0], self.quad.state[1],self.quad.state[2]),
                        #                              Quaternionr(q1,q2,q3,q4))
                        #print("Drone States:", for_estimation)
                        # estimation = self.debugGatePoses(for_estimation , r, theta, psi)
                        # print("prediction of gate:", estimation.x_val, estimation.y_val, estimation.z_val)
                        #time.sleep(0.001)
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

                            # Trajectory generate
                            #ef_traj = self.get_trajectory(pose_gate_body, ground_truth = False) # Self olarak trajectory yollacayacak, quad_sim 'in icine
                            self.Tf = random.uniform(0.5,1)*pose_gate_body[0][0] # T=r*0.5
                            waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
                            pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                            posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
                            yaw0 = self.quad.state[5]
                            yaw_diff = pose_gate_body[3][0]
                            yawf = np.pi +(self.quad.state[5]+yaw_diff)-np.pi/2
                            

                            if time_or_speed == 0:
                                newTraj = Trajectory(self.trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf)
                                print "Time based trajectory, T: ", newTraj.t_wps[1] 
                            else:
                                self.v_average = random.uniform(0.5,3.0)
                                newTraj = Trajectory(self.trajSelect, self.quad.state, 1.0, pos0, posf, yaw0, yawf, v_average=self.v_average)
                                print "Velocity based trajectory, V_avg: ", self.v_average, ", T: ", newTraj.t_wps[1]

                            Waypoint_length = int(newTraj.t_wps[1] / self.dtau)
                            t = linspace(0, newTraj.t_wps[1], num = Waypoint_length)
                            prediction_std = prediction_std.ravel()
                            
                            #print "T_total: ", newTraj.t_wps

                            # Call for Controller
                            for i in range(Waypoint_length): 
                                t_current = t[i]
                                time_rate = float(t_current / self.Tf)

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

                                fail_check = self.quad.collect_data(self.Tf, self.dtau, i, current_traj, prediction_std)

                                quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
                                self.MP_states[algorithm].append(self.quad.state)
                                self.client.simSetVehiclePose(QuadPose(quad_pose), True)

                                # print ("acc_x:{0:.2}-jerk_x:{1:.2}-snap_x:{2:.2}, acc_y:{3:.2}-jerk_y:{4:.2}-snap_y:{5:.2}, psid:{6:.2}-psid_dot:{7:.2}-psid_ddot:{8:.2}"
                                #     .format(xd_ddot,xd_dddot,xd_ddddot, yd_ddot,yd_dddot,yd_ddddot, psid,psid_dot,psid_ddot))

                                if fail_check:
                                    break 

                                if self.check_arrival(quad_pose, gate_index=gate_index):
                                    track_completed = True
                                    print "Drone has arrived to the {0}. Gate for {1}".format(gate_index,algorithm) 
                                    break


                                prev_traj = np.copy(current_traj)
                                self.xd_ddot_pr = xd_ddot
                                self.yd_ddot_pr = yd_ddot
                                self.xd_dddot_pr = xd_dddot
                                self.yd_dddot_pr = yd_dddot
                                self.psid_pr = psid
                                self.psid_dot_pr = psid_dot

                    self.curr_idx += 1

                self.MP_cost[algorithm] = self.quad.costValue
                print "For ", algorithm, " cost value: ",self.MP_cost[algorithm]

            min_cost_index = min(self.MP_cost.items(), key=lambda x: x[1])[0]
            print ">>>Best method for gate ", (gate_index + 1), " is ", min_cost_index
            # write_stats(flight_columns,
            #     [state[0], state[1], state[2], state[6], state[7], state[8], state[3], state[4], state[5], state[9], state[10], state[11],
            #     current_traj[0], current_traj[1], current_traj[2], current_traj[3], current_traj[4], current_traj[5], current_traj[6], current_traj[7], current_traj[8],
            #     state[0]-current_traj[0], state[1]-current_traj[1], state[2]-current_traj[2], time_rate, t_current, Tf, 
            #     prev_traj[0], prev_traj[1], prev_traj[2], prev_traj[3], prev_traj[4], prev_traj[5], prev_traj[6], prev_traj[7], prev_traj[8], 
            #     Upr_abs_sum, r_std, phi_std, theta_std, psi_std, min_cost_index], flight_filename)



    def visualize_drone(self, MP_list):
        for algorithm in MP_list:
            print "Drone flies by the algorithm, ", algorithm
            self.client.simSetVehiclePose(self.drone_init, True)
            state_list = self.MP_states[algorithm]
            for state in state_list:
                quad_pose = [state[0], state[1], state[2], -state[3], -state[4], state[5]]
                self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                time.sleep(0.001)




    def update(self):
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

        MP_list = ["min_vel", "min_acc", "min_jerk", "min_snap"] 
        if self.with_gate:
            for i, gate in enumerate(self.track):
                #print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
                self.client.simSetObjectPose(self.tgt_name, gate, True)
            # self.client.plot_tf([p_o_g], duration=20.0)
        # request quad img from AirSim
        time.sleep(0.001)
        
        r = R.from_quat([self.drone_init.orientation.x_val, self.drone_init.orientation.y_val, self.drone_init.orientation.z_val, self.drone_init.orientation.w_val])
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)

        self.state0 = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val,
                  roll, pitch, yaw, 0., 0., 0., 0., 0., 0.,]

        prev_traj = np.copy(self.state0)

        self.collect_data(MP_list)
        self.visualize_drone(MP_list)
        #self.get_video(MP_list[0])
        
        

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
