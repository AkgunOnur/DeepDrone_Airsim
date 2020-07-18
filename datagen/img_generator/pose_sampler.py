import cv2
import numpy as np

import os
import sys

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
from numpy import zeros
from sklearn.externals.joblib import dump, load
from sklearn.preprocessing import StandardScaler
from quadrotor import *
from mytraj import MyTraj
from geom_utils import QuadPose

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
        self.quadrotor_freq = int(1. / self.dtau)
        self.method = "MAX"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = "cpu"
        self.model = Net()
        self.model = self.model.to(self.device)
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.model.eval()
        
        with open('dataset.pkl', 'r') as f:  # Python 3: open(..., 'wb')
            X_train, X_val, X_test, y_train, y_val, y_test = pickle.load(f)

        scaler = StandardScaler()
        scaler.fit(X_train)
        self.scaler = scaler

        #---- Model import ---------------------------------
        self.device = torch.device("cpu")

        input_size = 4
        output_size = 4
        lstmR_hidden_size = 16
        lstmR_num_layers = 1

        # Dronet
        self.Dronet =  Dronet.ResNet(Dronet.BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(self.device)
        print("Dronet Model:", self.Dronet)
        self.Dronet.load_state_dict(torch.load('/home/merkez/Downloads/kamil_airsim/weights/Dronet_yeni.pth'))   
        self.Dronet.eval()

        # LstmR
        self.lstmR = lstmf.LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
        self.lstmR.to(self.device)
        print("lstmR Model:", self.lstmR)
        self.lstmR.load_state_dict(torch.load('/home/merkez/Downloads/kamil_airsim/weights/16_hidden_lstm_R_PG.pth'))   
        self.lstmR.eval() 
        
        # Transformation
        self.transformation = transforms.Compose([
                transforms.Resize([200, 200]),
                #transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.ToTensor()]
        )

        self.drone_init = Pose(Vector3r(0.,0.,-0.1), Quaternionr(0., 0., -0.70710678, 0.70710678))
        self.gate = [Pose(Vector3r(0.,-5.,-2.), Quaternionr(0., 0., 0., 1.)),
                     Pose(Vector3r(2.,-8.,-2.5), Quaternionr(0., 0., 0.25881905, 0.96592583)),
                     Pose(Vector3r(4.,-10.,-3.), Quaternionr(0., 0., 0.38268343, 0.92387953)),
                     Pose(Vector3r(6.,-12.,-3.5), Quaternionr(0., 0., 0.5, 0.8660254)),
                     Pose(Vector3r(8.,-14.,-4), Quaternionr(0., 0., 0.70710678, 0.70710678))]
        #-----------------------------------------------------------------------             
    def polarTranslation(self,r, theta, psi):
        # follow math convention for polar coordinates
        # r: radius
        # theta: azimuth (horizontal)
        # psi: vertical
        x = r * np.cos(theta) * np.sin(psi)
        y = r * np.sin(theta) * np.sin(psi)
        z = r * np.cos(psi)
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
        gate_yaw = self.quad.state[5] + yaw_diff # yaw_gate = drone_yaw + yaw_diff
        yawf = yaw_diff - 0*np.pi/2
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
        pose_prediction = np.zeros((9999,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)
        path = '/home/merkez/Downloads/kamil_airsim/images'
        # create and set pose for the quad
        #p_o_b, phi_base = racing_utils.geom_utils.randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)
        self.client.simSetVehiclePose(self.drone_init, True)
        # create and set gate pose relative to the quad
        #p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV, correction)
        #self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        if self.with_gate:
            for i, gate in enumerate(self.gate):
                print ("gate: ", gate)
                gate_name = "gate_" + str(i)
                self.tgt_name = self.client.simSpawnObject(gate_name, "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
                self.client.simSetObjectPose(self.tgt_name, gate, True)
            # self.client.plot_tf([p_o_g], duration=20.0)
        # request quad img from AirSim
        time.sleep(0.001)
        
        r = R.from_quat([self.drone_init.orientation.x_val, self.drone_init.orientation.y_val, self.drone_init.orientation.z_val, self.drone_init.orientation.w_val])
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)

        state0 = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val,
                  roll, pitch, yaw, 0., 0., 0., 0., 0., 0.,]
        self.quad = Quadrotor(state0)
        
        while True: 
            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]

            # save all the necessary information to file
            self.writeImgToFile(image_response)
            image = image = Image.open(os.path.join(path,str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'))
            image = self.transformation(image)
                
            # Determine Gat location with Neural Networks
            with torch.no_grad():

                pose_gate_body = self.Dronet(image)
                print(pose_gate_body)
                r,theta,psi,phi = np.asarray(pose_gate_body[0])
                print(self.curr_idx)

                q1,q2,q3,q4 = R.from_euler('zyx',[self.quad.state[5], self.quad.state[4], self.quad.state[3]], degrees=False).as_quat()

                # print("Initial value of Gate:", self.gate.position.x_val, self.gate.position.y_val, self.gate.position.z_val)
                # for_estimation = Pose(Vector3r(self.quad.state[0], self.quad.state[1],self.quad.state[2]),
                #                              Quaternionr(q1,q2,q3,q4))

                #print("Drone States:", for_estimation)
                # estimation = self.debugGatePoses(for_estimation , r, theta, psi)
                # print("prediction of gate:", estimation.x_val, estimation.y_val, estimation.z_val)
                time.sleep(0.001)
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

                    if abs(pose_gate_body[0][0]) < 0.5:
                        print ("Drone final position, x= {0:.3}, y= {1:.3}, z= {2:.3}".format(self.quad.state[0], self.quad.state[1], self.quad.state[2]))
                        break

                    # Trajectory generate
                    ref_traj = self.get_trajectory(pose_gate_body, ground_truth = False) # Self olarak trajectory yollacayacak, quad_sim 'in icine
                    waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
                    N_length = np.minimum(len(ref_traj), self.quadrotor_freq)
                    prediction_std = prediction_std.ravel()

                    # Call for Controller
                    for i in range(N_length):  #Kontrolcu frekansi kadar itera edecek
                        prev_i = np.maximum(0, i-1)
                        current_traj = ref_traj[i]
                        prev_traj = ref_traj[prev_i]
                        self.quad.simulate(self.Tf, self.dtau, i, current_traj, prev_traj, prediction_std, scaler=self.scaler, model=self.model, device=self.device, method=self.method)
                        quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
                        
                        
                        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
                        #time.sleep(self.dtau)

            self.curr_idx += 1

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
            cv2.imwrite(os.path.join(self.base_path, 'images', str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)  # write to png
        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)
