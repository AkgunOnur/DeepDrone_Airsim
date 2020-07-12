import cv2
import numpy as np

import os
import sys
from numpy import zeros

import airsimdroneracingvae as airsim
# print(os.path.abspath(airsim.__file__))
from airsimdroneracingvae.types import Pose, Vector3r, Quaternionr
import airsimdroneracingvae.types
import airsimdroneracingvae.utils
import time
import pickle

import torch
from torchvision import models, transforms
from torchvision.transforms.functional import normalize, resize, to_tensor
from scipy.spatial.transform import Rotation

# import utils
curr_dir = os.path.dirname(os.path.abspath(__file__))
import_path = os.path.join(curr_dir, '..', '..')
sys.path.insert(0, import_path)
import racing_utils

from models.Dronet import ResNet,BasicBlock
from models.Lstmf import LstmNet
from quadrotor import *

METHOD_LIST = ["MAX", "MIX", "DICE"]#, "DT", "FOREST", "Backstepping_1", "Backstepping_2", "Backstepping_3", "Backstepping_4"]


GATE_YAW_RANGE = [-np.pi, np.pi]  # world theta gate
# GATE_YAW_RANGE = [-1, 1]  # world theta gate -- this range is btw -pi and +pi
UAV_X_RANGE = [-30, 30] # world x quad
UAV_Y_RANGE = [-30, 30] # world y quad
UAV_Z_RANGE = [-2, -3] # world z quad

UAV_YAW_RANGE = [-np.pi, np.pi]  #[-eps, eps] [-np.pi/4, np.pi/4]
eps = np.pi/10.0  # 18 degrees
UAV_PITCH_RANGE = [-eps, eps]  #[-np.pi/4, np.pi/4]
UAV_ROLL_RANGE = [-eps, eps]  #[-np.pi/4, np.pi/4]

R_RANGE = [0.1, 20]  # in meters
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
#         with open('file.pickle', 'rb') as f:
#             self.state = pickle.load(f)
#         #print(state[0])
#         print(len(self.state))
#         for i in range(len(self.state)):

#             for j in [1,2,3]:
#                 self.state[i][j] -=20
        

        #Drone parameters
        self.quad = None
        self.dtau = 1e-3
        self.quadrotor_freq = int(1. / self.dtau)
        self.method = "MAX"
        self.device torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net()
        self.model = model.to(device)
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.model.eval()


        #---- Model İmport
        
        input_size = 4
        output_size = 4
        lstmR_hidden_size = 16
        lstmR_num_layers = 1

        # Dronet
        self.Dronet =  ResNet(BasicBlock, [1,1,1,1], num_classes = 4)
        self.Dronet.to(device)
        print("Dronet Model:", self.Dronet)
        self.Dronet.load_state_dict(torch.load('/home/kca/Desktop/all_files/Airsim_Deepdrone/weights/Dronet.pth'))   
        self.Dronet.eval()

        # LstmR
        self.lstmR = LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
        self.lstmR.to(device)
        print("lstmR Model:", self.lstmR)
        self.lstmR.load_state_dict(torch.load('/home/kca/Desktop/all_files/Airsim_Deepdrone/weights/16_hidden_lstm_R_PG.pth'))   
        self.lstmR.eval() 
        
        # Transformation
        self.transformation = transforms.Compose([
                transforms.Resize([200, 200]),
                #transforms.ColorJitter(hue=.05, saturation=.05),
                transforms.ToTensor()]
        )
       
        
    def get_trajectory(self, waypoint_body_noisy, yaw_diff):
        #We have waypoint_body_noisy values, which are 
        #r_noisy, phi_noisy, theta_noisy, psi_noisy
        #We need to convert them into the cartesian form in world frame.
        r = waypoint_body_noisy[0]
        Tf = r*0.5
        waypoint_world = spherical_to_cartesian(quad.state, waypoint_body_noisy)


        pos0 = [quad.state[0], quad.state[1], quad.state[2]]
        vel0 = [quad.state[6], quad.state[7], quad.state[8]]
        acc0 = [0., 0., 0.]
        posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
        gate_yaw = quad.state[5] + yaw_diff # yaw_gate = drone_yaw + yaw_diff
        yawf = gate_yaw - np.pi/2
        velf = [0., 0., 0.]
        accf = [0., 0., 0.]

        traj = MyTraj(gravity = -9.81)
        trajectory = traj.givemetraj(pos0, vel0, acc0, posf, velf, accf, Tf)

        N = int(Tf/dtau)
        t = linspace(0,Tf,num = N)

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
            ts += dtau


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
        
        #for i in range(len(self.state)):
        # Gate pose from txt file. It is temporary solution
        pose_quad, phi_base = racing_utils.geom_utils.randomQuadPose(UAV_X_RANGE, UAV_Y_RANGE, UAV_Z_RANGE, UAV_YAW_RANGE, UAV_PITCH_RANGE, UAV_ROLL_RANGE)

        # Determine pose for the gate, It isn't random.
        gate_p_o_b = racing_utils.geom_utils.randomGatePose(pose_quad, phi_base, r_range, cam_fov, correction)

        # Generate the Gate in given pose
        self.client.simSetObjectPose(self.tgt_name, gate_p_o_b, True)

        # Generate quadrotor according to given pose
        self.client.simSetVehiclePose(pose_quad, True)
        time.sleep(0.001)
        
        r = R.from_quat([pose_quad.q0, pose_quad.q1, pose_quad.q2, pose_quad.q3])
        yaw, pitch, roll = r.as_euler('zyx', degrees=False)
        yaw
        state0 = [pose_quad.x, pose_quad.y, pose_quad.z, roll, pitch, yaw, 0., 0., 0., 0., 0., 0.,]
        self.quad = Quadrotor(state0)
        
        while True:   
          
            # Take image to calculate
            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            
             # Transfrom the Image
            image = self.transformation(image_response)

            # Determine Gate location with Neural Networks
            pose_gate_body = Dronet(image_response)
            pose_gate_cov = Lstm_R(pose_gate[:-12])
            
            # Gate ground truth values will be implemented
            if abs(pose_gate[0]) < 0.5:
                break

            # Trajectory generate
            ref_traj = get_trajectory(self, pose_gate_body, yaw_diff) # Self olarak trajectory yollacayacak, quad_sim 'in içine

            # Call for Controller
            for i in range(self.quadrotor_freq):  #Kontrolcü frekansı kadar itera edecek
                state, costValue = quad.simulate(Tf, dtau, N, current_traj, prev_traj, std_list, scaler=self.scaler, model=self.model, device=self.device, method=self.method)
                Rot = Rotation.from_euler('ZYX', [state[5], state[4], state[3]])  # capital letters denote intrinsic rotation (lower case would be extrinsic)
                q = Rot.as_quat()
                pose_quad = QuadPose([state[0], state[1], state[2], q[0], q[0], q[2], q[3]])
                self.client.simSetVehiclePose(pose_quad, True)
                time.sleep(self.dtau)

           
    
            # Generate quadrotor according to given pose
            
            self.curr_idx += 1
            
                
        #_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV, correction)
        """for i, p_o_b in enumerate(pose_file):


            self.client.simSetVehiclePose(p_o_b, True)
            image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
            self.writeImgToFile(image_response)
            self.writePosToFile(r, theta, psi, phi_rel)
        # create and set gate pose relative to the quad
        #p_o_g, r, theta, psi, phi_rel = racing_utils.geom_utils.randomGatePose(p_o_b, phi_base, R_RANGE, CAM_FOV, correction)
        # self.client.simSetObjectPose(self.tgt_name, p_o_g_new, True)
        if self.with_gate:
            self.client.simSetObjectPose(self.tgt_name, p_o_g, True)
            # self.client.plot_tf([p_o_g], duration=20.0)
        # request quad img from AirSim
        #image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
        # save all the necessary information to file
        #self.writeImgToFile(image_response)
        #self.writePosToFile(r, theta, psi, phi_rel)
        """

    def configureEnvironment(self):
        for gate_object in self.client.simListSceneObjects(".*[Gg]ate.*"):
            self.client.simDestroyObject(gate_object)
            time.sleep(0.05)
        if self.with_gate:
            self.tgt_name = self.client.simSpawnObject("gate", "RedGate16x16", Pose(position_val=Vector3r(0,0,15)), 0.75)
            # self.tgt_name = self.client.simSpawnObject("gate", "CheckeredGate16x16", Pose(position_val=Vector3r(0,0,15)))
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
            cv2.imwrite(os.path.join(self.base_path, 'images', str(self.curr_idx/Quadrotor_Hz).zfill(len(str(self.num_samples))) + '.png'), img_rgb)  # write to png
        else:
            print('ERROR IN IMAGE SIZE -- NOT SUPPOSED TO HAPPEN')

    # writes your data to file
    def writePosToFile(self, r, theta, psi, phi_rel):
        data_string = '{0} {1} {2} {3}\n'.format(r, theta, psi, phi_rel)
        self.file.write(data_string)
