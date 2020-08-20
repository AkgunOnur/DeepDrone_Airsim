import numpy as np
from numpy import array, zeros, diag, ones, sin, cos, tan, linspace, dot, pi
from numpy.random import uniform
import time
from scipy import integrate
from numpy import isnan, pi, isinf
from numpy.random import normal
import pandas as pd
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
import math
from scipy.spatial.transform import Rotation as R

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.drop_layer = nn.Dropout(p=0.0)
        self.fc1 = nn.Linear(41, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 96)
        self.fc4 = nn.Linear(96, 64)
        self.fc5 = nn.Linear(64, 48)
        self.fc6 = nn.Linear(48, 32)
        self.fc7 = nn.Linear(32, 24)
        self.fc8 = nn.Linear(24, 16)
        self.fc9 = nn.Linear(16, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc2(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc3(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc4(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc5(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc6(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc7(x))
        x = self.drop_layer(x)
        x = F.relu(self.fc8(x))
        x = self.drop_layer(x)
        x = self.fc9(x)
        return x

def cartesian_to_spherical(state, waypoint_world):
    x_gate, y_gate, z_gate, yaw_gate = waypoint_world
    x_diff = -state[0] + x_gate
    y_diff = -state[1] + y_gate
    z_diff = -state[2] + z_gate

    waypoint_body = world_to_body(state, np.array([x_diff, y_diff, z_diff]))

    r = math.sqrt(waypoint_body[0]**2 + waypoint_body[1]**2 + waypoint_body[2]**2)
    phi = math.atan2(waypoint_body[1], waypoint_body[0])
    theta = math.acos(waypoint_body[2] / r)                          
    return r, phi, theta
    
def spherical_to_cartesian(state, waypoint_body):
    r, phi, theta, _ = waypoint_body.ravel()
    x = r*sin(theta)*cos(phi)
    y = r*sin(theta)*sin(phi)
    z = r*cos(theta)
    
    waypoint_world = body_to_world(state, np.array([x,y,z]))
    x_gate, y_gate, z_gate = state[0]+waypoint_world[0], state[1]+waypoint_world[1], state[2]+waypoint_world[2]
    
    return x_gate, y_gate, z_gate



def world_to_body(state, waypoint_world):
    psi, theta, phi = state[5], state[4], state[3]
    rot = R.from_euler('zyx', [[psi, theta, phi]], degrees=False).as_dcm()
    # R = np.array([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
    #               [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
    #               [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    waypoint_body = np.dot(rot.T, waypoint_world.reshape(-1,1))
    
    return waypoint_body.ravel()


def body_to_world(state, waypoint_body):
    psi, theta, phi = state[5], state[4], state[3]
    rot = R.from_euler('zyx', [[psi, theta, phi]], degrees=False).as_dcm()
    # R = np.array([[cos(theta)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
    #               [cos(theta)*sin(psi), sin(phi)*sin(theta)*sin(psi)+cos(phi)*cos(psi), cos(phi)*sin(theta)*sin(psi)-sin(phi)*cos(psi)],
    #               [-sin(theta), sin(phi)*cos(theta), cos(phi)*cos(theta)]])
    waypoint_world = np.dot(rot, waypoint_body.reshape(-1,1))
    
    return waypoint_world.ravel()


class Quadrotor:
    
    def __init__(self, state0, coeff_pos=1.0, coeff_angle = 0.25, coeff_control = 0.0):
        
        self.state = state0
        self.U = [1, 0., 0., 0.]
        self.costValue = 0.
        self.coeff_pos = coeff_pos
        self.coeff_angle = coeff_angle
        self.coeff_control = coeff_control
        self.Controllers = ["Backstepping_1", "Backstepping_2", "Backstepping_3", "Backstepping_4"]
        self.count_dict = {"Backstepping_1": 0, "Backstepping_2": 0, "Backstepping_3": 0, "Backstepping_4": 0}

    def predict(self, X, model, device, method="MAX"):
        softmax = nn.Softmax(dim=1)
        inputs = torch.from_numpy(X).to(device)
        outputs = model(inputs.float())
     
        if method=="MAX":
            _, pred = torch.max(outputs, 1)
            return pred[0]
        elif method=="DICE":
            probs = softmax(outputs).cpu().detach().numpy()[0]
            pred_index = np.random.choice([0, 1, 2, 3], 1, p=probs)[0]
            return pred_index
        elif method=="MIX":
            probs = softmax(outputs).cpu().detach().numpy()[0]     
            return probs
        
    
    def model_parameters(self):
        g = 9.81
        m = 1.52
        Ixx, Iyy, Izz = 0.0347563, 0.0458929, 0.0977
        I1 = (Iyy - Izz) / Ixx
        I2 = (Izz - Ixx) / Iyy
        I3 = (Ixx - Iyy) / Izz
        Jr = 0.0001
        l = 0.09
        b = 8.54858e-6
        d = 1.6e-2

        return g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d


    def model_dynamics(self, t, state):
        g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d = self.model_parameters()
        #states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]

        x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot = state
        U1, U2, U3, U4 = self.U

        # if (d*U1 - 2*d*U3 - b*U4) < 0:
        #     omega1 = - np.sqrt(- d*U1 + 2*d*U3 + b*U4) / (2*np.sqrt(b*d))
        # else:
        #     omega1 = - np.sqrt(d*U1 - 2*d*U3 - b*U4) / (2*np.sqrt(b*d))

        # if (d*U1 - 2*d*U2 + b*U4) < 0:
        #     omega2 = -np.sqrt(-d*U1 + 2*d*U2 - b*U4) / (2*np.sqrt(b*d))
        # else:
        #     omega2 = -np.sqrt(d*U1 - 2*d*U2 + b*U4) / (2*np.sqrt(b*d))


        # if (d*U1 + 2*d*U3 - b*U4) < 0:
        #     omega3 = -np.sqrt(-d*U1 - 2*d*U3 + b*U4) / (2*np.sqrt(b*d))
        # else:
        #     omega3 = -np.sqrt(d*U1 + 2*d*U3 - b*U4) / (2*np.sqrt(b*d))

        # if (d*U1 + 2*d*U2 + b*U4) < 0:
        #     omega4 = -np.sqrt(-d*U1 - 2*d*U2 - b*U4) / (2*np.sqrt(b*d))
        # else:
        #     omega4 = -np.sqrt(d*U1 + 2*d*U2 + b*U4) / (2*np.sqrt(b*d))


        omega = 0.

        state_dot = np.zeros(12)
        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = z_dot
        state_dot[3] = phi_dot
        state_dot[4] = theta_dot
        state_dot[5] = psi_dot

        state_dot[6] = (cos(phi)*sin(theta)*cos(psi) + sin(phi)*sin(psi))*U1/m
        state_dot[7] = (cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi))*U1/m
        state_dot[8] = -g + cos(phi)*cos(theta)*U1/m    
        state_dot[9] = theta_dot*psi_dot*I1 - Jr / Ixx * theta_dot * omega  + l/Ixx*U2
        state_dot[10] = phi_dot*psi_dot*I2 + Jr / Iyy * phi_dot * omega + l/Iyy*U3
        state_dot[11] = phi_dot*theta_dot*I3 + 1/Izz*U4

        return state_dot


    

    def backstepping(self, A1, A2, A3, A4, A5, A6, U_list, ref_traj):
        g, m, Ixx, Iyy, Izz, I1, I2, I3, Jr, l, b, d = self.model_parameters()

        U1, U2, U3, U4 = U_list

        #self.state: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]
        x, y, z = self.state[0], self.state[1], self.state[2]
        phi, theta, psi = self.state[3], self.state[4], self.state[5]
        x_dot, y_dot, z_dot = self.state[6], self.state[7], self.state[8]
        phi_dot, theta_dot, psi_dot = self.state[9], self.state[10], self.state[11]

    #     ref_traj = [xd[i], yd[i], zd[i], xd_dot[i], yd_dot[i], zd_dot[i], 
    #                 xd_ddot[i], yd_ddot[i], zd_ddot[i], xd_dddot[i], yd_dddot[i],
    #                 xd_ddddot[i], yd_ddddot[i], psid[i], psid_dot[i], psid_ddot[i]]


        xd, yd, zd = ref_traj[0], ref_traj[1], ref_traj[2], 
        xd_dot, yd_dot, zd_dot = ref_traj[3], ref_traj[4], ref_traj[5]
        xd_ddot, yd_ddot, zd_ddot = ref_traj[6], ref_traj[7], ref_traj[8]
        xd_dddot, yd_dddot = ref_traj[9], ref_traj[10]
        xd_ddddot, yd_ddddot = ref_traj[11], ref_traj[12]
        psid, psid_dot, psid_ddot = ref_traj[13], ref_traj[14], ref_traj[15]

        x1, x2, x3 = array([[x], [y]]), array([[x_dot], [y_dot]]), array([[phi], [theta]])
        x4, x5, x6 = array([[phi_dot], [theta_dot]]), array([[psi], [z]]), array([[psi_dot], [z_dot]])

        g0 = array([[np.cos(psi), np.sin(psi)],  [np.sin(psi), -np.cos(psi)]])
        g0_inv = array([[np.cos(psi), np.sin(psi)],  [np.sin(psi), -np.cos(psi)]])

        g1 = array([[theta_dot*psi_dot*I1],  [phi_dot*psi_dot*I2]])
        g2 = array([[phi_dot*theta_dot*I3],  [-g]])

        l0 = array([[np.cos(phi)*np.sin(theta)],  [np.sin(phi)]])*U1/m 
        dl0_dx3 = array([[-np.sin(phi)*np.sin(theta), np.cos(phi)*np.cos(theta)],  [np.cos(phi), 0]])*U1/m 
        dl0_dx3_inv = array([[0, 1/np.cos(phi)],  [1/np.cos(theta)*1/np.cos(phi), 1/np.cos(phi)*np.tan(theta)*np.tan(phi)]])*m/U1 
        dl0_dx3_inv_dot = array([[0, 1/np.cos(phi)*np.tan(phi)*phi_dot], 
                                 [1/np.cos(theta)*1/np.cos(phi)*(np.tan(theta)*theta_dot + np.tan(phi)*phi_dot), 1/np.cos(phi)*((1/np.cos(theta))**2*np.tan(phi)*theta_dot + (-1+2*(1/np.cos(phi))**2)*np.tan(theta)*phi_dot)]])*m/U1 

    #     Omega_square = Omega_coef_inv * abs([U1/b  U2/b  U3/b  U4/d]) 
    #     Omega_param = sqrt(Omega_square) 
    #     omega = Omega_param(2) + Omega_param[3] - Omega_param(1) - Omega_param(3) 

    #     h1 = [-Jr/Ixx*theta_dot*omega  Jr/Iyy*phi_dot*omega] 
        h1 = 0 
        k1 = diag([l/Ixx, l/Iyy]) 
        k1_inv = diag([Ixx/l, Iyy/l]) 
        k2 = diag([1/Izz, np.cos(phi)*np.cos(theta)/m]) 
        k2_inv = diag([Izz, m/(np.cos(phi)*np.cos(theta))]) 

        x1d = array([[xd], [yd]])  
        x1d_dot = array([[xd_dot], [yd_dot]]) 
        x1d_ddot = array([[xd_ddot], [yd_ddot]]) 
        x1d_dddot = array([[xd_dddot], [yd_dddot]]) 
        x1d_ddddot = array([[xd_ddddot], [yd_ddddot]]) 

        x5d = array([[psid], [zd]])
        x5d_dot = array([[psid_dot], [zd_dot]]) 
        x5d_ddot = array([[psid_ddot], [zd_ddot]]) 

        z1 = x1d - x1 
        v1 = x1d_dot + dot(A1,z1) 
        z2 = v1 - x2 
        z1_dot = -dot(A1,z1) + z2 
        v1_dot = x1d_ddot + dot(A1,z1_dot) 
        v2 = dot(g0_inv, z1 + v1_dot + dot(A2,z2)) 
        z3 = v2 - l0  
        z2_dot = -z1 - dot(A2,z2) + dot(g0,z3) 
        z1_ddot = -dot(A1,z1_dot) + z2_dot 
        v1_ddot = x1d_dddot + dot(A1, z1_ddot) 
        v2_dot = dot(g0_inv, z1_dot + v1_ddot + dot(A2,z2_dot)) 
        v3 = dot(dl0_dx3_inv, dot(g0.T,z2) + v2_dot + dot(A3, z3)) 
        z4 = v3 - x4 
        z3_dot = -dot(g0.T, z2) - dot(A3,z3) + dot(dl0_dx3, z4) 
        z2_ddot = - z1_dot - dot(A2, z2_dot) + dot(g0, z3_dot) 
        z1_dddot = -dot(A1, z1_ddot) + z2_ddot 
        v1_dddot = x1d_ddddot + dot(A1, z1_dddot) 
        v2_ddot = dot(g0_inv, z1_ddot + v1_dddot + dot(A2, z2_ddot)) 
        v3_dot = dot(dl0_dx3_inv, dot(g0.T, z2_dot) + v2_ddot + dot(A3, z3_dot)) + dot(dl0_dx3_inv_dot, dot(g0.T, z2) + v2_dot + dot(A3, z3))
        l1 = dot(k1_inv, dot(dl0_dx3.T, z3) + v3_dot - g1 - h1 + dot(A4, z4)).ravel()

        z5 = x5d - x5 
        v5 = x5d_dot + dot(A5, z5) 
        z6 = v5 - x6 
        z5_dot = - dot(A5, z5) + z6 
        v5_dot = x5d_ddot + dot(A5, z5_dot) 
        l2 = dot(k2_inv, z5 + v5_dot - g2 + dot(A6, z6)).ravel()

        U1, U2, U3, U4 = l2[1], l1[0], l1[1], l2[0]

        U1 = np.clip(U1, 1.0, 1e2)
        U2 = np.clip(U2, -1e2, 1e2)
        U3 = np.clip(U3, -1e2, 1e2)
        U4 = np.clip(U4, -1e2, 1e2)

        U = np.array([U1, U2, U3, U4])

        return U

    def get_control_input(self, cont, current_traj):
        U0 = self.U 
        if (cont == self.Controllers[0]): #Backstepping_1
            A1, A2, A3 = 15*diag([1,1]), 10*diag([1,1]), 15*diag([1,1]) 
            A4, A5, A6 = 10*diag([1,1]), 15*diag([1,1]), 10*diag([1,1]) 
            U = self.backstepping(A1, A2, A3, A4, A5, A6, U0, current_traj) 
        elif (cont == self.Controllers[1]): #Backstepping_2
            A1, A2, A3 = 10*diag([1,1]), 5*diag([1,1]), 10*diag([1,1]) 
            A4, A5, A6 = 5*diag([1,1]), 10*diag([1,1]), 5*diag([1,1])
            U = self.backstepping(A1, A2, A3, A4, A5, A6, U0, current_traj) 
        elif (cont == self.Controllers[2]): #Backstepping_3
            A1, A2, A3 = 5*diag([1,1]), 3*diag([1,1]), 10*diag([1,1]) 
            A4, A5, A6 = 7*diag([1,1]), 1*diag([1,1]), 1*diag([1,1])  
            U = self.backstepping(A1, A2, A3, A4, A5, A6, U0, current_traj)
        elif (cont == self.Controllers[3]): #Backstepping_4
            A1, A2, A3 = 2*diag([1,1]), 5*diag([1,1]), 2*diag([1,1]) 
            A4, A5, A6 = 5*diag([1,1]), 2*diag([1,1]), 5*diag([1,1]) 
            U = self.backstepping(A1, A2, A3, A4, A5, A6, U0, current_traj)
        return U

    def bring_quad_to_des(self, des_traj, dtau, cont="Backstepping_4"):
        #des_traj = [x,y,z,yaw]
        x,y,z,yaw = des_traj

        ref_traj = [x, y, z, 0, 0, 0, 
                    0, 0, 0, 0, 0,
                    0, 0, yaw, 0, 0]
        self.U = self.get_control_input(cont, ref_traj)
        sol = integrate.solve_ivp(fun=self.model_dynamics, t_span=(0, dtau), y0=self.state)
        self.state = sol.y[:,-1]



    
    def simulate(self, Tf, dtau, i, current_traj, prev_traj, std_list, scaler=None, model=None, device=None, method="MAX"):
    #     states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]
        r_std, phi_std, theta_std, psi_std = std_list
        N = 1 / dtau

        t_current = Tf / N * i

        fail_check = False

#         feature_names = ['x0', 'y0', 'z0', 'x_dot0','y_dot0','z_dot0', 'phi0','theta0','yaw0', 'phi_dot0','theta_dot0','yaw_dot0', 
#                  'xf', 'yf', 'zf', 'x_dotf','y_dotf','z_dotf','x_ddotf','y_ddotf','z_ddotf',
#                  'pos_diffx','pos_diffy','pos_diffz','time_rate','t', 'Tf', 
#                  'xp', 'yp', 'zp', 'x_dotp','y_dotp','z_dotp','x_ddotp','y_ddotp','z_ddotp', 'u_abs_p',
#                  'r_std', 'phi_std', 'psi_std']


        ## ADD NOISE ##
        self.state[6] = normal(self.state[6], 0*r_std / 3.0)
        self.state[7] = normal(self.state[7], 0*r_std / 3.0)
        self.state[8] = normal(self.state[8], 0*r_std / 3.0)
        self.state[9] = normal(self.state[9], 0*phi_std)
        self.state[10] = normal(self.state[10], 0*theta_std)
        self.state[11] = normal(self.state[11], 0*psi_std)

        # X_test = np.array([self.state[0], self.state[1], -self.state[2], self.state[6], self.state[7], self.state[8], self.state[3], self.state[4], self.state[5], self.state[9], self.state[10], self.state[11],
        #             current_traj[0], current_traj[1], -current_traj[2], current_traj[3], current_traj[4], current_traj[5], current_traj[6], current_traj[7], current_traj[8],
        #             self.state[0]-current_traj[0], self.state[1]-current_traj[1], -self.state[2]+current_traj[2], time_rate, t_current, Tf, 
        #             prev_traj[0], prev_traj[1], -prev_traj[2], prev_traj[3], prev_traj[4], prev_traj[5], prev_traj[6], prev_traj[7], prev_traj[8], 
        #             Upr_abs_sum, r_std, phi_std, theta_std, psi_std]).reshape(1,-1)

        # X_test = scaler.transform(X_test)

        # #print ("X_test: ", X_test)
        # if method == "MAX":
        #     pred = self.predict(X_test, model, device, method)
        #     cont = self.Controllers[pred]
        #     U = self.get_control_input(cont, current_traj)
        #     self.count_dict[cont] += 1

        # elif method == "DICE":
        #     pred = self.predict(X_test, model, device, method)
        #     cont = self.Controllers[pred]
        #     U = self.get_control_input(cont, current_traj)
        #     self.count_dict[cont] += 1

        # elif method == "MIX":
        #     probs = self.predict(X_test, model, device, method)
        #     U = 0
        #     for k in range(4):
        #         cont = self.Controllers[k]
        #         cont_prob = probs[k]
        #         U_single = self.get_control_input(cont, current_traj)
        #         U = U + (cont_prob * U_single)
        # else:
        #     #This time a controller info comes in method variable
        #     U = self.get_control_input(cont, current_traj)

        U = self.get_control_input("Backstepping_3", current_traj)

        sol = integrate.solve_ivp(fun=self.model_dynamics, t_span=(0, dtau), y0=self.state)
        self.state = sol.y[:,-1]
        self.U = U

        if (np.abs(self.state[3]) > np.pi)  | (np.abs(self.state[4]) > np.pi):
            self.costValue = 1e12
            fail_check = True
            print "Drone has crashed!"
        else:
            position_tracking_error = np.power((current_traj[0]-self.state[0]),2) + np.power((current_traj[1]-self.state[1]),2) + np.power((current_traj[2]-self.state[2]),2)
            velocity_tracking_error = np.power((current_traj[3]-self.state[6]),2) + np.power((current_traj[4]-self.state[7]),2) + np.power((current_traj[5]-self.state[8]),2)
            angular_error = np.power((current_traj[13]-self.state[5]),2)
            cont_input = self.U[0]**2 + self.U[1]**2 + self.U[2]**2 + self.U[3]**2
            self.costValue = self.costValue + (self.coeff_pos*position_tracking_error + 
                self.coeff_angle*angular_error + 
                self.coeff_control*cont_input)

        #print ("Cost Value: ", self.costValue)
        # if (method == "MAX" or method == "DICE" or method == "DT" or method == "FOREST"):
        #     print ("How many times Backstepping_1 is called?: ", self.count_dict["Backstepping_1"])
        #     print ("How many times Backstepping_2 is called?: ", self.count_dict["Backstepping_2"])
        #     print ("How many times Backstepping_3 is called?: ", self.count_dict["Backstepping_3"])
        #     print ("How many times Backstepping_4 is called?: ", self.count_dict["Backstepping_4"])
        #print ("Final state, x: {0:.3}, y: {1:.3}, z: {2:.3}, phi: {3:.3}, theta: {4:.3}, psi: {5:.3}, vx: {6:.3}, vy: {7:.3}, vz: {8:.3}, p: {9:.3}, q: {10:.3}, r: {11:.3}"
        #    .format(self.state[0],self.state[1],self.state[2],self.state[3],self.state[4],self.state[5], self.state[6],self.state[7],self.state[8], self.state[9],self.state[10],self.state[11]))

        return fail_check

    def calculate_cost(self, target):
        xd, yd, zd, psid = target
        position_tracking_error = (xd-self.state[0])**2 + (yd-self.state[1])**2 + (zd-self.state[2])**2
        angular_error = (np.abs(psid-self.state[5])-np.pi/2)**2 #in perfect conditions, difference between yaw angles of gate and drone should be pi/2
        cont_input = self.U[0]**2 + self.U[1]**2 + self.U[2]**2 + self.U[3]**2
        self.costValue = self.costValue + (self.coeff_pos*position_tracking_error + 
                        self.coeff_angle*angular_error + 
                        self.coeff_control*cont_input)



    def collect_data(self, Tf, dtau, i, current_traj, std_list):
    #     states: [x,y,z,phi,theta,psi,x_dot,y_dot,z_dot,phi_dot,theta_dot,psi_dot]
        r_std, phi_std, theta_std, psi_std = std_list
        N = 1 / dtau

        t_current = Tf / N * i

        fail_check = False

#         feature_names = ['x0', 'y0', 'z0', 'x_dot0','y_dot0','z_dot0', 'phi0','theta0','yaw0', 'phi_dot0','theta_dot0','yaw_dot0', 
#                  'xf', 'yf', 'zf', 'x_dotf','y_dotf','z_dotf','x_ddotf','y_ddotf','z_ddotf',
#                  'pos_diffx','pos_diffy','pos_diffz','time_rate','t', 'Tf', 
#                  'xp', 'yp', 'zp', 'x_dotp','y_dotp','z_dotp','x_ddotp','y_ddotp','z_ddotp', 'u_abs_p',
#                  'r_std', 'phi_std', 'psi_std']


        ## ADD NOISE ##
        self.state[6] = normal(self.state[6], 0*r_std / 3.0)
        self.state[7] = normal(self.state[7], 0*r_std / 3.0)
        self.state[8] = normal(self.state[8], 0*r_std / 3.0)
        self.state[9] = normal(self.state[9], 0*phi_std)
        self.state[10] = normal(self.state[10], 0*theta_std)
        self.state[11] = normal(self.state[11], 0*psi_std)

        U = self.get_control_input("Backstepping_4", current_traj)

        sol = integrate.solve_ivp(fun=self.model_dynamics, t_span=(0, dtau), y0=self.state)
        self.state = sol.y[:,-1]
        self.U = U

        if (np.abs(self.state[3]) > np.pi/2)  | (np.abs(self.state[4]) > np.pi/2):
            self.costValue = 1e12
            fail_check = True
            print "Drone has crashed!"
        else:
            target = [current_traj[0], current_traj[1], current_traj[2], current_traj[13]]
            self.calculate_cost(target)

        
        return fail_check
