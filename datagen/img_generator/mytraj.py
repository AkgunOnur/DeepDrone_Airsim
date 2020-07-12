from __future__ import print_function, division
import quadrocoptertrajectory as quadtraj
import numpy as np

class MyTraj:
    def __init__(self,gravity=[0,0,-9.81]): 
        self.gravity = gravity

    def givemetraj(self,pos0, vel0, acc0, posf, velf, accf,Tf):
        traj = quadtraj.RapidTrajectory(pos0, vel0, acc0, self.gravity)
        traj.set_goal_position(posf)
        traj.set_goal_velocity(velf)
        traj.set_goal_acceleration(accf)
        # Run the algorithm, and generate the trajectory.
        traj.generate(Tf)
        return traj


    def givemepoint(self,trajectory,t):
        self.des_pos = trajectory.get_position(t)
        self.des_vel = trajectory.get_velocity(t)
        self.des_acc = trajectory.get_acceleration(t)
        self.des_thrust = trajectory.get_thrust(t)
        self.des_rate = np.linalg.norm(trajectory.get_body_rates(t))
        return self.des_pos, self.des_vel, self.des_acc