def test_collision():
    phi = - np.pi/9
    theta =  np.pi/8
    psi = pi/2
    print "\nDrone Pos x={0:.3}, y={1:.3}, z={2:.3}".format(self.track[0].position.x_val, self.track[0].position.y_val, self.track[0].position.z_val)
    quad_pose = [self.track[0].position.x_val, self.track[0].position.y_val, self.track[0].position.z_val, -phi, -theta, psi]
    self.client.simSetVehiclePose(QuadPose(quad_pose), True)
    time.sleep(1)
    self.check_collision()

    rot_matrix = Rotation.from_quat([self.track[0].orientation.x_val, self.track[0].orientation.y_val, 
                                  self.track[0].orientation.z_val, self.track[0].orientation.w_val]).as_dcm().reshape(3,3)
    gate_x_range = [0.5, -0.75]
    gate_z_range = [0.5, -0.75]
    edge_ind = 0
    #print "\nGate Ind: {0}, Gate x={1:.3}, y={2:.3}, z={3:.3}".format(i+1, self.track[i].position.x_val, self.track[i].position.y_val, self.track[i].position.z_val)
    gate_pos = np.array([self.track[0].position.x_val, self.track[0].position.y_val, self.track[0].position.z_val])
    gate_edge_list = []
    for x_rng in gate_x_range:
        gate_edge_range = np.array([x_rng, 0., 0.])
        gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
        gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
        print "\nDrone Pos x={0:.3}, y={1:.3}, z={2:.3}".format(gate_edge_point[0], gate_edge_point[1], gate_edge_point[2])
        self.quad.state = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
        quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], -phi, -theta, psi]
        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.check_collision()
        time.sleep(5)
        

    for z_rng in gate_z_range:
        gate_edge_range = np.array([0., 0., z_rng])
        gate_edge_world = np.dot(rot_matrix, gate_edge_range.reshape(-1,1)).ravel()
        gate_edge_point = np.array([gate_pos[0]+gate_edge_world[0], gate_pos[1]+gate_edge_world[1], gate_pos[2]+gate_edge_world[2]])
        edge_ind += 1
        print "\nDrone Pos x={0:.3}, y={1:.3}, z={2:.3}".format(gate_edge_point[0], gate_edge_point[1], gate_edge_point[2])
        self.quad.state = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], phi, theta, psi, 0., 0., 0., 0., 0., 0.]
        quad_pose = [gate_edge_point[0], gate_edge_point[1], gate_edge_point[2], -phi, -theta, psi]
        self.client.simSetVehiclePose(QuadPose(quad_pose), True)
        self.check_collision()
        time.sleep(5) 



def check_collision(self, max_distance = 0.15):

        drone_x_range = [.1, -.1]
        drone_y_range = [.1, -.1]
        # drone_z_range = [.05, -.05]
        rot_matrix = R.from_euler('ZYX',[self.quad.state[5], self.quad.state[4], self.quad.state[3]],degrees=False).as_dcm()
        drone_pos = np.array([self.quad.state[0], self.quad.state[1], self.quad.state[2]])
        edge_ind = 0

        eps = 0.1

        for i, line in enumerate(self.line_list):
            edge_i, edge_j = line[0], line[1]
            same_point_ind = abs(edge_i - edge_j) <= eps
            diff_point_ind = abs(edge_i - edge_j) > eps
            edge_upper_limit = edge_i[same_point_ind] + max_distance
            edge_lower_limit = edge_i[same_point_ind] - max_distance
            
            on_same_line_cond = (edge_lower_limit < drone_pos[same_point_ind]) & (drone_pos[same_point_ind] < edge_upper_limit)
            if np.all(on_same_line_cond):
                if (edge_i[diff_point_ind] <= drone_pos[diff_point_ind] <= edge_j[diff_point_ind]) or \
                   (edge_j[diff_point_ind] <= drone_pos[diff_point_ind] <= edge_i[diff_point_ind]):
                    print "Collision detected!"
                    print "Ind: {0}, Drone center x={1:.3}, y={2:.3}, z={3:.3}".format(i, drone_pos[0], drone_pos[1], drone_pos[2])

                    return True

        for x_rng in drone_x_range:
            for y_rng in drone_y_range:
                for z_rng in drone_z_range:
                    drone_range = np.array([x_rng, y_rng, z_rng])
                    drone_range_world = np.dot(rot_matrix, drone_range.reshape(-1,1)).ravel()
                    drone_edge_point = np.array([drone_pos[0]+drone_range_world[0], drone_pos[1]+drone_range_world[1], drone_pos[2]+drone_range_world[2]])
                    edge_ind += 1
                    
                    
                    for i, line in enumerate(self.line_list):
                        edge_i, edge_j = line[0], line[1]
                        same_point_ind = abs(edge_i - edge_j) <= eps
                        diff_point_ind = abs(edge_i - edge_j) > eps 
                        edge_upper_limit = edge_i[same_point_ind] + max_distance
                        edge_lower_limit = edge_i[same_point_ind] - max_distance
                        
                        on_same_line_cond = (edge_lower_limit < drone_edge_point[same_point_ind]) & (drone_edge_point[same_point_ind] < edge_upper_limit)
                        if np.all(on_same_line_cond):
                            if (edge_i[diff_point_ind] <= drone_edge_point[diff_point_ind] <= edge_j[diff_point_ind]) or \
                               (edge_j[diff_point_ind] <= drone_edge_point[diff_point_ind] <= edge_i[diff_point_ind]):
                                print "Collision detected!"
                                print "Ind: {0}, Corner x={1:.3}, y={2:.3}, z={3:.3}".format(edge_ind, drone_edge_point[0], drone_edge_point[1], drone_edge_point[2])
                                return True


        return False




def isThereAnyGate(self, img_rgb):
        # loop over the boundaries
        lower, upper = [100, 17, 15], [200, 50, 56]
        create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")
        find the colors within the specified boundaries and apply
        the mask
        mask = cv2.inRange(img_rgb, lower, upper)
        output = cv2.bitwise_and(img_rgb, img_rgb, mask = mask)

        if output.any():
            #print "there is a gate on the frame!"
            return True

        return False


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

    final_target = [self.track[-1].position.x_val, self.track[-1].position.y_val, self.track[-1].position.z_val]


    # To check collision algorithm, comment it out
    # for i in range(100):
    #     gate_index = np.random.randint(0,5)
    #     self.test_collision(gate_index)

    if self.flight_log:
        f=open(self.log_path, "a")



    while((not track_completed) and (not fail_check)):
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


        with torch.no_grad():   
            # Determine Gat location with Neural Networks
            pose_gate_body = self.Dronet(image)

            if self.curr_idx % 6 == 0:
                noise_coeff = np.random.uniform(-1,1)
            elif self.curr_idx % 3 == 0:
                noise_coeff = 0.

            pose_gate_body[0][0] += (noise_coeff*pose_gate_body[0][0]) 

            pose_gate_body = np.asarray(pose_gate_body.reshape(-1,1).to('cpu')).reshape(-1,1)
            pose_prediction[self.curr_idx] = pose_gate_body.ravel()
            
            # for i,num in enumerate(pose_gate_body.reshape(-1,1)):
            #     #print(num, i , self.curr_idx)
            #     pose_prediction[self.curr_idx][i] = num.item()


            # if self.curr_idx % 12 == 0:
            #     self.brightness = random.uniform(30.,40.)
            #     self.contrast = random.uniform(30.,40.)
            #     self.saturation = random.uniform(30.,40.)
            #     self.transformation = transforms.Compose([
            #             transforms.Resize([200, 200]),
            #             #transforms.Lambda(self.gaussian_blur),
            #             transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            #             transforms.ToTensor()])
            # elif self.curr_idx % 6 == 0:
            #     self.brightness = random.uniform(0.,1.)
            #     self.contrast = random.uniform(0.,1.)
            #     self.saturation = random.uniform(0.,1.)
            #     self.transformation = transforms.Compose([
            #             transforms.Resize([200, 200]),
            #             #transforms.Lambda(self.gaussian_blur),
            #             transforms.ColorJitter(brightness=self.brightness, contrast=self.contrast, saturation=self.saturation),
            #             transforms.ToTensor()])

            if self.curr_idx >= 11:
                pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                
                for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
                    prediction_std[i] = p_g_c.item()
        
                # Gate ground truth values will be implemented
                pose_gate_body = pose_gate_body.reshape(-1,1)
                prediction_std = np.clip(prediction_std, 0, prediction_std)
                prediction_std = prediction_std.ravel()
                covariance_sum = np.sum(prediction_std)
               
                # Trajectory generate
                waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)

                pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                acc0 = [0.,0.,0.]
                posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]

                
                accf = [0.,0.,0.]
                yaw0 = self.quad.state[5]
                yaw_diff = pose_gate_body[3][0]
                yawf = (self.quad.state[5]+yaw_diff) + np.pi/2
                #yawf = Rotation.from_quat([self.track[self.current_gate].orientation.x_val, self.track[self.current_gate].orientation.y_val, 
                #                   self.track[self.current_gate].orientation.z_val, self.track[self.current_gate].orientation.w_val]).as_euler('ZYX',degrees=False)[0] - np.pi/2

                print "\nCurrent index: {0}".format(self.curr_idx)
                print "Predicted r: {0:.3}, Variance r: {1:.3}".format(pose_gate_body[0][0], prediction_std[0])
                print "Gate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)
                print "Brightness: {0:.3}, Contast: {1:.3}, Saturation: {2:.3}".format(self.brightness, self.contrast, self.saturation)
                #print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])
                if self.flight_log:
                    f.write("\nCurrent index: {0}".format(self.curr_idx))
                    f.write("\nPredicted r: {0:.3}, Variance r: {1:.3}".format(pose_gate_body[0][0], prediction_std[0]))
                    f.write("\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))
                    f.write("\nBrightness: {0:.3}, Contast: {1:.3}, Saturation: {2:.3}".format(self.brightness, self.contrast, self.saturation))
                    #f.write("\nVariance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]))
                    #f.write("\nBlur coefficient: {0:.3}/{1:.3}. Variance sum: {2:.3}".format(self.blur_coeff, self.blur_range, covariance_sum))

                min_jerk_check = False
                if use_model:
                    #true_init_x, true_init_y, true_init_z, blur_coeff, var_sum, diff_x, diff_y, diff_z, diff_phi, diff_theta, diff_psi, r_std, phi_std, theta_std, psi_std, Tf, MP_Method, Cost
                    #[arr[i][5],arr[i][6],arr[i][7],arr[i][8],arr[i][9],arr[i][10], arr[i][4],arr[i][11],arr[i][12],arr[i][13],arr[i][14],int(arr[i][16])]
                
                    X_test = np.array([posf[0]-pos0[0], posf[1]-pos0[1], posf[2]-pos0[2], -self.quad.state[3], -self.quad.state[4], yawf-yaw0,
                                     covariance_sum, prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]]).reshape(1,-1)

                    X_mp_test = self.mp_scaler.transform(X_test)
                    X_time_test = self.time_scaler.transform(X_test)

                    mp_method = self.predict(X_mp_test, model=self.mp_classifier, isClassifier=True)

                    self.trajSelect[0] = labels_dict[mp_method] 
                    self.trajSelect[1] = 2
                    self.trajSelect[2] = 0

                    if self.trajSelect[0] != -1:
                        print "Predicted MP Algorithm: ", self.MP_names[int(self.trajSelect[0])]

                        self.Tf = self.predict(X_time_test, model=self.time_regressor, isClassifier=False)
                        print "Time based trajectory, T: {0:.3}".format(self.Tf)
                        print "Predicted Time Length: {0:.3}".format(self.Tf)
                        if self.flight_log:
                            f.write("\nTime based trajectory, T: {0:.3}".format(self.Tf))
                            f.write("\nPredicted Time Length: {0:.3}".format(self.Tf))
                    else:
                        print "Drone is in Safe Mode"
                        if self.flight_log:
                            f.write("\nDrone is in Safe Mode")

                    

                    if labels_dict[mp_method] == 5: #min_jerk
                        min_jerk_check = True
                        
                else:
                    self.trajSelect[0] = self.MP_methods[method]
                    self.trajSelect[1] = 2
                    self.trajSelect[2] = 0
                    self.Tf = self.time_coeff*abs(pose_gate_body[0][0])
                    #print "Prediction mode is off. MP algorithm: " + method 
                    #print "Estimated time of arrival: " + str(self.Tf) + " s."
                    if self.flight_log:
                        f.write("\nPrediction mode is off. MP algorithm: " + method)
                        f.write("\nEstimated time of arrival: " + str(self.Tf) + " s.")

                    if method == "min_jerk":
                        min_jerk_check = True
                        
                
                if self.trajSelect[0] != -1:
                    velf = [float(posf[0]-pos0[0])/self.Tf, float(posf[1]-pos0[1])/self.Tf, float(posf[2]-pos0[2])/self.Tf]
                    time_list = np.hstack((0., self.Tf)).astype(float)
                    waypoint_list = np.vstack((pos0, posf)).astype(float)
                    yaw_list = np.hstack((yaw0, yawf)).astype(float)

                    self.test_arrival_time[method] += self.Tf

                    if min_jerk_check:
                        mp_algorithm = MyTraj(gravity = -9.81)
                        traj = mp_algorithm.givemetraj(pos0, vel0, acc0, posf, velf, accf, self.Tf)

                    newTraj = Trajectory(self.trajSelect, self.quad.state, time_list, waypoint_list, yaw_list) 

                    flight_period = self.Tf / self.period_denum
                    Waypoint_length = flight_period // self.dtau

                    if init_start:
                        t_list = linspace(0, flight_period, num = Waypoint_length)
                        init_start = False
                    else:
                        t_list = linspace(flight_period, 2*flight_period, num = Waypoint_length)
                                        
                    
                    self.vel_sum = 0.
                    self.quad.costValue = 0.
                    # Call for Controller
                    for ind, t_current in enumerate(t_list): 

                        self.vel_sum += (self.quad.state[6]**2+self.quad.state[7]**2+self.quad.state[8]**2)

                        pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)
                        if min_jerk_check:
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
                            self.vel_sum = self.vel_sum / (ind + 1)
                            self.test_costs[method] = self.Tf * self.quad.costValue
                            print "Drone has finished the lap. Current cost: {0:.6}".format(self.test_costs[method]) 
                            if self.flight_log:
                                f.write("\nDrone has finished the lap. Current cost: {0:.6}".format(self.test_costs[method]))
                            break        
                        


                    if (not track_completed) and (not fail_check) and (not collision_check) and (anyGate): # drone didn't arrive or crash
                        self.vel_sum = self.vel_sum / Waypoint_length
                        #print "Velocity Sum (Normalized): ", self.vel_sum
                        self.test_costs[method] = self.Tf * self.quad.costValue
                        print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_costs[method])
                        if self.flight_log:
                            f.write("\nDrone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_costs[method]))


                    if track_completed or fail_check or collision_check or not anyGate: # drone arrived to the gate or crashed or collided                      
                        break

        self.curr_idx += 1

    if self.flight_log:
        f.close()



def check_collision(self, max_distance = 0.15):

    drone_x_range = [.1, -.1]
    drone_y_range = [.1, -.1]
    # drone_z_range = [.05, -.05]
    rot_matrix = R.from_euler('ZYX',[self.quad.state[5], self.quad.state[4], self.quad.state[3]],degrees=False).as_dcm()
    drone_pos = np.array([self.quad.state[0], self.quad.state[1], self.quad.state[2]])
    edge_ind = 0

    eps = 0.1


    for i, line in enumerate(self.line_list):
        distance = line.distance(Point3D(drone_pos[0], drone_pos[1], drone_pos[2])).evalf()
        edge_i, edge_j, u_v = self.line_list_2[i]
        distance_from_center = edge_i - drone_pos
        distance_2 = np.linalg.norm(np.cross(distance_from_center, u_v)) / np.linalg.norm(u_v)
        print "Edge: {0}, (Symbolic) Distance from the center: {1:.3}".format(i, distance) 
        print "Edge: {0}, (Numeric) Distance from the center: {1:.3}".format(i, distance_2) 
        if distance < max_distance:
            print "Collision detected!"
            #print "Index: {0}, Drone center x={1:.3}, y={2:.3}, z={3:.3}".format(i, drone_pos[0], drone_pos[1], drone_pos[2])

            return True

    # for x_rng in drone_x_range:
    #     for y_rng in drone_y_range:
    #         # for z_rng in drone_z_range:
    #         drone_range = np.array([x_rng, y_rng, 0.])
    #         drone_range_world = np.dot(rot_matrix, drone_range.reshape(-1,1)).ravel()
    #         drone_edge_point = np.array([drone_pos[0]+drone_range_world[0], drone_pos[1]+drone_range_world[1], drone_pos[2]+drone_range_world[2]])
    #         edge_ind += 1
            
            
    #         for i, line in enumerate(self.line_list):
    #             distance = line.distance(Point3D(drone_edge_point[0], drone_edge_point[1], drone_edge_point[2])).evalf()
    #             #print "Edge: {0}, Distance from the center: {1:.3}".format(i, distance) 
    #             if distance < max_distance:
    #                 print "Collision detected!"
    #                 print "Ind: {0}, Drone center x={1:.3}, y={2:.3}, z={3:.3}".format(i, drone_pos[0], drone_pos[1], drone_pos[2])

    #                 return True
    
    #print "No Collision!"
    return False

    # def fly_drone(self, f, gate_index, method, pos_ranges, angles_start):
    #     pose_prediction = np.zeros((2000,4),dtype=np.float32)
    #     prediction_std = np.zeros((4,1),dtype=np.float32)

    #     x_range, y_range, z_range = pos_ranges
    #     phi_start, theta_start, gate_psi, psi_start = angles_start
        
    #     if gate_index == 0: #if drone is at initial point
    #         quad_pose = [self.drone_init.position.x_val+x_range, self.drone_init.position.y_val+y_range, self.drone_init.position.z_val+z_range, -phi_start, -theta_start, psi_start]
    #         self.state0 = [self.drone_init.position.x_val+x_range, self.drone_init.position.y_val+y_range, self.drone_init.position.z_val+z_range, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]
    #         true_init_pos = [self.drone_init.position.x_val, self.drone_init.position.y_val, self.drone_init.position.z_val]
    #     else:
    #         quad_pose = [self.track[gate_index-1].position.x_val+x_range, self.track[gate_index-1].position.y_val+y_range, self.track[gate_index-1].position.z_val+z_range, -phi_start, -theta_start, psi_start]
    #         self.state0 = [self.track[gate_index-1].position.x_val+x_range, self.track[gate_index-1].position.y_val+y_range, self.track[gate_index-1].position.z_val+z_range, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]
    #         true_init_pos = [self.track[gate_index-1].position.x_val, self.track[gate_index-1].position.y_val, self.track[gate_index-1].position.z_val]


    #     self.client.simSetVehiclePose(QuadPose(quad_pose), True)
    #     self.quad = Quadrotor(self.state0)
    #     # this is only used for yaw motion planning
    #     self.trajSelect[0] = self.MP_methods[method]
    #     self.trajSelect[1] = 2 
    #     self.trajSelect[2] = 0
    #     self.curr_idx = 0
    #     #self.Controller_states[method].append(self.quad.state)
    #     self.MP_states[method].append(self.quad.state)

    #     self.xd_ddot_pr = 0.
    #     self.yd_ddot_pr = 0.
    #     self.xd_dddot_pr = 0.
    #     self.yd_dddot_pr = 0.
    #     self.psid_pr = 0.
    #     self.psid_dot_pr = 0.

        
    #     cov_coeff = 0.

    #     initial_start = True

    #     final_target = [self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val]


    #     print "\n>>>MP Method: ", method
    #     track_completed = False
    #     fail_check = False

    #     if self.flight_log:
    #         f.write("\n\n>>>MP Method: %s " %method)

    #     while((not track_completed) and (not fail_check)):
    #         image_response = self.client.simGetImages([airsim.ImageRequest('0', airsim.ImageType.Scene, False, False)])[0]
    #         #if len(image_response.image_data_uint8) == image_response.width * image_response.height * 3:
    #         img1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)  # get numpy array
    #         img_rgb = img1d.reshape(image_response.height, image_response.width, 3)  # reshape array to 4 channel image array H X W X 3
    #         img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    #         #cv2.imwrite(os.path.join(self.base_path, 'images', "frame" + str(self.curr_idx).zfill(len(str(self.num_samples))) + '.png'), img_rgb)
    #         img =  Image.fromarray(img_rgb)
    #         image = self.transformation(img)
    #         quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]


    #         with torch.no_grad():   
    #             # Determine Gat location with Neural Networks
    #             pose_gate_body = self.Dronet(image)
                
    #             for i,num in enumerate(pose_gate_body.reshape(-1,1)):
    #                 #print(num, i , self.curr_idx)
    #                 pose_prediction[self.curr_idx][i] = num.item()

    #             if self.curr_idx >= 11:
    #                 pose_gate_cov = self.lstmR(torch.from_numpy(pose_prediction[self.curr_idx-11:self.curr_idx+1].reshape(1,12,4)).to(self.device))
                    
    #                 for i, p_g_c in enumerate(pose_gate_cov.reshape(-1,1)):
    #                     prediction_std[i] = p_g_c.item()
            
    #                 # Gate ground truth values will be implemented
    #                 pose_gate_body = pose_gate_body.numpy().reshape(-1,1)
    #                 prediction_std = np.clip(prediction_std, 0, prediction_std)
    #                 prediction_std = prediction_std.ravel()
    #                 covariance_sum = np.sum(prediction_std)
                    
    #                 # r,theta,psi,phi = pose_gate_body[0][0],pose_gate_body[1][0],pose_gate_body[2][0],pose_gate_body[3][0] # notation is different. In our case, phi equals to psi
    #                 # q1,q2,q3,q4 = R.from_euler('ZYX',[self.quad.state[5], self.quad.state[4], self.quad.state[3]], degrees=False).as_quat()
    #                 # quad_pose = Pose(Vector3r(self.quad.state[0], self.quad.state[1],self.quad.state[2]),Quaternionr(q1,q2,q3,q4))
    #                 # estimation = geom_utils.debugGatePoses(quad_pose , r, theta, psi)
                    
    #                 # Trajectory generate
    #                 self.Tf = self.time_coeff*pose_gate_body[0][0] # T=r*0.5
    #                 waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
    #                 pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
    #                 vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
    #                 #acc0 = [float(self.quad.state[6]-self.x_dot_pr)/self.dtau, float(self.quad.state[7]-self.y_dot_pr)/self.dtau, float(self.quad.state[8]-self.z_dot_pr)/self.dtau]
    #                 acc0 = [0., 0., 0.]
    #                 yaw0 = self.quad.state[5]

    #                 posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
    #                 velf = [float(posf[0]-pos0[0])/self.Tf, float(posf[1]-pos0[1])/self.Tf, float(posf[2]-pos0[2])/self.Tf]
    #                 accf = [0., 0., 0.]
    #                 yaw_diff = pose_gate_body[3][0]
    #                 yawf = (self.quad.state[5]+yaw_diff) + np.pi/2

    #                 pos_next = np.array(velf) * self.Tf

    #                 time_list = np.hstack((0., self.Tf, 2*self.Tf)).astype(float)
    #                 waypoint_list = np.vstack((pos0, posf, pos_next)).astype(float)
    #                 yaw_list = np.hstack((yaw0, yawf, yawf)).astype(float)

    #                 newTraj = Trajectory(self.trajSelect, self.quad.state, time_list, waypoint_list, yaw_list) 

    #                 print "\nTime of arrival: {0:.3} s., time coefficient: {1:.3}".format(self.Tf, self.time_coeff)
    #                 print "Gate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)
    #                 print "Gate Ground truth, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(final_target[0], final_target[1], final_target[2], (gate_psi-np.pi/2)*180/np.pi)
    #                 print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])
    #                 print "Blurring coefficient: {0:.3}/{1:.3}".format(self.blur_coeff,self.blur_range)


    #                 if self.flight_log:
    #                     f.write("\nTime of arrival: {0:.3}, time coefficient: {1:.3}".format(self.Tf, self.time_coeff))
    #                     f.write("\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))
    #                     f.write("\nGate Ground truth, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(self.track[gate_index].position.x_val, self.track[gate_index].position.y_val, self.track[gate_index].position.z_val, (gate_psi-np.pi/2)*180/np.pi))
    #                     f.write("\nVariance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]))
    #                     f.write("\nBlurring coefficient: {0:.3}".format(self.blur_coeff))
                    
    #                 # if initial_start:
    #                 #     init_time = 0.
    #                 #     initial_start = False
    #                 # else:
    #                 #     init_time = newTraj.t_wps[1] / 6.0


    #                 Waypoint_length = newTraj.t_wps[1] // self.dtau
    #                 t_list = linspace(0., newTraj.t_wps[1], num = Waypoint_length)


    #                 # if time_or_speed == 0:
    #                 #     newTraj = Trajectory(self.trajSelect, self.quad.state, self.Tf, pos0, posf, yaw0, yawf)
    #                 #     print "Time based trajectory, T: {0:.3}".format(newTraj.t_wps[1])
    #                 #     if self.flight_log:
    #                 #         f.write("Time based trajectory, T: {0:.3}".format(newTraj.t_wps[1]))
    #                 # else:
    #                 #     newTraj = Trajectory(self.trajSelect, self.quad.state, 1.0, pos0, posf, yaw0, yawf, v_average=self.v_average)
    #                 #     print "Velocity based trajectory, V_avg: {0:.3}, T: {1:.3}".format(self.v_average, newTraj.t_wps[1])
    #                 #     if self.flight_log:
    #                 #         f.write("Velocity based trajectory, V_avg: {0:.3}, T: {1:.3}".format(self.v_average, newTraj.t_wps[1]))

                    

    #                 # mp_algorithm = MyTraj(gravity = -9.81)
    #                 # traj = mp_algorithm.givemetraj(pos0, vel0, acc0, posf, velf, accf, self.Tf)
                    
    #                 # Waypoint_length = int(self.Tf / self.dtau)
    #                 # t_list = linspace(0., self.Tf, num = Waypoint_length)
                    

    #                 for t_current in t_list: 
    #                     pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)

    #                     #pos_des, vel_des, acc_des = mp_algorithm.givemepoint(traj, t_current)

    #                     xd, yd, zd = pos_des[0], pos_des[1], pos_des[2]
    #                     xd_dot, yd_dot, zd_dot = vel_des[0], vel_des[1], vel_des[2]
    #                     xd_ddot, yd_ddot, zd_ddot = acc_des[0], acc_des[1], acc_des[2]

    #                     xd_dddot = (xd_ddot - self.xd_ddot_pr) / self.dtau
    #                     yd_dddot = (yd_ddot - self.yd_ddot_pr) / self.dtau
    #                     xd_ddddot = (xd_dddot - self.xd_dddot_pr) / self.dtau
    #                     yd_ddddot = (yd_dddot - self.yd_dddot_pr) / self.dtau

    #                     psid = euler_des[2]

    #                     psid_dot = (psid - self.psid_pr) / self.dtau
    #                     psid_ddot = (psid_dot - self.psid_dot_pr) / self.dtau

    #                     current_traj = [xd, yd, zd, xd_dot, yd_dot, zd_dot, xd_ddot, yd_ddot, zd_ddot,
    #                                  xd_dddot, yd_dddot, xd_ddddot, yd_ddddot,
    #                                  psid, psid_dot, psid_ddot]

    #                     fail_check = self.quad.simulate(self.dtau, current_traj, final_target, prediction_std)

    #                     quad_pose = [self.quad.state[0], self.quad.state[1], self.quad.state[2], -self.quad.state[3], -self.quad.state[4], self.quad.state[5]]
    #                     self.MP_states[method].append(self.quad.state)
    #                     self.client.simSetVehiclePose(QuadPose(quad_pose), True)


    #                     self.xd_ddot_pr = xd_ddot
    #                     self.yd_ddot_pr = yd_ddot
    #                     self.xd_dddot_pr = xd_dddot
    #                     self.yd_dddot_pr = yd_dddot
    #                     self.psid_pr = psid
    #                     self.psid_dot_pr = psid_dot


    #                     if fail_check:
    #                         self.MP_cost[method] = self.quad.costValue
    #                         print "Drone has crashed! Current cost: {0:.6}".format(self.MP_cost[method])
    #                         if self.flight_log:
    #                             f.write("\nDrone has crashed! Current cost: {0:.6}".format(self.MP_cost[method]))
    #                         break 

    #                     check_arrival, on_road = self.check_completion(quad_pose, gate_index=gate_index)

    #                     if check_arrival: # drone arrives to the gate
    #                         track_completed = True
    #                         self.MP_cost[method] = self.Tf * self.quad.costValue + cov_coeff*np.sum(prediction_std)
    #                         print "Drone has arrived to the {0}. gate. Current cost: {1:.6}".format(gate_index+1, self.MP_cost[method]) 
    #                         if self.flight_log:
    #                             f.write("\nDrone has arrived to the {0}. gate. Current cost: {1:.6}".format(gate_index+1, self.MP_cost[method]))
    #                         break        
    #                     elif not on_road: #drone can not complete the path, but still loop should be ended
    #                         track_completed = True
    #                         self.MP_cost[method] = self.quad.costValue
    #                         print "Drone is out of the path. Current cost: {0:.6}".format(self.MP_cost[method])
    #                         if self.flight_log:
    #                             f.write("\nDrone is out of the path. Current cost: {0:.6}".format(self.MP_cost[method]))
    #                         break


    #                 if (not track_completed) and (not fail_check): # drone didn't arrive or crash
    #                     self.MP_cost[method] = self.Tf * self.quad.costValue + cov_coeff*np.sum(prediction_std)
    #                     print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.MP_cost[method])
    #                     if self.flight_log:
    #                         f.write("\nDrone hasn't reached the gate yet. Current cost: {0:.6}".format(self.MP_cost[method]))

    #                 self.write_stats(flight_columns,
    #                     [true_init_pos[0], true_init_pos[1], true_init_pos[2], posf[0]-pos0[0], posf[1]-pos0[1], posf[2]-pos0[2], -phi_start, -theta_start, yawf-yaw0,
    #                      prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3], self.Tf, method, self.MP_cost[method]], flight_filename)
    #                 #print "Flight data is written to the file"

    #                 self.quad.costValue = 0.
    #                 if track_completed or fail_check: # drone arrived to the gate or crashed                        
    #                     break

    #         self.curr_idx += 1



    def fly_drone_2(self, f, method, pos_offset, angle_start):
        pose_prediction = np.zeros((1000,4),dtype=np.float32)
        prediction_std = np.zeros((4,1),dtype=np.float32)

        x_offset, y_offset, z_offset = pos_offset
        phi_start, theta_start, gate_psi, psi_start = angle_start

        #if drone is at initial point
        quad_pose = [self.drone_init.position.x_val+x_offset, self.drone_init.position.y_val+y_offset, self.drone_init.position.z_val+z_offset, -phi_start, -theta_start, psi_start]
        self.state0 = [self.drone_init.position.x_val+x_offset, self.drone_init.position.y_val+y_offset, self.drone_init.position.z_val+z_offset, phi_start, theta_start, psi_start, 0., 0., 0., 0., 0., 0.]

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

        final_target = [self.track[-1].position.x_val, self.track[-1].position.y_val, self.track[-1].position.z_val]


        self.trajSelect[0] = 3 #min_vel
        self.trajSelect[1] = 2 #yaw_follow
        self.trajSelect[2] = 0 #time based


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
                    
                    self.Tf = self.time_coeff*pose_gate_body[0][0]
                    # Trajectory generate
                    waypoint_world = spherical_to_cartesian(self.quad.state, pose_gate_body)
                    waypoint_world = [self.track[self.current_gate].position.x_val, self.track[self.current_gate].position.y_val, self.track[self.current_gate].position.z_val]
                    pos0 = [self.quad.state[0], self.quad.state[1], self.quad.state[2]]
                    vel0 = [self.quad.state[6], self.quad.state[7], self.quad.state[8]]
                    acc0 = [0., 0., 0.]
                    posf = [waypoint_world[0], waypoint_world[1], waypoint_world[2]]
                    velf = [float(posf[0]-pos0[0])/self.Tf, float(posf[1]-pos0[1])/self.Tf, float(posf[2]-pos0[2])/self.Tf]
                    accf = [0., 0., 0.]
                    yaw0 = self.quad.state[5]
                    yaw_diff = pose_gate_body[3][0]
                    yawf = (self.quad.state[5]+yaw_diff) + np.pi/2
                    yawf = Rotation.from_quat([self.track[self.current_gate].orientation.x_val, self.track[self.current_gate].orientation.y_val, 
                                       self.track[self.current_gate].orientation.z_val, self.track[self.current_gate].orientation.w_val]).as_euler('ZYX',degrees=False)[0] - np.pi/2

                    print "MP algorithm: " + method 
                    print "Estimated time of arrival: " + str(self.Tf) + " s."
                    if self.flight_log:
                        f.write("\nPrediction mode is off. MP algorithm: " + method)
                        f.write("\nEstimated time of arrival: " + str(self.Tf) + " s.")

                    print "\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi)
                    print "Variance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3])
                    print "Blurring coefficient: {0:.3}/{1:.3}".format(self.blur_coeff,self.blur_range)
                    if self.flight_log:
                        f.write("\nGate Predicted, x: {0:.3}, y: {1:.3}, z: {2:.3}, psi: {3:.3} deg".format(waypoint_world[0], waypoint_world[1], waypoint_world[2], yawf*180/np.pi))
                        f.write("\nVariance values, r: {0:.3}, phi: {1:.3}, theta: {2:.3}, psi: {3:.3}".format(prediction_std[0], prediction_std[1], prediction_std[2], prediction_std[3]))
                        f.write("\nBlurring coefficient: {0:.3}/{1:.3}".format(self.blur_coeff,self.blur_range))

                    
                    
                    if method == "min_jerk":
                        mp_algorithm = MyTraj(gravity = -9.81)
                        traj = mp_algorithm.givemetraj(pos0, vel0, acc0, posf, velf, accf, self.Tf)
                        

                    else:
                        self.trajSelect[0] = self.MP_methods[method]
                         
                        
                    # pos_next = np.array(velf) * self.Tf
                    # time_list = np.hstack((0., self.Tf, 2*self.Tf)).astype(float)
                    # waypoint_list = np.vstack((pos0, posf, pos_next)).astype(float)
                    time_list = np.hstack((0., self.Tf)).astype(float)
                    waypoint_list = np.vstack((pos0, posf)).astype(float)
                    yaw_list = np.hstack((yaw0, yawf)).astype(float)
                    newTraj = Trajectory(self.trajSelect, self.quad.state, time_list, waypoint_list, yaw_list)
                    Waypoint_length = int(self.Tf / self.dtau)
                    t_list = linspace(0., self.Tf, num = Waypoint_length)
                        


                    # Call for Controller
                    for t_current in t_list: 

                        pos_des, vel_des, acc_des, euler_des = newTraj.desiredState(t_current, self.dtau, self.quad.state)
                        if method == "min_jerk":
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

                        fail_check = self.quad.simulate(self.dtau, current_traj, final_target, prediction_std)

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
                                f.write("\nDrone has crashed! Current cost: {0:.6}".format(self.test_cost))
                            break 

                        check_arrival, on_road = self.check_completion(quad_pose, -1, test_control = True)

                        if check_arrival: # drone arrives to the gate
                            track_completed = True
                            self.test_cost = newTraj.t_wps[1] * self.quad.costValue
                            print "Drone has arrived finished the lap. Current cost: {0:.6}".format(self.test_cost) 
                            if self.flight_log:
                                f.write("\nDrone has finished the lap. Current cost: {0:.6}".format(self.test_cost))
                            break        
                        elif not on_road: #drone can not complete the path, but still loop should be ended
                            track_completed = True
                            self.test_cost = self.quad.costValue
                            print "Drone is out of the path. Current cost: {0:.6}".format(self.test_cost)
                            if self.flight_log:
                                f.write("\nDrone is out of the path. Current cost: {0:.6}".format(self.test_cost))
                            break


                    if (not track_completed) and (not fail_check): # drone didn't arrive or crash
                        self.test_cost = newTraj.t_wps[1] * self.quad.costValue
                        print "Drone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost)
                        if self.flight_log:
                            f.write("\nDrone hasn't reached the gate yet. Current cost: {0:.6}".format(self.test_cost))


                    if track_completed or fail_check: # drone arrived to the gate or crashed                        
                        break

            self.curr_idx += 1


# def visualize_drone(self, MP_list):
    #     for algorithm in MP_list:
    #         print "Drone flies by the algorithm, ", algorithm
    #         self.client.simSetVehiclePose(self.drone_init, True)
    #         state_list = self.MP_states[algorithm]
    #         for state in state_list:
    #             quad_pose = [state[0], state[1], state[2], -state[3], -state[4], state[5]]
    #             self.client.simSetVehiclePose(QuadPose(quad_pose), True)
    #             time.sleep(0.001)

    # def get_video(self, algorithm):

    #     pathIn= self.base_path + 'images/'
    #     pathOut = self.base_path + algorithm + '_video.avi'
    #     fps = 0.5
    #     frame_array = []
    #     files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]#for sorting the file names properly
    #     files.sort(key = lambda x: x[5:-4])
    #     for i in range(len(files)):
    #         filename=pathIn + files[i]
    #         #reading each files
    #         img = cv2.imread(filename)
    #         height, width, layers = img.shape
    #         size = (width,height)
            
    #         #inserting the frames into an image array
    #         frame_array.append(img)

    #     out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    #     for i in range(len(frame_array)):
    #         # writing to a image array
    #         out.write(frame_array[i])
    #     out.release()

    

