import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import normalize, resize, to_tensor
from torch.nn import functional as F

from Dronet import ResNet, BasicBlock
from lstmf import LstmNet
from dataprocessing import DatasetProcessing
from numpy.linalg import inv
#-------------------------- System Check ------------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print (os.getcwd())

#-------------------------- Initialization ---------------------
batch_size = 1
time_stamp = 12
num_workers = 4

# Lstmf Model 
input_size = 4

lstmf_hidden_size = 16
lstmf_num_layers = 1

lstmQ_hidden_size = 16
lstmQ_num_layers = 1

lstmR_hidden_size = 16
lstmR_num_layers = 1
output_size = 4



#----------------- Data For Dronet Model -----------------------------
data_path = 'C:\\Users\deepd\OneDrive\Masaüstü\\Dataset\\airsim_dataset'

test_data = 'test'

test_label_file = 'test_label.txt'

test_image_file = 'test_img.txt'
#----------------------------------------------------------------------------------

#--------------------------------Import-Dataset------------------------------------
transformations = transforms.Compose([
        transforms.Resize([200, 200]),
        transforms.ToTensor()]
        )
#transforms.CenterCrop(200),
#transforms.ColorJitter(hue=.05, saturation=.05),
#transforms.Grayscale(num_output_channels=1),
        
dset_test = DatasetProcessing(
    data_path, test_data, test_image_file, test_label_file, transformations)


image_datasets = {}
image_datasets["test"] = dset_test

dataset_sizes = {}
dataset_sizes["test"] = len(image_datasets["test"])



test_loader = DataLoader(dset_test,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers
                         )
dataloaders = {}
dataloaders['test'] = test_loader

#val_data_path = '/home/deepdrone/Dataset/OurBasic/val_label.txt'
test_data_path = "C:\\Users\deepd\OneDrive\Masaüstü\\Dataset\\airsim_dataset\\test_label.txt"
test_data = np.loadtxt(test_data_path, dtype=np.float32)
test_data_size = test_data.shape[0]

dronet_data_path = "C:\\Users\deepd\OneDrive\Masaüstü\\Dataset\\airsim_dataset\\predfor_Rtest.txt"
dronet_data = np.loadtxt(dronet_data_path, dtype=np.float32)
dronet_data_size = dronet_data.shape[0]

test_imu_data_path = "C:\\Users\deepd\OneDrive\Masaüstü\\Dataset\\airsim_dataset\\test_label_imu.txt"
test_imu_data = np.loadtxt(test_imu_data_path, dtype=np.float32)
test_imu_data_size = test_imu_data.shape[0]


#----------------------------------------------------------------------------------
if __name__ == "__main__":
    print("Test data size {} images".format(test_data_size))
    Dronet =  ResNet(BasicBlock, [1,1,1,1], num_classes = 4)
    Dronet.to(device)
    print("Dronet Model:",Dronet)
    Dronet.load_state_dict(torch.load('C:\\Users\\deepd\\OneDrive\\Masaüstü\\newlittle\\model\\test_6\\16_0.0001_10_loss_0.0137_PG.pth'))   
    Dronet.eval()
    #--------------------------------------------------------
    lstmf = LstmNet(input_size, output_size, lstmf_hidden_size, lstmf_num_layers)
    lstmf.to(device)
    print("LSTMf Model:", lstmf)
    #lstmf.load_state_dict(torch.load('C:\\Users\\deepd\\OneDrive\\Masaüstü\\LSTMf\\model\\test_9_RESNET\\256\\256hidden_2_0.0001_7_loss_0.1466_PG.pth'))   
    lstmf.load_state_dict(torch.load('C:\\Users\\deepd\\OneDrive\\Masaüstü\\LSTMf\\model\\test_6\\2_0.0001_18_loss_0.1160_PG.pth'))
    lstmf.eval()

    lstmQ = LstmNet(input_size, output_size, lstmQ_hidden_size, lstmQ_num_layers)
    lstmQ.to(device)
    print("lstmQ Model:", lstmQ)
    #lstmQ.load_state_dict(torch.load('C:\\Users\deepd\OneDrive\Masaüstü\\LSTMQR\\model\\testQ_5\\16_2_0.0001_7_loss_6.6478_PG.pth'))
    lstmQ.load_state_dict(torch.load('C:\\Users\deepd\OneDrive\Masaüstü\\LSTMQR\\model\\testQ_4\\2_0.0005_13_loss_5.2322_PG.pth'))   
    lstmQ.eval()

    lstmR = LstmNet(input_size, output_size, lstmR_hidden_size, lstmR_num_layers)
    lstmR.to(device)
    print("lstmR Model:", lstmR)
    lstmR.load_state_dict(torch.load('C:\\Users\deepd\OneDrive\Masaüstü\LSTMQR\model\\testR_4\\2_0.0005_1_loss_0.0049_PG.pth'))   
    lstmR.eval()    

    #---------------------------------------------------------------------------

    criterion = nn.MSELoss()


    # Kalman Filter Initilization
    P_t = np.eye(4)
    #dronet_data = np.zeros((test_data_size,4),dtype=np.float32)
    y_lstmf = np.zeros((test_data_size,4),dtype=np.float32)
    y_estimate = np.zeros((test_data_size,4),dtype=np.float32)
    Q_log = np.zeros((test_data_size,4),dtype = np.float32)
    R_log = np.zeros((test_data_size,4),dtype= np.float32)
    label_list=[]
    running_loss = 0.0
    counter = 0
    #y = {s: np.zeros((4,1)) for s in range(783)}
    error_depth = []
    error_angle1 = []
    error_angle3 = []
    error_angle2 = []



    # Save Dronet model outputs
    """for i, (inputs, labels)  in enumerate(dataloaders['test']):
        inputs = inputs.to(device)
        labels = labels.to(device) 
        
        #print(i)   
        outputs = Dronet(inputs)
        #print("outputs.shape:", outputs.shape)
        for j, k in enumerate(outputs.T):
            #print(j,k.item())
            dronet_data[i][j] = k.item()"""

    #print(dronet_data, len(dronet_data))
    
    for i in range(12,test_data_size):
        
        with torch.no_grad():
            
            lstm_inputs = dronet_data[i-time_stamp:i].reshape(1,12,4)
            lstm_inputs = torch.from_numpy(lstm_inputs)
            lstm_inputs = lstm_inputs.to(device)
            
            output_lstm = lstmf(lstm_inputs)
            
            for j, k in enumerate(output_lstm.T):
                y_lstmf[i][j] = k.item()
            
            
            if i > 2*time_stamp -2:  
                # PROCESS 
                
                F  = (y_lstmf[i]-y_lstmf[i-1])
                lstmQ_inputs = y_lstmf[i-time_stamp+1:i+1].reshape(1,12,4)
                lstmQ_inputs = torch.from_numpy(lstmQ_inputs)
                lstmQ_inputs = lstmQ_inputs.to(device)

                Q = np.zeros((4,4), dtype=np.float32)
                Q_log[i] = np.abs(np.array(lstmQ(lstmQ_inputs).to('cpu')))
                np.fill_diagonal(Q,Q_log[i])
                #Q = np.eye((4), dtype=np.float32)*0.5
                #P_t_= F*P_t*F.T + Q            
                P_t_= np.dot(np.dot(F,P_t),F.T) + Q            
                

                # UPDATE
                lstmR_inputs = dronet_data[i - time_stamp + 1:i+1].reshape(1,12,4)
                lstmR_inputs = torch.from_numpy(lstmR_inputs)
                lstmR_inputs = lstmR_inputs.to(device)

                R = np.zeros((4,4), dtype=np.float32)
                R_log[i] = np.abs(np.array((lstmR(lstmR_inputs)).to('cpu')))
                np.fill_diagonal(R,R_log[i])
                #print(R, R_log[i])

                K = np.dot(P_t_, inv(P_t_ + R))
                
                y_estimate[i] = y_lstmf[i] + np.dot(K,(dronet_data[i] - y_lstmf[i]))
                P_t = np.dot((np.eye(len(K)) - K),P_t_)
                #print(P_t)
                #print("i:", i,"estimate:", y_estimate[i], "lstm:,", y_lstmf[i], "dronet:", y[i])
            


    for i, labels in enumerate(test_data):

        if i > 2*time_stamp -2: 
            counter += 1
            labels = torch.from_numpy(labels).to(device) 
            loss = criterion(torch.from_numpy(y_estimate[i].reshape(-1,1)).to(device), labels.reshape(-1,1))
            running_loss += loss
            label_list.append(np.array(labels.to('cpu')).reshape(-1,1))

    y_estimate = y_estimate[~np.all(y_estimate == 0, axis=1)]


    for i in range(test_data_size-23):
        
        error_depth.append(np.abs(label_list[i][0] - y_estimate[i][0])*100)
        error_angle1.append(np.abs(label_list[i][1] - y_estimate[i][1])*57)
        error_angle2.append(np.abs(label_list[i][2] -  y_estimate[i][2])*57)
        error_angle3.append(np.abs(label_list[i][3] -  y_estimate[i][3])*57)


    print(counter, len(label_list),i )
    print("general_loss:", running_loss/counter, print(counter))
    print( "Test {}'lik set üzerinden yapılmıştır ve santimetre-derece cinsinden verilmektedir.".format(counter))
    print("Ortalama Derinlik Hatası:", np.sum(error_depth)/counter)
    print("Ortalama spehrical Açı hataları:", np.sum(error_angle1)/counter,np.sum(error_angle2)/counter)
    print("Ortalama respective yaw açısı hatası:", np.sum(error_angle3)/counter)

    print("Maximum errorlar sırasıyla:", np.max(error_depth), np.max(error_angle1), np.max(error_angle2), np.max(error_angle3))






