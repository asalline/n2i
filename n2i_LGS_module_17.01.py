import odl
from odl.contrib.torch import OperatorModule
import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt


def get_images(path, amount_of_images='all', scale_number=1):

    all_images = []
    all_image_names = os.listdir(path)
    # print(len(all_image_names))
    if amount_of_images == 'all':
        for name in all_image_names:
            # print(path + '\\' + name)
            temp_image = cv.imread(path + '//' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[80:420, 80:420]
            image = image[0:340:scale_number, 0:340:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)
    else:
        temp_indexing = np.random.permutation(len(all_image_names))[:amount_of_images]
        images_to_take = [all_image_names[i] for i in temp_indexing]
        for name in images_to_take:
            temp_image = cv.imread(path + '/' + name, cv.IMREAD_UNCHANGED)
            image = temp_image[90:410, 90:410]
            image = image[0:320:scale_number, 0:320:scale_number]
            image = image / 0.07584485627272729
            all_images.append(image)

    return all_images


def geometry_and_ray_trafo(setup='full', min_domain_corner=[-1,-1], max_domain_corner=[1,1], \
                           shape=(100,100), source_radius=2, detector_radius=1, \
                           dtype='float32', device='cpu', factor_lines = 1):

    device = 'astra_' + device
    # print(device)
    domain = odl.uniform_discr(min_domain_corner, max_domain_corner, shape, dtype=dtype)

    if setup == 'full':
        angles = odl.uniform_partition(0, 2*np.pi, 1024)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(1028/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (1024, int(1028/factor_lines))
    elif setup == 'sparse':
        angle_measurements = 100
        line_measurements = int(512/factor_lines)
        angles = odl.uniform_partition(0, 2*np.pi, angle_measurements)
        lines = odl.uniform_partition(-1*np.pi, np.pi, line_measurements)
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (angle_measurements, line_measurements)
    elif setup == 'limited':
        starting_angle = 0
        final_angle = np.pi * 3/4
        angles = odl.uniform_partition(starting_angle, final_angle, 360)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(360), int(512/factor_lines))
        
    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

    return domain, geometry, ray_transform, output_shape

def data_split(num_of_images, num_of_splits, shape, averaged, noisy_sinograms, geometry, domain, device, ray_transform):    
    sinogram_split = torch.zeros((num_of_images, num_of_splits) + (int(noisy_sinograms.shape[1]/num_of_splits), noisy_sinograms.shape[2]))
    rec_split = torch.zeros((num_of_images, num_of_splits) + (shape))
    operator_split = []
    # # print(rec_split.shape)
    # for k in range(num_of_images):
    #     print(k)
    #     for j in range(num_of_splits):
    #         split = geometry[j::num_of_splits]
    #         # noisy_sinogram = noisy_sinogram[j,:,:]
    #         sinogram_split = noisy_sinograms[:, j::num_of_splits, :]
    #         operator_split = odl.tomo.RayTransform(domain, split, impl='astra_' + device)
    #         # print(f'during {j}',operator_split.range)
    #         split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
    #         split_FBP_module = OperatorModule(split_FBP)
    #         # reco = split_FBP_module(sinogram_split)
    #         # print('reco',reco.shape)
    #         # rec_split = split_FBP_module(sinogram_split)
    #         rec_split[k,j,:,:] = split_FBP_module(sinogram_split)[k,:,:]
    #         # print(type(split_FBP))
    #         # FBP_domain = split_FBP.domain
    #         # print(split_FBP.domain)
    #         # split_FBP = split_FBP.asarray()
    #         # sinogram_split[j,:,:] = domain.element(sinogram_split[j,:,:])
    #         # print(type(sinogram_split[j,:,:]))
    #         # rec_split[j,:,:] = split_FBP(sinogram_split[j,:,:])
            
    for j in range(num_of_splits):
        split = geometry[j::num_of_splits]
        operator_split.append(odl.tomo.RayTransform(domain, split, impl='astra_' + device))
        # print('ray', operator_split)
        # print(f'during {j}',operator_split.range)
        split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split[-1], padding=1)
        split_FBP_module = OperatorModule(split_FBP)
        
        for k in range(num_of_images):
            # print(k)
            if k == 0:
                # print(np.shape(noisy_sinograms[k, j::num_of_splits, :]))
                # print(np.shape(sinogram_split))
                sinogram_split[k,j,:,:] = noisy_sinograms[k, j::num_of_splits, :][:,:]
                to_FBP = noisy_sinograms[:, j::num_of_splits, :]
                rec_split[k,j,:,:] = split_FBP_module(to_FBP)[k,:,:]
            else:
                sinogram_split[k,j,:,:] = noisy_sinograms[k, j::num_of_splits, :][:,:]
                to_FBP = noisy_sinograms[:, j::num_of_splits, :]
                rec_split[k,j,:,:] = split_FBP_module(to_FBP)[k,:,:]
    
    # print('after',operator_split.range)
    # split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
    # # print('asd')
    # fbp_operator_module = OperatorModule(split_FBP).to('cuda')
    # # print('fbp',split_FBP.range)
    # # split_reco = torch.zeros((num_of_splits, ) + (shape))
    # rec_split = fbp_operator_module(sinogram_split)
    # print('rec split', rec_split.shape)
    # print('noisy',np.shape(noisy_sinograms))
    input_reco = np.zeros((num_of_images, ) + shape)#.astype('float32')
    target_reco = np.zeros((num_of_images, ) + shape)#.astype('float32'))
    sinogram_reco = np.zeros((num_of_images, ) + (int(np.shape(noisy_sinograms)[1]/num_of_splits), np.shape(noisy_sinograms)[2]))
    target_sinogram_reco = np.zeros((num_of_images, ) + (int(np.shape(noisy_sinograms)[1]/num_of_splits), np.shape(noisy_sinograms)[2]))
    # print('reco',np.shape(sinogram_reco))
    # eval_reco = torch.zeros((averaged, ) + shape)
    # print('rec', rec_split.shape)
    for j in range(num_of_images):
        for k in range(averaged):
        # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
            input_reco[j,:,:] = input_reco[j,:,:] + rec_split[j,k,:,:].cpu().detach().numpy()
            sinogram_reco[j,:,:] = sinogram_reco[j,:,:] + noisy_sinograms[j, k::num_of_splits, :].cpu().detach().numpy()
            
    # print('input', input_reco.shape)
    input_reco = input_reco / averaged
    sinogram_reco = sinogram_reco / averaged
    
    if num_of_splits - averaged != 0:
        for j in range(num_of_images):
            for k in range(num_of_splits - averaged):
                target_reco[j,:,:] = target_reco[j,:,:] + rec_split[j,averaged + k,:,:].cpu().detach().numpy()
                target_sinogram_reco[j,:,:] = target_sinogram_reco[j,:,:] + noisy_sinograms[j, (averaged + k)::num_of_splits, :].cpu().detach().numpy()
    
        target_reco = target_reco / (num_of_splits - averaged)
        target_sinogram_reco = target_sinogram_reco / (num_of_splits - averaged)

# torch.as_tensor(input_reco), torch.as_tensor(target_reco)    
# input_reco, target_reco
    # print('sinogram split',np.shape(sinogram_split))
    return torch.as_tensor(input_reco), torch.as_tensor(target_reco), torch.as_tensor(sinogram_reco), \
        torch.as_tensor(target_sinogram_reco), rec_split, sinogram_split, operator_split


# def data_split(num_of_splits, shape, averaged, test_amount, noisy_sinograms, geometry, domain, device, ray_transform):    
#     sinogram_split = torch.zeros((num_of_splits, ) + (int(noisy_sinograms.shape[0]/num_of_splits), noisy_sinograms.shape[1]))
#     rec_split = torch.zeros((num_of_splits, ) + (shape))
#     for j in range(num_of_splits):
#         split = geometry[j::num_of_splits]
#         # noisy_sinogram = noisy_sinogram[j,:,:]
#         sinogram_split[j,:,:] = noisy_sinograms[j::num_of_splits, :]
#         operator_split = odl.tomo.RayTransform(domain, split, impl='astra_' + device)
#         # split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
#         # split_FBP_module = OperatorModule(split_FBP)
#         # rec_split[j,:,:] = split_FBP_module(sinogram_split[j,:,:])
#         # print(type(split_FBP))
#         # FBP_domain = split_FBP.domain
#         # print(split_FBP.domain)
#         # split_FBP = split_FBP.asarray()
#         # sinogram_split[j,:,:] = domain.element(sinogram_split[j,:,:])
#         # print(type(sinogram_split[j,:,:]))
#         # rec_split[j,:,:] = split_FBP(sinogram_split[j,:,:])
    
#     # print('split shape',sinogram_split.shape)
#     split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
#     # print('asd')
#     fbp_operator_module = OperatorModule(split_FBP).to('cuda')
#     # print('fbp',split_FBP.range)
#     # split_reco = torch.zeros((num_of_splits, ) + (shape))
#     rec_split = fbp_operator_module(sinogram_split)
#     # print('rec split', rec_split.shape)
#     input_reco = np.zeros(shape)
#     target_reco = np.zeros(shape)
#     # eval_reco = torch.zeros((averaged, ) + shape)
#     for k in range(averaged):
#         # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
#         input_reco = input_reco + rec_split[k,:,:].cpu().detach().numpy()
        
#     input_reco = input_reco / averaged
    
#     for k in range(num_of_splits - averaged):
#         target_reco = target_reco + rec_split[averaged + k,:,:].cpu().detach().numpy()
    
#     target_reco = target_reco / (num_of_splits - averaged)
    
#     return input_reco, target_reco, sinogram_split, operator_split


class LGS(nn.Module):
    def __init__(self, adjoint_operator_module,
                 in_channels, out_channels, step_length, n_iter, device='cuda'):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.step_length = step_length
        self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.n_iter = n_iter
        self.device = device
        # self.operator = operator_module
        # self.gradient_of_f = adjoint_operator_module
        
        ### tried seven convs!
        LGD_layers = [
            nn.Conv2d(in_channels=self.in_channels, \
                                    out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
                                    kernel_size=(3,3), padding=1),
        ]
        
        # self.layers = nn.Sequential(*LGD_layers)
        
        self.layers = [nn.Sequential(*LGD_layers) for i in range(n_iter)]
        
        # self.layers2 = nn.Sequential(*LGD_layers)
        
        # self.layers3 = [self.layers2 for i in range(n_iter)]
            
        # self.conv1 = nn.Conv2d(in_channels=self.in_channels, \
        #                         out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu2 = nn.ReLU()
        # self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu3 = nn.ReLU()
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1)
        # self.relu4 = nn.ReLU()
        # self.conv5 = nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
        #                         kernel_size=(3,3), padding=1)
        # self.relu5 = nn.ReLU()
        
    def forward(self, f_rec_images, g_sinograms, operator_split, domain, device='cuda'):
        
        # print('step length = ', self.step_length)
        for i in range(self.n_iter):
            print('f_rec', f_rec_images.shape)
            print('g sinos', g_sinograms.shape)
            # print('oper', operator_split.shape)
            f_sinogram = torch.zeros((g_sinograms.shape[1], ) + (g_sinograms.shape[2], g_sinograms.shape[3]))
            f_sinogram = f_sinogram[None,:,:,:]
            adjoint_eval = torch.zeros((g_sinograms.shape[1], ) + (f_rec_images.shape[2], f_rec_images.shape[3]))
            if g_sinograms.shape[1] == 2:
                for j in range(g_sinograms.shape[1]):
                    operator = operator_split[j]
                    adjoint_operator = OperatorModule(operator.adjoint).to(device)
                    operator = OperatorModule(operator_split[j]).to(device)
                    f_sinogram[:,j,:,:] = operator(f_rec_images[:,j,:,:])
                    adjoint_eval[j,:,:] = adjoint_operator(f_sinogram[:,j,:,:].to(device) - g_sinograms[0,j,:,:].to(device))
            else:
                for j in range(g_sinograms.shape[1]):
                    operator = operator_split[j]
                    adjoint_operator = OperatorModule(operator.adjoint).to(device)
                    operator = OperatorModule(operator_split[j]).to(device)
                    f_sinogram[:,j,:,:] = operator(f_rec_images[0,j,:,:])
                    adjoint_eval[j,:,:] = adjoint_operator(f_sinogram[:,j,:,:].to(device) - g_sinograms[0,j,:,:].to(device))
            
            # print('f sino', f_sinogram.shape)
            
            # f_rec_images = torch.as_tensor(torch.mean(f_rec_images, dim=0)).to(device)
            f_rec_images = f_rec_images.to(device)
            # print('f_rec', f_rec_images.shape)
            # print('adjoint', adjoint_eval.shape)
            # adjoint_final = torch.as_tensor(torch.mean(adjoint_eval, dim=0)).to(device)
            adjoint_final = torch.as_tensor(adjoint_eval[None,:,:]).to(device)
            
            
            # operator = operator_split[-1]
            # adjoint_operator = OperatorModule(operator.adjoint).to(device)
            # operator = OperatorModule(operator_split[-1]).to(device)
            # # print('f-rec', np.shape(f_rec_images))
            # # print('ray', operator.domain)
            # f_sinogram = operator(f_rec_images)
            # plt.figure()
            # plt.imshow(f_sinogram[0,0,:,:].cpu().detach().numpy())
            # plt.show()
            # print('f_sino', f_sinogram.device)
            # print('g_sino', g_sinograms.device)
            # dual layer
            # grad_f = adjoint_operator(f_sinogram - g_sinograms) # (output of dual - g_sinograms)
            print('f_rec', f_rec_images.shape)
            print('adjoint', adjoint_final.shape)
            u = torch.zeros((2, g_sinograms.shape[1]) + (adjoint_final.shape[2], adjoint_final.shape[3])).to(device)
            df = torch.zeros((2, g_sinograms.shape[1]) + (adjoint_final.shape[2], adjoint_final.shape[3])).to(device)
            print('df2', df.shape)
            print('u', u.shape)
            print('concat', torch.cat([f_rec_images[:,0,:,:], adjoint_final[:,0,:,:]], dim=0).shape)
            for jj in range(u.shape[0]):
                u[jj,:,:,:] = torch.cat([f_rec_images[:,jj,:,:], adjoint_final[:,jj,:,:]], dim=0).clone()
                print('u123', u[jj,:,:,:].shape)
                u[jj,:,:,:] = self.layers[i].to(device)(u[jj,:,:,:]).clone()
                print('u', u.shape)
                df[jj,:,:,:] = -self.step_length * u[jj,:,:,:]
                print('df2', df.shape)
                print('frec2', f_rec_images.shape)
                # f_rec_images[:,jj,:,:] = f_rec_images[:,jj,:,:].clone() + df[jj,0,:,:]
                
            
            # u = torch.cat([f_rec_images, adjoint_final], dim=1)#.to('cpu')
            # print('u', u.shape)
            # print('u', u.device)
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(f_rec_images[0,0,:,:].cpu().detach().numpy())
            # plt.subplot(1,3,2)
            # plt.imshow(u[0,1,:,:].cpu().detach().numpy()-u[1,1,:,:].cpu().detach().numpy())
            # plt.subplot(1,3,3)
            # plt.imshow(u[1,0,:,:].cpu().detach().numpy()-u[1,1,:,:].cpu().detach().numpy())
            # plt.show()
            #primal layer
            # u = u.to(device)
            
            # u = self.layers(u)
            
            # print(u.shape)
            # for k in range(u.shape[1]):
            #     u[:,k,:,:] = self.layers[i].to(device)(u[k,:,:,:])
                
            # u = self.layers[i].to(device)(u)
            # df = torch.zeros((2, g_sinograms.shape[1]) + (adjoint_final.shape[2], adjoint_final.shape[3]))
            # df = -self.step_length * u[:,:,:,:]
            # print('df', df.shape)
            # # df = torch.as_tensor(torch.mean(df, dim=1))
            # # df = torch.mean(df, dim=0)
            # # print('df', df.shape)
            # f_rec_images = f_rec_images + df
            
            # plt.figure()
            # plt.subplot(1,3,1)
            # plt.imshow(f_rec_images[0,0,:,:].cpu().detach().numpy())
            # plt.subplot(1,3,2)
            # plt.imshow(df[0,0,:,:].cpu().detach().numpy())
            # plt.subplot(1,3,3)
            # plt.imshow(u[0,0,:,:].cpu().detach().numpy())
            # plt.show()
        f_rec_images = f_rec_images.clone() + df[:,0,:,:]
        print('frec3', f_rec_images.shape)
        f_rec_images = torch.mean(f_rec_images, dim=1)
        print('frec4', f_rec_images.shape)
        
        return f_rec_images, self.step_length


