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
        # self.step_length = nn.Parameter(torch.zeros(1,1,1,1))
        self.step_length = nn.Parameter(torch.tensor(0.1))
        # self.step_length = 0.075
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
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=self.out_channels, \
                                    kernel_size=(3,3), padding=1),
        ]
        
        # self.layers = nn.Sequential(*LGD_layers)
        
        # self.layers = [nn.Sequential(*LGD_layers) for i in range(n_iter)]
        
        self.layers2 = nn.Sequential(*LGD_layers)
        
        self.layers3 = [self.layers2 for i in range(n_iter)]
            
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
        
    def forward(self, f, g_sinograms, operator_split, domain, num_of_splits, average, device='cuda'):
        
        f2 = f.clone().detach()
        # print('frec', f.shape)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(f[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow((f[0,1,:,:]-f[0,0,:,:]).cpu().detach().numpy())
        # plt.show()
        
        
        # print('step length = ', self.step_length)
        # print('g',g_sinograms.shape)
        # print('frec', f[:,0,:,:].shape)
        # f_sinograms = torch.zeros((g_sinograms.shape[1],) + (g_sinograms.shape[2], g_sinograms.shape[3])).to(device)
        adjoint_eval = torch.zeros((1,) + (f.shape[2], f.shape[3])).to(device)
        # f_sinograms = f_sinograms[None,:,:,:]
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(g_sinograms[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(2,2,2)
        # plt.imshow(g_sinograms[0,1,:,:].cpu().detach().numpy())
        # plt.subplot(2,2,3)
        # plt.imshow(g_sinograms[1,0,:,:].cpu().detach().numpy())
        # plt.subplot(2,2,4)
        # plt.imshow(g_sinograms[1,1,:,:].cpu().detach().numpy())
        # plt.show()
        u = torch.zeros((g_sinograms.shape[1], num_of_splits-average) + (f.shape[2], f.shape[3])).to(device)
        df = torch.zeros((g_sinograms.shape[1], num_of_splits-average) + (f.shape[2], f.shape[3])).to(device)
        # print('adj2', adjoint_eval[None,0,:,:].shape)
        for j in range(g_sinograms.shape[1]):
            operator = operator_split[j]
            # adjoint_operator = operator.adjoint
            adjoint_operator = OperatorModule(operator.adjoint).to(device)
            operator = OperatorModule(operator).to(device)
            # f_sinograms[j,:,:] = operator(f[:,j,:,:])
            # print(f_sinograms.device)
            adjoint_eval[0,:,:] = adjoint_eval[0,:,:] + adjoint_operator(operator(f[:,j,:,:]) - g_sinograms[:,j,:,:])
            # adjoint_eval = adjoint_eval[None,:,:,:]
            # f_sinograms[j,:,:] = torch.as_tensor(operator(f[0,j,:,:].cpu().detach().numpy()))
            # adjoint_eval[j,:,:] = torch.as_tensor(adjoint_operator(f_sinograms[j,:,:].cpu().detach().numpy() - g_sinograms[0,j,:,:].cpu().detach().numpy()))
            # u2 = torch.cat([f[:,j,:,:], adjoint_eval[:,j,:,:]], dim=0)
            # plt.figure()
            # plt.subplot(1,2,1)
            # plt.imshow(u2[0,:,:].cpu().detach().numpy())
            # plt.subplot(1,2,2)
            # plt.imshow(u2[1,:,:].cpu().detach().numpy())
            # plt.show()
        
        # print('adjoint', adjoint_eval.shape)
        # print('g', g_sinograms.shape)
        # adjoint_eval = adjoint_eval / ((num_of_splits-1)**2)
        
        # adjoint_eval = adjoint_eval / ((np.shape(g_sinograms)[1])**2)
        
        adjoint_eval = adjoint_eval / ((average)**2)
        
        # plt.figure()
        # plt.imshow(adjoint_eval[0,:,:].cpu().detach().numpy())
        # plt.show()
        # print('adjoint', adjoint_eval.shape)
        
        f = torch.mean(f, dim=1)
        # print('f3', f.shape)
        # print('u', u.shape)
        
        for i in range(self.n_iter):
            u[0,:,:,:] = self.layers3[i].to(device)(torch.cat([f[None,:,:,:], adjoint_eval[None,:,:,:]], dim=1))
            # print('u', u.shape)
        
        # print('u', u.shape)
        u[0,:,:] = torch.mean(u.clone(), dim=0)
        df[0,:,:,:] = -self.step_length * u[0,:,:].clone()
            # print('sinos', f_sinograms.shape)
            # print('adj', adjoint_eval.shape)
        # print('adjoint', adjoint_eval.shape)
        # print('df', df[0:1,0:1,:,:].shape)
        # print('df', df.shape)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(adjoint_eval[0,:,:].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow(adjoint_eval[0,:,:].cpu().detach().numpy())
        # plt.show()
        f = f + df[0,0,:,:]
        
        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(df[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(2,2,2)
        # plt.imshow((df[0,1,:,:]-df[0,0,:,:]).cpu().detach().numpy())
        # plt.subplot(2,2,3)
        # plt.imshow(df[1,0,:,:].cpu().detach().numpy())
        # plt.subplot(2,2,4)
        # plt.imshow(df[1,1,:,:].cpu().detach().numpy())
        # plt.show()
        
        # plt.figure()
        # plt.imshow((u[0,1,:,:]-u[1,0,:,:]).cpu().detach().numpy())
        # plt.show()
        # print('frec3', f.shape)
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(f[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(1,2,2)
        # plt.imshow((torch.mean(f, dim=1)[0,:,:]-f[0,0,:,:]).cpu().detach().numpy())
        # plt.show()
        
        # plt.figure()
        # plt.subplot(1,3,1)
        # plt.imshow(f2[0,0,:,:].cpu().detach().numpy())
        # plt.subplot(1,3,2)
        # plt.imshow((f[0,:,:]).cpu().detach().numpy())
        # plt.subplot(1,3,3)
        # plt.imshow((f2[0,0,:,:]-f[0,:,:]).cpu().detach().numpy())
        # # plt.imshow((f[0,0,:,:]-f2[0,0,:,:]).cpu().detach().numpy())
        # plt.show()
        # # (f[0,1,:,:]-
        
        
        return f[0,:,:], self.step_length














