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


class LPD_step(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.device = device

        ### Primal block of the network
        self.primal_step = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        ### Dual block of the network
        self.dual_step = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3,3), padding=1),
            nn.PReLU(num_parameters=32, init=0),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3,3), padding=1)
            )

        self.to(device)

    ### Must needed forward function
    def forward(self, f, g, h, operator_split, domain, num_of_splits, averaged, device='cuda'):
        
        f_sinograms = torch.zeros((num_of_splits-averaged, g.shape[1]) + (g.shape[2], g.shape[3])).to(device)
        adjoint_eval = torch.zeros((num_of_splits-averaged, f.shape[1]) + (f.shape[2], f.shape[3])).to(device)
        u = torch.zeros((num_of_splits-averaged, g.shape[1]) + (g.shape[2], g.shape[3])).to(device)
        
        for j in range(g.shape[1]):
            operator = operator_split[j]
            operator = OperatorModule(operator).to(device)
            
            f_sinograms[0,:,:] = f_sinograms[0,:,:] + operator(f[:,j,:,:])
        
        # print('h', h.shape)
        # print('f_sino', f_sinograms.shape)
        # print('g', g.shape)
        # print('u', u.shape)
        
        u = torch.cat([h, f_sinograms, g], dim=1)
        h = h + self.dual_step(u)
        
        for k in range(g.shape[1]):
            operator = operator_split[j]
            adjoint_operator = OperatorModule(operator.adjoint).to(device)
            # print('adjoint', adjoint_operator(h[:,k,:,:]).shape)
            adjoint_eval[0,:,:] = adjoint_eval[0,:,:] + adjoint_operator(h[:,k,:,:])
        
        adjoint_eval = adjoint_eval / ((num_of_splits-averaged)**2)
        # print('asjoint_eval', adjoint_eval.shape)
        # print('f', f.shape)
        
        u = torch.cat([f, adjoint_eval], dim=1)
        # print('u2', u.shape)
        f = f + self.primal_step(u)
        
        # ### Dual iterate happens here
        # f_sinogram = self.operator(f) / self.operator_norm
        # u = torch.cat([h, f_sinogram, g / self.operator_norm], dim=1)
        # h = h + self.dual_step(u)

        # ### Primal iterate happens here
        # adjoint_eval = self.adjoint_operator(h) / self.operator_norm
        # u = torch.cat([f, adjoint_eval], dim=1)
        # f = f + self.primal_step(u)
        
        return f, h
        
class LPD(nn.Module):
    def __init__(self, operator, adjoint_operator, operator_norm, n_iter, device='cuda'):
        super().__init__()
        
        self.operator = operator
        self.adjoint_operator = adjoint_operator
        self.operator_norm = operator_norm
        self.n_iter = n_iter
        self.device = device

        ### Initializing the parameters for every unrolled iteration step.
        for k in range(self.n_iter):
            step = LPD_step(operator, adjoint_operator, operator_norm, device=self.device)
            setattr(self, f'step{k}', step)
            
    def forward(self, f, g, operator_split, domain, num_of_splits, averaged, device='cuda'):

        ### Initializing "h" as a zero matrix
        h = torch.zeros(g.shape).to(self.device)

        ### Here happens the unrolled iterations
        for k in range(self.n_iter):
            step = getattr(self, f'step{k}')
            # print('iter', k)
            f, h = step(f, g, h, operator_split, domain, num_of_splits, averaged, device='cuda')
            
        return f















