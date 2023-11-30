import odl
from odl.contrib.torch import OperatorModule
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os


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
    sinogram_split = torch.zeros((num_of_splits, ) + (int(noisy_sinograms.shape[0]/num_of_splits), noisy_sinograms.shape[1]))
    rec_split = torch.zeros((num_of_images, num_of_splits) + (shape))
    # print(rec_split.shape)
    for k in range(num_of_images):
        for j in range(num_of_splits):
            split = geometry[j::num_of_splits]
            # noisy_sinogram = noisy_sinogram[j,:,:]
            sinogram_split = noisy_sinograms[:, j::num_of_splits, :]
            operator_split = odl.tomo.RayTransform(domain, split, impl='astra_' + device)
            # print(f'during {j}',operator_split.range)
            split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
            split_FBP_module = OperatorModule(split_FBP)
            # reco = split_FBP_module(sinogram_split)
            # print('reco',reco.shape)
            # rec_split = split_FBP_module(sinogram_split)
            rec_split[k,j,:,:] = split_FBP_module(sinogram_split)[k,:,:]
            # print(type(split_FBP))
            # FBP_domain = split_FBP.domain
            # print(split_FBP.domain)
            # split_FBP = split_FBP.asarray()
            # sinogram_split[j,:,:] = domain.element(sinogram_split[j,:,:])
            # print(type(sinogram_split[j,:,:]))
            # rec_split[j,:,:] = split_FBP(sinogram_split[j,:,:])
    
    # print('after',operator_split.range)
    # split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
    # # print('asd')
    # fbp_operator_module = OperatorModule(split_FBP).to('cuda')
    # # print('fbp',split_FBP.range)
    # # split_reco = torch.zeros((num_of_splits, ) + (shape))
    # rec_split = fbp_operator_module(sinogram_split)
    # print('rec split', rec_split.shape)
    input_reco = np.zeros((num_of_images, ) + shape)#.astype('float32')
    target_reco = np.zeros((num_of_images, ) + shape)#.astype('float32'))
    # eval_reco = torch.zeros((averaged, ) + shape)
    # print('rec', rec_split.shape)
    for j in range(num_of_images):
        for k in range(averaged):
        # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
            input_reco[j,:,:] = input_reco[j,:,:] + rec_split[j,k,:,:].cpu().detach().numpy()
        
    # print('input', input_reco.shape)
    input_reco = input_reco / averaged
    
    if num_of_splits - averaged != 0:
        for j in range(num_of_images):
            for k in range(num_of_splits - averaged):
                target_reco[j,:,:] = target_reco[j,:,:] + rec_split[j,averaged + k,:,:].cpu().detach().numpy()
    
        target_reco = target_reco / (num_of_splits - averaged)

# torch.as_tensor(input_reco), torch.as_tensor(target_reco)    
# input_reco, target_reco
    return torch.as_tensor(input_reco), torch.as_tensor(target_reco), rec_split, sinogram_split, operator_split


# def data_split(num_of_splits, shape, averaged, noisy_sinograms, geometry, domain, device, ray_transform):    
#     sinogram_split = torch.zeros((num_of_splits, ) + (int(noisy_sinograms.shape[0]/num_of_splits), noisy_sinograms.shape[1]))
#     rec_split = torch.zeros((num_of_splits, ) + (shape))
#     for j in range(num_of_splits):
#         split = geometry[j::num_of_splits]
#         # noisy_sinogram = noisy_sinogram[j,:,:]
#         sinogram_split[j,:,:] = noisy_sinograms[j::num_of_splits, :]
#         operator_split = odl.tomo.RayTransform(domain, split, impl='astra_' + device)
#         split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
#         # split_FBP_module = OperatorModule(split_FBP)
#         rec_split[j,:,:] = torch.as_tensor(split_FBP(sinogram_split[j,:,:]))
#         # print(type(split_FBP))
#         # FBP_domain = split_FBP.domain
#         # print(split_FBP.domain)
#         # split_FBP = split_FBP.asarray()
#         # sinogram_split[j,:,:] = domain.element(sinogram_split[j,:,:])
#         # print(type(sinogram_split[j,:,:]))
#         # rec_split[j,:,:] = split_FBP(sinogram_split[j,:,:])
    
#     # print('split shape',sinogram_split.shape)
#     # split_FBP = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split, padding=1)
#     # print('asd')
#     # fbp_operator_module = OperatorModule(split_FBP).to('cuda')
#     # print('fbp',split_FBP.range)
#     # split_reco = torch.zeros((num_of_splits, ) + (shape))
#     # rec_split = fbp_operator_module(sinogram_split)
    
#     # print('rec split', rec_split.shape)
#     input_reco = np.zeros(shape)
#     target_reco = np.zeros(shape)
#     # eval_reco = torch.zeros((averaged, ) + shape)
#     for k in range(averaged):
#         # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
#         input_reco = input_reco + rec_split[k,:,:].cpu().detach().numpy()
        
#     input_reco = input_reco / averaged
    
#     if num_of_splits - averaged != 0:
#         for k in range(num_of_splits - averaged):
#             target_reco = target_reco + rec_split[averaged + k,:,:].cpu().detach().numpy()
        
#         target_reco = target_reco / (num_of_splits - averaged)
    
#     return torch.as_tensor(input_reco), torch.as_tensor(target_reco), rec_split, sinogram_split, operator_split


def double_conv_and_ReLU(in_channels, out_channels):
    list_of_operations = [
        nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), padding=1),
        nn.ReLU()
    ]

    return nn.Sequential(*list_of_operations)


class encoding(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        ### Defining instance variables
        self.in_channels = in_channels

        self.convs_and_relus1 = double_conv_and_ReLU(self.in_channels, out_channels=32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=32, out_channels=64)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus3 = double_conv_and_ReLU(in_channels=64, out_channels=128)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus4 = double_conv_and_ReLU(in_channels=128, out_channels=256)
        self.maxpool4 = nn.MaxPool2d(kernel_size=(2,2))
        self.convs_and_relus5 = double_conv_and_ReLU(in_channels=256, out_channels=512)

    ### Must have forward function. Follows skip connecting UNet architechture
    def forward(self, g):
        g_start = g
        encoding_features = []
        g = self.convs_and_relus1(g)
        encoding_features.append(g)
        g = self.maxpool1(g)
        g = self.convs_and_relus2(g)
        encoding_features.append(g)
        g = self.maxpool2(g)
        g = self.convs_and_relus3(g)
        encoding_features.append(g)
        g = self.maxpool3(g)
        g = self.convs_and_relus4(g)
        encoding_features.append(g)
        g = self.maxpool4(g)
        g = self.convs_and_relus5(g)

        return g, encoding_features, g_start

### Class for decoding part of the UNet. This is the part of the UNet which
### goes back up with transpose of the convolution
class decoding(nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        ### Defining instance variables
        self.out_channels = out_channels

        self.transpose1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus1 = double_conv_and_ReLU(in_channels=512, out_channels=256)
        self.transpose2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus2 = double_conv_and_ReLU(in_channels=256, out_channels=128)
        self.transpose3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus3 = double_conv_and_ReLU(in_channels=128, out_channels=64)
        self.transpose4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2,2), stride=2, padding=0)
        self.convs_and_relus4 = double_conv_and_ReLU(in_channels=64, out_channels=32)
        self.final_conv = nn.Conv2d(in_channels=32, out_channels=self.out_channels, kernel_size=(3,3), padding=1)


    ### Must have forward function. Follows skip connecting UNet architechture
    def forward(self, g, encoding_features, g_start):
        g = self.transpose1(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus1(g)
        g = self.transpose2(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus2(g)
        g = self.transpose3(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus3(g)
        g = self.transpose4(g)
        g = torch.cat([g, encoding_features[-1]], dim=1)
        encoding_features.pop()
        g = self.convs_and_relus4(g)
        g = self.final_conv(g)

        g = g_start + g

        return g

### Class for the UNet model itself
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        ### Defining instance variables
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoder = encoding(self.in_channels)
        self.decoder = decoding(self.out_channels)

    ### Must have forward function. Calling encoder and deoder classes here
    ### and making the whole UNet model
    def forward(self, g):
        
        g, encoding_features, g_start = self.encoder(g)
        g = self.decoder(g, encoding_features, g_start)

        return g
    

