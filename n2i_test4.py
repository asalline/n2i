import odl
import torch

from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
# from n2i_module import geometry_and_ray_trafo, get_images, UNet, data_split
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

import cv2 as cv
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
        final_angle = 1
        angles = odl.uniform_partition(starting_angle, final_angle, 1)
        lines = odl.uniform_partition(-1*np.pi, np.pi, int(512/factor_lines))
        geometry = odl.tomo.FanBeamGeometry(angles, lines, source_radius, detector_radius)
        output_shape = (int(1), int(512/factor_lines))
        
    ray_transform = odl.tomo.RayTransform(domain, geometry, impl=device)

    return domain, geometry, ray_transform, output_shape


def data_split(num_of_images, num_of_splits, shape, averaged, noisy_sinograms, geometry, domain, device, ray_transform):    
    sinogram_split = torch.zeros((num_of_splits, ) + (int(noisy_sinograms.shape[0]/num_of_splits), noisy_sinograms.shape[1]))
    rec_split = torch.zeros((num_of_images, num_of_splits) + (shape))
    print(rec_split.shape)
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
    input_reco = np.zeros((num_of_images, ) + shape)
    target_reco = np.zeros((num_of_images, ) + shape)
    # eval_reco = torch.zeros((averaged, ) + shape)
    print('rec', rec_split.shape)
    for j in range(num_of_images):
        for k in range(averaged):
        # eval_reco[k,:,:] = rec_split[k,:,:]#.cpu().detach().numpy()
            input_reco[j,:,:] = input_reco[j,:,:] + rec_split[j,k,:,:].cpu().detach().numpy()
        
    print('input', input_reco.shape)
    input_reco = input_reco / averaged
    
    if num_of_splits - averaged != 0:
        for j in range(num_of_images):
            for k in range(num_of_splits - averaged):
                target_reco[j,:,:] = target_reco[j,:,:] + rec_split[j,averaged + k,:,:].cpu().detach().numpy()
    
        target_reco = target_reco / (num_of_splits - averaged)
    
    return input_reco, target_reco, rec_split, sinogram_split, operator_split

test_amount = 10
image = get_images('/scratch2/antti/summer2023/test_walnut', test_amount, scale_number=2)
print(np.shape(image))
shape = (np.shape(image)[1], np.shape(image)[2])
images = np.array(image, dtype='float32')
images = torch.from_numpy(images).float().to(device)

domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)

fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

sinograms = ray_transform_module(images)

reco = fbp_operator_module(sinograms)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(sinograms[0,:,:].cpu().detach().numpy())
plt.subplot(1,2,2)
plt.imshow(reco[0,:,:].cpu().detach().numpy())
plt.show()
mean = 0
percentage = 0.05
noisy_sinograms = torch.zeros((test_amount, ) + output_shape)
for k in range(test_amount):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    # mean = 0.05 #* coeff
    # variance = 0.01 #* coeff
    # sigma = variance ** 0.5
    test_sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.std(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
    # noisy_sinogram = sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(sinograms.shape[1], sinograms.shape[2]))

print(noisy_sinograms.shape)
input_reco, target_reco, eval_reco, sinogram_split, operator_split = data_split(test_amount, 4, shape, 1, noisy_sinograms, geometry, domain, device, ray_transform)

print('eval',eval_reco.shape)
print(sinogram_split.shape)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(eval_reco[0,0,:,:].cpu().detach().numpy())
plt.subplot(1,2,2)
plt.imshow(eval_reco[0,1,:,:].cpu().detach().numpy())
plt.show()

plt.figure()
plt.imshow(eval_reco[0,0,:,:].cpu().detach().numpy() - eval_reco[0,1,:,:].cpu().detach().numpy())
plt.show()

print('input', input_reco.shape)
print('target', target_reco.shape)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(input_reco[0,:,:])
plt.subplot(1,2,2)
plt.imshow(target_reco[0,:,:])
plt.show()

plt.figure()
plt.imshow(input_reco[0,:,:] - target_reco[0,:,:])
plt.show()



# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(sinogram_split[0,:,:].cpu().detach().numpy())
# plt.subplot(1,2,2)
# plt.imshow(sinogram_split[1,:,:].cpu().detach().numpy())
# plt.show()



