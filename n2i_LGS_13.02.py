### Importing packages and modules
import odl
import torch
import torch.nn as nn
import torch.optim as optim
from odl.contrib.torch import OperatorModule
#from odl.contrib import torch as odl_torch
#from torch.nn.utils import clip_grad_norm_
import numpy as np
from n2i_LGS_module import get_images, geometry_and_ray_trafo, LGS, data_split
import matplotlib.pyplot as plt


### Check if nvidia CUDA is available and using it, if not, the CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(device)
# torch.cuda.empty_cache()
train_batch = 2
### Using function "get_images" to import images from the path.
test_amount = 35
test_images = get_images('/scratch2/antti/summer2023/test_walnut', amount_of_images=test_amount, scale_number=2)
### Converting images such that they can be used in calculations
test_images = np.array(test_images, dtype='float32')
test_images = torch.from_numpy(test_images).float().to(device)

### Using functions from "UNet_functions". Taking shape from images to produce
### odl parameters and getting Radon transform operator and its adjoint.
shape = (np.shape(test_images)[1], np.shape(test_images)[2])
domain, geometry, ray_transform, output_shape = geometry_and_ray_trafo(setup='full', shape=shape, device=device, factor_lines = 2)
# print(domain)
# print(geometry)
# print(ray_transform)
# print(output_shape)
fbp_operator = odl.tomo.analytic.filtered_back_projection.fbp_op(ray_transform, padding=1)

### Using odl functions to make odl operators into PyTorch modules
ray_transform_module = OperatorModule(ray_transform).to(device)
adjoint_operator_module = OperatorModule(ray_transform.adjoint).to(device)
fbp_operator_module = OperatorModule(fbp_operator).to(device)

### Making sinograms from the images using Radon transform module
test_sinograms = ray_transform_module(test_images)
test_sinograms = torch.as_tensor(test_sinograms)

### Allocating used tensors
test_noisy_sinograms = torch.zeros((test_sinograms.shape[0], ) + output_shape)
rec_images = torch.zeros((test_sinograms.shape[0], ) + shape)
test_input_reco = torch.zeros((test_sinograms.shape[0], ) + shape)
test_target_reco = torch.zeros((test_sinograms.shape[0], ) + shape)

mean = 0
# variance = 0.005
# sigma = variance ** 0.5
percentage = 0.05
num_of_splits = 4
averaged = 3

all_arrangements = torch.zeros((test_amount, num_of_splits) + shape)

### Adding Gaussian noise to the sinograms. Here some problem solving is
### needed to make this smoother.
for k in range(test_amount):
    #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
    # mean = 0.05 #* coeff
    # variance = 0.01 #* coeff
    # sigma = variance ** 0.5
    test_sinogram_k = test_sinograms[k,:,:].cpu().detach().numpy()
    noise = np.random.normal(mean, test_sinogram_k.max(), test_sinogram_k.shape) * percentage
    test_noisy_sinogram = test_sinogram_k + noise
    test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
    # noisy_sinogram = sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(sinograms.shape[1], sinograms.shape[2]))
    # test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)

test_input_reco, _, _, _, all_arrangements, all_sinograms, operator_split = \
    data_split(test_amount, num_of_splits, shape, averaged, test_noisy_sinograms, geometry, domain, device, ray_transform)

### Using FBP to get reconstructed images from noisy sinograms
rec_images = fbp_operator_module(test_noisy_sinograms)

### All the data into same device
test_sinograms = test_sinograms[:,None,:,:].cpu().detach()
test_noisy_sinograms = test_noisy_sinograms[:,None,:,:].cpu().detach()
rec_images = rec_images[:,None,:,:].cpu().detach()
test_images = test_images[:,None,:,:].cpu().detach()
test_input_reco = test_input_reco[:,None,:,:].cpu().detach()
test_target_reco = test_target_reco[:,None,:,:].cpu().detach()

print('arrang',all_arrangements.shape)
print('sinos',all_sinograms.shape)
print(test_input_reco.shape)
print(test_target_reco.shape)
print('operator', operator_split)

fbp_operator2 = odl.tomo.analytic.filtered_back_projection.fbp_op(operator_split[1], padding=1)

plt.figure()
plt.subplot(1,3,1)
plt.imshow(all_arrangements[4,0,:,:].cpu().detach().numpy())
plt.subplot(1,3,2)
plt.imshow(all_sinograms[4,0,:,:].cpu().detach().numpy())
plt.subplot(1,3,3)
plt.imshow(all_arrangements[4,0,:,:].cpu().detach().numpy()-fbp_operator2(all_sinograms[4,0,:,:]))
plt.show()

### Setting UNet as model and passing it to the used device
LGS_net = LGS(adjoint_operator_module, in_channels=2, \
              out_channels=1, step_length=0.005, n_iter=5).to(device)

# print(LGS_net.device)

### Getting model parameters
LGS_parameters = list(LGS_net.parameters())

### Defining PSNR function.
def psnr(loss):
    
    psnr = 10 * np.log10(1.0 / loss+1e-10)
    
    return psnr


loss_train = nn.MSELoss()
loss_test = nn.MSELoss()

# fbp_reco = fbp_operator(noisy_sinograms[50,0,:,:].cpu().detach().numpy())
# plt.figure()
# plt.imshow(fbp_reco)
# plt.show()
# print(psnr(loss_test(rec_images[50,0,:,:], images[50,0,:,:]).cpu().detach().numpy())**0.5)
# # print('fbp_reco_shape', fbp_reco.size())
# # print('image size', images[50,0,:,:].cpu().detach().numpy().size())
# # print(psnr(loss_test(fbp_operator(noisy_sinograms[50,0,:,:].cpu().detach().numpy()), \
# #                      images[50,0,:,:].cpu().detach().numpy())))

### Defining evaluation (test) function
def eval(net, g, f):

    test_loss = []
    
    ### Setting network into evaluation mode
    net.eval()
    test_loss.append(torch.sqrt(loss_test(net(g), f)).item())
    print(test_loss)
    out3 = net(g[0,None,:,:])

    return out3

### Setting up some lists used later
running_loss = []
running_test_loss = []
torch.autograd.set_detect_anomaly(True)
### Defining training scheme
def train_network(net, n_train=300, batch_size=25): #g_train, g_test, f_train, f_test, 

    ### Defining optimizer, ADAM is used here
    optimizer = optim.Adam(LGS_parameters, lr=0.001) #betas = (0.9, 0.99)
    
    ### Definign scheduler, can be used if wanted
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train)
    
    ### Setting network into training mode
    
    net.train()
    ### Here starts the itarting in training
    for i in range(n_train):
        # print(i)
        net.train()
        if i % train_batch == 0:
            images = get_images('/scratch2/antti/summer2023/usable_walnuts', amount_of_images=train_batch, scale_number=2)
            ### Converting images such that they can be used in calculations
            images = np.array(images, dtype='float32')
            images = torch.from_numpy(images).float().to(device)
            sinograms = ray_transform_module(images)
            sinograms = torch.as_tensor(sinograms)

            ### Allocating used tensors
            noisy_sinograms = torch.zeros((sinograms.shape[0], ) + output_shape)
            rec_images = torch.zeros((sinograms.shape[0], ) + shape)
            input_reco = torch.zeros((sinograms.shape[0], ) + shape)
            target_reco = torch.zeros((sinograms.shape[0], ) + shape)

            # averaged = 1
            ### Adding Gaussian noise to the sinograms. Here some problem solving is
            ### needed to make this smoother.
            for k in range(train_batch):
                #coeff = np.max(np.max(sinograms[k,:,:].cpu().detach().numpy()))
                # mean = 0.05 #* coeff
                # variance = 0.01 #* coeff
                # sigma = variance ** 0.5
                sinogram_k = sinograms[k,:,:].cpu().detach().numpy()
                noise = np.random.normal(mean, sinogram_k.max(), sinogram_k.shape) * percentage
                noisy_sinogram = sinogram_k + noise
                noisy_sinograms[k,:,:] = torch.as_tensor(noisy_sinogram)
                # noisy_sinogram = sinograms[k,:,:].cpu().detach().numpy() + np.random.normal(mean, sigma, size=(sinograms.shape[1], sinograms.shape[2]))
                # test_noisy_sinograms[k,:,:] = torch.as_tensor(test_noisy_sinogram)
                
            input_reco, target_reco, sinogram_reco, target_sinogram_reco, rec_split, sinogram_split2, operator_split, \
                = data_split(train_batch, num_of_splits, shape, averaged, noisy_sinograms, geometry, domain, device, ray_transform)
                

            ### Using FBP to get reconstructed images from noisy sinograms
            rec_images = fbp_operator_module(noisy_sinograms)

            ### All the data into same device
            sinograms = sinograms[:,None,:,:].cpu().detach()
            noisy_sinograms = noisy_sinograms[:,None,:,:].cpu().detach()
            rec_images = rec_images[:,None,:,:].cpu().detach()
            images = images[:,None,:,:].cpu().detach()
            input_reco = input_reco[:,None,:,:].cpu().detach()
            target_reco = target_reco[:,None,:,:].cpu().detach()
            sinogram_reco = sinogram_reco[:,None,:,:]
            target_sinogram_reco = target_sinogram_reco[:,None,:,:]
            # print('rec', rec_split.shape)
            # sinogram_split2 = sinogram_split2 [:,None,:,:]
        
        ### Taking batch size amount of data pieces from the random 
        ### permutation of all training data
        # print('input', input_reco.dtype)
        # print('target', target_reco.shape)
        # print('split2', sinogram_split2.shape)
        n_index = np.random.permutation(input_reco.shape[0])[:batch_size]
        g_batch = sinogram_split2[n_index,0:num_of_splits-averaged,:,:].to(device)
        f_batch2 = rec_split[n_index,0:num_of_splits-averaged,:,:].to(device)
        f_batch = target_reco[n_index,:,:,:].to(device)
        
        # print('g_batch', g_batch.shape)
        # net.train()
        # optimizer.zero_grad()
        ### Taking out some enhanced images
        # print('g_batch', g_batch.shape)
        # print('f_batch2', f_batch2.shape)
        outs, step_length2 = net(f_batch2.float(), g_batch.float(), operator_split, domain, num_of_splits, averaged)

        
        # plt.figure()
        # plt.imshow(outs[0,0,:,:].cpu().detach().numpy()-f_batch[0,0,:,:].cpu().detach().numpy())
        # plt.show()
        optimizer.zero_grad()
        
        
        ### Setting gradient to zero
        # optimizer.zero_grad()
        
        ### Calculating loss of the outputs
        loss = loss_train(outs[None,None,:,:], f_batch.float())

        # loss = torch.from_numpy(loss)

        #loss.requires_grad=True
        
        ### Calculating gradient
        loss.backward()
        
        ### Here gradient clipping could be used
        # torch.nn.utils.clip_grad_norm_(UNet_parameters, max_norm=1.0, norm_type=2)
        
        ### Taking optimizer step
        optimizer.step()
        scheduler.step()
        
        #running_loss.append(loss.item())

        ### Here starts the running tests
        if i % 100 == 0:
            
            net.eval()
            with torch.no_grad():
                
                # outs2 = torch.zeros((test_amount, num_of_splits) + shape).to(device)
                # outs3 = torch.zeros((test_amount, ) + shape).to(device)
                outs4 = torch.zeros((test_amount, num_of_splits) + shape).to(device)
                
                for k in range(test_amount):
                    for j in range(num_of_splits):
                    # print('hmm2', torch.swapaxes(all_arrangements[[k],:,:,:], 0,1).shape)
                        # print('hmm2', type(all_arrangements[[k],j,:,:][:,None,:,:].to(device)))
                        # print('hmm3', type(all_sinograms[[k],j,:,:][:,None,:,:].to(device)))
                        outs4[k,j,:,:], _ = net(all_arrangements[[k],j,:,:][:,None,:,:].to(device), all_sinograms[[k],j,:,:][:,None,:,:].to(device), \
                                           operator_split, domain, num_of_splits, averaged)
                    # print('outs3', outs3.shape)
                    # outs4[k,:,:] = torch.mean(outs3, dim=0)
                
                outs4 = torch.mean(outs4, dim=1)
                
                ### Calculating test loss with test data outputs
                test_loss = loss_test(outs4[:,None,:,:], test_images.to(device)).item()**0.5
                train_loss = loss.item() ** 0.5
                running_loss.append(train_loss)
                running_test_loss.append(test_loss)
                # print(f'step length = {step_length2}')
            
            ### Printing some data out
            if i % 1000 == 0:
                print(f'Iter {i}/{n_train} Train Loss: {train_loss:.2e}, Test Loss: {test_loss:.2e}, PSNR: {psnr(test_loss**2):.2f}') #, end='\r'
                plt.figure()
                plt.subplot(1,2,1)
                plt.imshow(outs4[4,:,:].cpu().detach().numpy())
                plt.subplot(1,2,2)
                plt.imshow(test_images[4,0,:,:].cpu().detach().numpy())
                plt.show()

    ### After iterating taking one reconstructed image and its ground truth
    ### and showing them
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(outs[0,0,:,:].cpu().detach().numpy())
    plt.subplot(1,2,2)
    plt.imshow(f_batch[0,0,:,:].cpu().detach().numpy())
    plt.show()

    ### Plotting running loss and running test loss
    plt.figure()
    plt.semilogy(running_loss)
    plt.semilogy(running_test_loss)
    plt.show()

    return running_loss, running_test_loss, net

### Calling training function to start the naural network training
running_loss, running_test_loss, ready_to_eval = train_network(LGS_net, n_train=30001, batch_size=1) # g_train, g_test, f_train, f_test


