"""
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import linalg
import pickle
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import Deep_KSVD

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device is: {device}")

# Overcomplete Discrete Cosinus Transform:
patch_size = 8
m = 16
Dict_init = Deep_KSVD.init_dct(patch_size, m)
Dict_init = Dict_init.to(device)

# Squared Spectral norm:
c_init = linalg.norm(Dict_init.cpu(), ord=2) ** 2
c_init = torch.FloatTensor((c_init,))
c_init = c_init.to(device)

# Average weight:
w_init = torch.normal(mean=1, std=1 / 10 * torch.ones(patch_size ** 2)).float()
w_init = w_init.to(device)

# Deep-KSVD:
D_in, H_1, H_2, H_3, D_out_lam, T, min_v, max_v = patch_size ** 2, 128, 64, 32, 1, 7, -1, 1
model = Deep_KSVD.DenoisingNet_MLP(
    patch_size,
    D_in,
    H_1,
    H_2,
    H_3,
    D_out_lam,
    T,
    min_v,
    max_v,
    Dict_init,
    c_init,
    w_init,
    device,
)

model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.to(device)

# Test image names:
file_test = open("test_gray.txt", "r")
onlyfiles_test = []
for e in file_test:
    onlyfiles_test.append(e[:-1])

# Rescaling in [-1, 1]:
mean = 255 / 2
std = 255 / 2
data_transform = transforms.Compose(
    [Deep_KSVD.Normalize(mean=mean, std=std), Deep_KSVD.ToTensor()]
)
# Noise level:
sigma = 25

# Test Dataset:
my_Data_test = Deep_KSVD.FullImagesDataset(
    root_dir="gray", image_names=onlyfiles_test, sigma=sigma, transform=data_transform
)

dataloader_test = DataLoader(my_Data_test, batch_size=1, shuffle=False, num_workers=0)

# List PSNR:
file_to_print = open("list_test_PSNR.csv", "w")
file_to_print.write(str(device) + "\n")
file_to_print.flush()

with open("list_test_PSNR.txt", "wb") as fp:
    with torch.no_grad():
        list_PSNR = []
        list_PSNR_init = []
        PSNR = 0
        for k, (image_true, image_noise) in enumerate(dataloader_test, 0):
            image_true_t = image_true[0, 0, :, :]
            image_true_t = image_true_t.to(device)

            image_noise_0 = image_noise[0, 0, :, :]
            image_noise_0 = image_noise_0.to(device)

            image_noise_t = image_noise.to(device)
            image_restored_t = model(image_noise_t)
            image_restored_t = image_restored_t[0, 0, :, :]

            PSNR_init = 10 * torch.log10(
                4 / torch.mean((image_true_t - image_noise_0) ** 2)
            )
            file_to_print.write("Init:" + " " + str(PSNR_init) + "\n")
            file_to_print.flush()

            list_PSNR_init.append(PSNR_init)

            PSNR = 10 * torch.log10(
                4 / torch.mean((image_true_t - image_restored_t) ** 2)
            )
            PSNR = PSNR.cpu()
            file_to_print.write("Test:" + " " + str(PSNR) + "\n")
            file_to_print.flush()

            list_PSNR.append(PSNR)

            # mpimg.imsave("results/im_noisy_"+str(k)+'.pdf',image_noise_0)
            # mpimg.imsave("results/im_restored_"+str(k)+'.pdf',image_restored_t)

            # Create a figure with appropriate size
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 10))

            # Plot image1 on the first subplot
            axes[0].imshow(image_noise_0.cpu(), cmap='gray')
            axes[0].axis('off')
            axes[0].set_title('im_noisy_' + str(k))

            # Plot image2 on the second subplot
            axes[1].imshow(image_restored_t.cpu(), cmap='gray')
            axes[1].axis('off')
            axes[1].set_title('im_restored_' + str(k))

            # Adjust layout to avoid overlap
            plt.tight_layout()

            # Save the figure as a single image
            plt.savefig(f'results\\combined_images_{k}.png', bbox_inches='tight')

            # Display the result if needed
            # plt.show()
            plt.close()


    mean = np.mean(list_PSNR)
    file_to_print.write("FINAL" + " " + str(mean) + "\n")
    file_to_print.flush()
    pickle.dump(list_PSNR, fp)
