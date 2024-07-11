import os
import random
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


DATA_PATH = Path("data")
# Setting up train and test paths
TRAIN_DIR = DATA_PATH / "train"   # f"{DATA_PATH}/train"
TEST_DIR = DATA_PATH / "test"   # f"{DATA_PATH}/test"


def walk_through_dir(directory_path):
    """
    Walk through directory path returning its contents
    :param directory_path:
    :return: directory contents
    """
    for dirpath, dirnames, filenames in os.walk(directory_path):
        print(f"There are {len(dirnames)}folders and {len(filenames)} images in {dirpath}")

# walk_through_dir(TRAIN_DIR)

# print(TRAIN_DIR)
# print(TEST_DIR)0

# random.seed(42)

# image_path_list = list(DATA_PATH.glob("*/*/*.jpg"))
# print(image_path_list[:10])

# # Printing random image
# random_image_path = random.choice(image_path_list)
#
# # Getting image class from path name
# image_class = random_image_path.parent.stem
#
# # Open the image using pillow library
# img = Image.open(random_image_path)

# # Print metadata
# print(f"Random image path: {random_image_path}")
# print(f"Image class: {image_class}")
# print(f"Image height: {img.height}")
# print(f"Image width: {img.width}")
# img.show()


print(f"Cuda enable status: {torch.cuda.is_available()}")


# Transform for training data
train_data_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor()
])
# Transform for testing data
test_data_transform = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor()
])

# transformed_image = data_transform(img)
# print(f"Transformed image shape: {transformed_image.shape}\nTransformed image data type{transformed_image.dtype}")

def plot_transformed_images(image_paths: list,
                            transform,
                            n=3,
                            seed=None):
    """
    Selects a random image from the given path, loads/transforms image
    then plots the original vs the transformed images
    """
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(nrows=1, ncols=2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original\nsize: {f.size}")
            ax[0].axis(False)

            # Transform and plot
            # convert from pytorch tensor to matplotlib compatible shape
            transformed_image = train_data_transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed\nshape: {transformed_image.shape}")
            ax[1].axis(False)

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
            plt.show()

# plot_transformed_images(image_paths = image_path_list,
#                         transform=data_transform,
#                         n=3)

# Loading image data using torchvision.datasets.ImageFolder
train_data = datasets.ImageFolder(root=TRAIN_DIR,
                                  transform=train_data_transform,
                                  target_transform=None)

test_data = datasets.ImageFolder(root=TEST_DIR,
                                 transform=test_data_transform,
                                 target_transform=None)

# print("TRAIN DATA: ")
# print(train_data)
# print("\nTEST DATA: ")
# print(test_data)
# print(f"Classes: {train_data.classes}")

# Creating a dictionary to get classnames to integers
# Classes: ['happy', 'neutral', 'sad']
# Class dictionary: {'happy': 0, 'neutral': 1, 'sad': 2}
class_names_dict = train_data.class_to_idx

# {0: 'happy', 1: 'neutral', 2: 'sad'}
idx_to_class = {v: k for k, v in class_names_dict.items()}



# Checking training set image
img, label = train_data[0][0], train_data[0][1]
print(f"Train data image shape: {img.shape}")
print(f"image dtype: {img.dtype}")
print(f"Label datatype {label}: {type(label)}")

# Rearranging tensor object 'img' to display image using matplotlib
img_permuted = img.permute(1, 2, 0)
# plotting the image
plt.figure(figsize=(8, 6))
plt.imshow(img_permuted)
plt.axis('off')
plt.title(idx_to_class[label], fontsize=14)
plt.show()


# Creating my dataloaders
BATCH_SIZE=5
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=BATCH_SIZE,
                              num_workers=os.cpu_count(),
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_data,
                             batch_size=BATCH_SIZE,
                             num_workers=os.cpu_count(),
                             shuffle=False)

