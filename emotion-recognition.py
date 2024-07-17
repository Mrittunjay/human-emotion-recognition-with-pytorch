import os
import random
import sys
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from timeit import default_timer as timer


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


# Creating model architecture here:
# class Custom_Emotion_Recognition(nn.Module):
#     def __init__(self):
#         super(Custom_Emotion_Recognition, self).__init__()
#         self.conv1 = nn.Conv2d(1,10,3)
#         self.conv2 = nn.Conv2d(10,10,3)
#         self.pool2 = nn.MaxPool2d(2,2)
#
#         self.conv3 = nn.Conv2d(10,10,3)
#         self.conv4 = nn.Conv2d(10,10,3)
#         self.pool4 = nn.MaxPool2d(2,2)
#
#         self.norm = nn.BatchNorm2d(10)
#
#         self.fc1 = nn.Linear(810,50)
#         self.fc2 = nn.Linear(50,3)      # 3 for three output classes (3 emotions)
#
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         self.fc_loc = nn.Sequential(
#             nn.Linear(640, 32),
#             nn.ReLU(True),
#             nn.Linear(32, 3*2)
#         )
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1,0,0,0,1,0], dtype=torch.float))
#
#     def stn(self, x):
#         xs = self.localization(x)
#         xs = xs.view(-1, 640)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#         # grid = F.affine_grid(theta, x.size())
#         # x = F.grid_sample(x, grid)
#         grid = F.affine_grid(theta, x.size(), align_corners= True)      # added align_corners=True to remove warnings
#         x = F.grid_sample(x, grid, align_corners=True)                  # added align_corners=True to remove warnings
#         return x
#
#     def forward(self, input):
#         out = self.stn(input)
#
#         out = F.relu(self.conv1(out))
#         out = self.conv2(out)
#         out = F.relu(self.pool2(out))
#
#         out = F.relu(self.conv3(out))
#         out = self.norm(self.conv4(out))
#         out = F.relu(self.pool4(out))
#
#         out = F.dropout(out)
#         out = out.view(-1, 810)
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#
#         return out


# class Custom_Emotion_Recognition(nn.Module):
#     def __init__(self):
#         super(Custom_Emotion_Recognition, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3)
#         self.conv2 = nn.Conv2d(32, 32, 3)
#         self.pool2 = nn.MaxPool2d(2, 2)
#
#         self.conv3 = nn.Conv2d(32, 64, 3)
#         self.conv4 = nn.Conv2d(64, 64, 3)
#         self.pool4 = nn.MaxPool2d(2, 2)
#
#         self.norm = nn.BatchNorm2d(64)
#
#         self.fc1 = nn.Linear(64 * 29 * 29, 256)  # Adjusted input size after conv and pooling layers
#         self.fc2 = nn.Linear(256, 3)  # 3 for three output classes
#
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True)
#         )
#
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 28 * 28, 32),  # Adjusted to match the expected input size
#             nn.ReLU(True),
#             nn.Linear(32, 3 * 2)
#         )
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
#
#     def stn(self, x):
#         """
#         # Spatial Transformation Network: Extract features through additional
#         output channels
#         :param x:
#         :return:
#         """
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 28 * 28)  # Adjusted to match the expected input size
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)
#         grid = F.affine_grid(theta, x.size(), align_corners=True)
#         x = F.grid_sample(x, grid, align_corners=True)
#         return x
#
#     def forward(self, input):
#         out = self.stn(input)
#
#         out = F.relu(self.conv1(out))
#         out = self.conv2(out)
#         out = F.relu(self.pool2(out))
#
#         out = F.relu(self.conv3(out))
#         out = self.norm(self.conv4(out))
#         out = F.relu(self.pool4(out))
#
#         out = F.dropout(out)
#         out = out.view(out.size(0), -1)
#         # print(f"Shape before FC1: {out.shape}")
#         out = F.relu(self.fc1(out))
#         out = self.fc2(out)
#
#         return out

class Custom_Emotion_Recognition(nn.Module):
    def __init__(self):
        super(Custom_Emotion_Recognition, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv6 = nn.Conv2d(128, 128, 3)
        self.pool6 = nn.MaxPool2d(2, 2)

        self.norm = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128 * 12 * 12, 256)  # Adjusted input size after conv and pooling layers
        self.fc2 = nn.Linear(256, 3)  # 3 for three output classes

        self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 28 * 28, 32),  # Adjusted to match the expected input size
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 28 * 28)  # Adjusted to match the expected input size
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x

    def forward(self, input):
        out = self.stn(input)

        out = F.relu(self.conv1(out))
        out = F.relu(self.conv2(out))
        out = self.pool2(out)

        out = F.relu(self.conv3(out))
        out = F.relu(self.conv4(out))
        out = self.pool4(out)

        out = F.relu(self.conv5(out))
        out = F.relu(self.conv6(out))
        out = self.norm(self.pool6(out))

        out = F.dropout(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out


# Creating train step function for the training loop
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device):
    # Put the model in train mode
    model.train()

    # Setup training loss and training accuracy
    train_loss, train_acc = 0, 0

    # Loop through dataloader batches
    for batch, (img, label) in enumerate(dataloader):
        # Sending data to target device
        img, label = img.to(device), label.to(device)

        # Forward pass
        y_pred = model(img)    # Output is raw values(raw model logits)

        # Calculating loss:
        loss = loss_fn(y_pred, label)
        train_loss +=loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate accuracy metric
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == label).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy pr batch
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)
    return train_loss, train_acc

# Creating a test step function for training loop
def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device):
    # Put model into evaluation mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference mode
    with torch.inference_mode():
        # Loop through dataloader batches
        for batch, (img, label) in enumerate(dataloader):
            # Putting the data to target device
            img, label = img.to(device), label.to(device)

            # Forward pass
            test_pred_logits = model(img)

            # Calculate the loss
            loss = loss_fn(test_pred_logits, label)
            test_loss += loss.item()

            # Calculate the accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == label).sum().item()/len(test_pred_labels))

        # Adjust the loss and accuracy per batch
        test_loss = test_loss/len(dataloader)
        test_acc = test_acc/len(dataloader)
        return test_loss, test_acc

# Train function:
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          device: str = 'cuda',
          epochs: int = 5):
    # Creating empty result dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}

    try:
        # Looping through training and testing step
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = train_step(
                model=model,
                dataloader=train_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                device=device
            )
            test_loss, test_acc = test_step(
                model=model,
                dataloader=test_dataloader,
                loss_fn=loss_fn,
                device=device
            )

            # Print details
            print(f"\nEpoch: {epoch} | Train loss: {train_loss:0.4f} | Train acc: {train_acc*100:.2f}% | test loss: {test_loss:.4f} | test_acc: {test_acc*100:.2f}%")

            scheduler.step(test_loss)

            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['test_loss'].append(test_loss)
            results['test_acc'].append(test_acc)
        return results
    except KeyboardInterrupt:
        print("\nTraining interrupted. Exiting gracefully . . . ")
        sys.exit(0)


if __name__ == '__main__':
    NUM_EPOCHS = 100
    BATCH_SIZE = 64

    # Setting up training and inference device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Cuda enable status: {device}")

    # Transform for training data
    train_data_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    # Transform for testing data
    test_data_transform = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.ImageFolder(root=TRAIN_DIR,
                                      transform=train_data_transform,
                                      target_transform=None)

    test_data = datasets.ImageFolder(root=TEST_DIR,
                                     transform=test_data_transform,
                                     target_transform=None)
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

    # # Rearranging tensor object 'img' to display image using matplotlib
    # img_permuted = img.permute(1, 2, 0)
    # # plotting the image
    # plt.figure(figsize=(8, 6))
    # plt.imshow(img_permuted)
    # plt.axis('off')
    # plt.title(idx_to_class[label], fontsize=14)
    # plt.show()

    # Creating my dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=BATCH_SIZE,
                                  num_workers=os.cpu_count(),
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=BATCH_SIZE,
                                 num_workers=os.cpu_count(),
                                 shuffle=False)

    print(f"Size of train loader: {len(train_dataloader)}")
    print(f"Size of test loader: {len(test_dataloader)}")

    # Creating model instance
    emotion_model_v1 = Custom_Emotion_Recognition().to(device)

    # Training Loop:
    # Setting loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=emotion_model_v1.parameters(),
                                 lr=0.0005, weight_decay=1e-3)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='min',
                                                           factor=0.1,
                                                           patience=3)

    # Training start time
    start_time = timer()

    # Run training function
    emotion_model_v1_results = train(model=emotion_model_v1,
                                     train_dataloader=train_dataloader,
                                     test_dataloader=test_dataloader,
                                     optimizer=optimizer,
                                     scheduler=scheduler,
                                     loss_fn=loss_fn,
                                     epochs=NUM_EPOCHS,
                                     device=device)

    # Training end time
    end_time = timer()
    print(f"Total training time: {round((((end_time - start_time) / 60) / 60), 3)} hrs")

    # Saving the model
    current_dir = os.getcwd()
    model_file = "emotion_model_v1_state_dict.pth"
    model_save_path = os.path.join(current_dir, model_file)
    torch.save(emotion_model_v1.state_dict(), model_save_path)
# END OF MAIN
