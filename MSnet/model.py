import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Tuple, List


# class MSnet_vocal(nn.Module):
#     def __init__(self):
#         """
#         Initializes the MSnet_vocal model, a convolutional neural network for vocal processing.

#         This model consists of three sequential convolutional layers, each followed by a max pooling layer, a bottom convolutional layer, and then three sequential up-convolutional layers with max un-pooling. The model uses SELU activation and Batch Normalization for each convolutional layer.

#         The architecture is designed to first downsample and then upsample the input, typical in segmentation or dense prediction tasks.
#         """
#         super(MSnet_vocal, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.BatchNorm2d(3), nn.Conv2d(3, 32, 5, padding=2), nn.SELU()
#         )
#         self.pool1 = nn.MaxPool2d((4, 1), return_indices=True)

#         self.conv2 = nn.Sequential(
#             nn.BatchNorm2d(32), nn.Conv2d(32, 64, 5, padding=2), nn.SELU()
#         )
#         self.pool2 = nn.MaxPool2d((4, 1), return_indices=True)

#         self.conv3 = nn.Sequential(
#             nn.BatchNorm2d(64), nn.Conv2d(64, 128, 5, padding=2), nn.SELU()
#         )
#         self.pool3 = nn.MaxPool2d((4, 1), return_indices=True)

#         self.bottom = nn.Sequential(
#             nn.BatchNorm2d(128), nn.Conv2d(128, 1, 5, padding=(0, 2)), nn.SELU()
#         )

#         self.up_pool3 = nn.MaxUnpool2d((4, 1))
#         self.up_conv3 = nn.Sequential(
#             nn.BatchNorm2d(128), nn.Conv2d(128, 64, 5, padding=2), nn.SELU()
#         )

#         self.up_pool2 = nn.MaxUnpool2d((4, 1))
#         self.up_conv2 = nn.Sequential(
#             nn.BatchNorm2d(64), nn.Conv2d(64, 32, 5, padding=2), nn.SELU()
#         )

#         self.up_pool1 = nn.MaxUnpool2d((4, 1))
#         self.up_conv1 = nn.Sequential(
#             nn.BatchNorm2d(32), nn.Conv2d(32, 1, 5, padding=2), nn.SELU()
#         )

#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, x: torch.Tensor) -> tuple:
#         """
#         Defines the forward pass of the MSnet_vocal model.

#         Args:
#         x (torch.Tensor): The input tensor to the network, representing the vocal data.

#         Returns:
#         tuple: A tuple containing two elements. The first element is the output of the network after applying softmax, and the second element is the output from the bottom layer of the network. Both elements are useful for different stages of vocal processing.

#         The forward pass involves sequential operations through convolutional, pooling, un-pooling, and up-convolutional layers, finally applying a softmax function for output.
#         """
#         c1, ind1 = self.pool1(self.conv1(x))
#         c2, ind2 = self.pool2(self.conv2(c1))
#         c3, ind3 = self.pool3(self.conv3(c2))
#         bm = self.bottom(c3)
#         u3 = self.up_conv3(self.up_pool3(c3, ind3))
#         u2 = self.up_conv2(self.up_pool2(u3, ind2))
#         u1 = self.up_conv1(self.up_pool1(u2, ind1))
#         output = self.softmax(torch.cat((bm, u1), dim=2))

#         return output, bm


# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class MSnet_vocal(nn.Module):
#     def __init__(self):
#         super(MSnet_vocal, self).__init__()

#         self.conv1 = nn.Sequential(
#             nn.BatchNorm2d(3), nn.Conv2d(3, 32, 5, padding=2), nn.SELU()
#         )
#         self.pool1 = nn.MaxPool2d((4, 1))

#         self.conv2 = nn.Sequential(
#             nn.BatchNorm2d(32), nn.Conv2d(32, 64, 5, padding=2), nn.SELU()
#         )
#         self.pool2 = nn.MaxPool2d((4, 1))

#         self.conv3 = nn.Sequential(
#             nn.BatchNorm2d(64), nn.Conv2d(64, 128, 5, padding=2), nn.SELU()
#         )
#         self.pool3 = nn.MaxPool2d((4, 1))

#         self.bottom = nn.Sequential(
#             nn.BatchNorm2d(128), nn.Conv2d(128, 1, 5, padding=(0, 2)), nn.SELU()
#         )

#         # Replace MaxUnpool2d with ConvTranspose2d
#         self.up_conv3 = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.ConvTranspose2d(128, 64, (4, 1), stride=(4, 1)),
#             nn.SELU(),
#         )

#         self.up_conv2 = nn.Sequential(
#             nn.BatchNorm2d(64),
#             nn.ConvTranspose2d(64, 32, (4, 1), stride=(4, 1)),
#             nn.SELU(),
#         )

#         self.up_conv1 = nn.Sequential(
#             nn.BatchNorm2d(32),
#             nn.ConvTranspose2d(32, 1, (4, 1), stride=(4, 1)),
#             nn.SELU(),
#         )

#         self.softmax = nn.Softmax(dim=2)

#     def forward(self, x):
#         c1 = self.pool1(self.conv1(x))
#         c2 = self.pool2(self.conv2(c1))
#         c3 = self.pool3(self.conv3(c2))
#         bm = self.bottom(c3)
#         u3 = self.up_conv3(c3)
#         u2 = self.up_conv2(u3)
#         u1 = self.up_conv1(u2)
#         output = self.softmax(torch.cat((bm, u1), dim=2))

#         return output, bm



class MSnet_vocal(nn.Module):
    def __init__(self):
        super(MSnet_vocal, self).__init__()

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 32, 5, padding=2),
            nn.SELU()
            )
        self.pool1 = nn.MaxPool2d((3,1), return_indices=True)

        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.SELU()
            )
        self.pool2 = nn.MaxPool2d((4,1), return_indices=True)

        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 5, padding=2),
            nn.SELU()
            )
        self.pool3 = nn.MaxPool2d((4,1), return_indices=True)

        self.bottom = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 1, (6,5), padding=(0,2)),
            nn.SELU()
            )

        self.up_pool3 = nn.MaxUnpool2d((4,1))
        self.up_conv3 = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, 5, padding=2),
            nn.SELU()
            )

        self.up_pool2 = nn.MaxUnpool2d((4,1))
        self.up_conv2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.SELU()
            )

        self.up_pool1 = nn.MaxUnpool2d((3,1))
        self.up_conv1 = nn.Sequential(
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.SELU()
            )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        c1, ind1 = self.pool1(self.conv1(x))
        c2, ind2 = self.pool2(self.conv2(c1))
        c3, ind3 = self.pool3(self.conv3(c2))

        bm = self.bottom(c3)

        u3 = self.up_conv3(self.up_pool3(c3, ind3))
        u2 = self.up_conv2(self.up_pool2(u3, ind2))
        u1 = self.up_conv1(self.up_pool1(u2, ind1))

        out = torch.cat((bm, u1), dim=2)
        out = torch.squeeze(out,1)
        output = self.softmax(out)

        return output