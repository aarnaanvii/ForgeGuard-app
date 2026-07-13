"""
CNN architecture and fixed SRM (Spatial Rich Model) high-pass filters,
copied and consolidated from the original research repo
(src/cnn/cnn.py + src/cnn/SRM_filters.py).

The SRM filters in the first conv layer are noise-residual filters
borrowed from image forensics: they suppress the actual image content
and amplify local pixel-correlation noise, which tends to differ
between an original camera-captured region and a spliced/edited one.
"""
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torch import Tensor, stack


def get_filters() -> Tensor:
    filters: Dict[str, Tensor] = {}

    # 1st Order
    filters["1O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, -1, 1, 0], [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]))
    filters["1O2"] = Tensor(np.rot90(filters["1O1"]).copy())
    filters["1O3"] = Tensor(np.rot90(filters["1O2"]).copy())
    filters["1O4"] = Tensor(np.rot90(filters["1O3"]).copy())
    filters["1O5"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, -1, 0, 0], [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]))
    filters["1O6"] = Tensor(np.rot90(filters["1O5"]).copy())
    filters["1O7"] = Tensor(np.rot90(filters["1O6"]).copy())
    filters["1O8"] = Tensor(np.rot90(filters["1O7"]).copy())
    # 2nd Order
    filters["2O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]))
    filters["2O2"] = Tensor(np.rot90(filters["2O1"]).copy())
    filters["2O3"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, -2, 0, 0], [0, 0, 0, 1, 0],
                                       [0, 0, 0, 0, 0]]))
    filters["2O4"] = Tensor(np.rot90(filters["2O3"]).copy())
    # 3rd Order
    filters["3O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 1, -3, 1, 0], [0, 0, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]))
    filters["3O2"] = Tensor(np.rot90(filters["3O1"]).copy())
    filters["3O3"] = Tensor(np.rot90(filters["3O2"]).copy())
    filters["3O4"] = Tensor(np.rot90(filters["3O3"]).copy())
    filters["3O5"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, -3, 0, 0], [0, 1, 0, 0, 0],
                                       [0, 0, 0, 0, 0]]))
    filters["3O6"] = Tensor(np.rot90(filters["3O5"]).copy())
    filters["3O7"] = Tensor(np.rot90(filters["3O6"]).copy())
    filters["3O8"] = Tensor(np.rot90(filters["3O7"]).copy())
    # SQUARE 3x3
    filters["S3O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, 1, -2, 1, 0], [0, -2, 4, -2, 0], [0, 1, -2, 1, 0],
                                        [0, 0, 0, 0, 0]]))
    # SQUARE 5x5
    filters["S5O1"] = Tensor(np.array([[-1, 2, -2, 2, -1], [2, -6, 8, -6, 2], [-2, 8, -12, 8, -2],
                                        [2, -6, 8, -6, 2], [-1, 2, -2, 2, -1]]) / 12.0)
    # EDGE 3x3 (4 rotations, using S3O1 rotated variants approximated by itself for remaining slots)
    filters["E3O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, 0, 0, 0, 0],
                                        [0, 0, 0, 0, 0]]))
    filters["E3O2"] = Tensor(np.rot90(filters["E3O1"]).copy())
    filters["E3O3"] = Tensor(np.rot90(filters["E3O2"]).copy())
    filters["E3O4"] = Tensor(np.rot90(filters["E3O3"]).copy())
    # EDGE 5x5 (4 rotations)
    filters["E5O1"] = Tensor(np.array([[0, 0, 0, 0, 0], [0, -1, 2, -1, 0], [0, 2, -4, 2, 0], [0, -1, 2, -1, 0],
                                        [0, 0, 0, 0, 0]]) / 2.0)
    filters["E5O2"] = Tensor(np.rot90(filters["E5O1"]).copy())
    filters["E5O3"] = Tensor(np.rot90(filters["E5O2"]).copy())
    filters["E5O4"] = Tensor(np.rot90(filters["E5O3"]).copy())

    ordered = list(filters.values())[:30]
    # If fewer than 30 distinct filters were defined above, pad by cycling
    while len(ordered) < 30:
        ordered.append(ordered[len(ordered) % len(filters)])

    stacked = stack(ordered)  # [30, 5, 5]
    full = stacked.unsqueeze(1).repeat(1, 3, 1, 1)  # [30, 3, 5, 5]
    return full


class CNN(nn.Module):
    """The convolutional feature-extractor / classifier."""

    def __init__(self):
        super(CNN, self).__init__()

        self.conv0 = nn.Conv2d(3, 3, kernel_size=5, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv0.weight)

        self.conv1 = nn.Conv2d(3, 30, kernel_size=5, stride=2, padding=0)
        self.conv1.weight = nn.Parameter(get_filters())

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(30, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv2.weight)

        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv3.weight)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv4.weight)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv5 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv5.weight)

        self.conv6 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv6.weight)

        self.conv7 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv7.weight)

        self.conv8 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0)
        nn.init.xavier_uniform_(self.conv8.weight)

        self.fc = nn.Linear(16 * 5 * 5, 2)
        self.drop1 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = f.relu(self.conv0(x))
        x = f.relu(self.conv1(x))
        lrn = nn.LocalResponseNorm(3)
        x = lrn(x)
        x = self.pool1(x)
        x = f.relu(self.conv2(x))
        x = f.relu(self.conv3(x))
        x = f.relu(self.conv4(x))
        x = f.relu(self.conv5(x))
        x = lrn(x)
        x = self.pool2(x)
        x = f.relu(self.conv6(x))
        x = f.relu(self.conv7(x))
        x = f.relu(self.conv8(x))
        x = x.view(-1, 16 * 5 * 5)

        if self.training:
            x = f.relu(self.fc(x))
            x = f.softmax(x, dim=1)

        return x
