import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
from sklearn.feature_extraction.text import TfidfVectorizer

from torch.utils.data import Dataset


class ImagesDataset(Dataset):
    RESCALE_SIZE = 100

    def __init__(self, images, descriptions, targets, test_data=False, path="./Flickr8k_Dataset/Flicker8k_Dataset/"):
        super().__init__()
        self.images = images
        self.descriptions = descriptions
        self.path = path
        self.test_data = test_data
        self.tfid = TfidfVectorizer()

        if not test_data:
            self.descriptions = self.tfid.fit_transform(self.descriptions).toarray()
            self.targets = targets
        else:
            self.descriptions = self.tfid.transform(self.descriptions).toarray()
            self.targets = None

    def __read_file(self, index):
        image = cv2.imread(self.path + self.images[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        desc = self.descriptions[index]

        image = cv2.resize(image, (self.RESCALE_SIZE, self.RESCALE_SIZE))

        return image.astype(np.float32), desc.astype(np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        if not self.test_data:
            img, desc = self.__read_file(index)
            target = self.targets[index]
            return (transform(img), torch.tensor(desc)), torch.tensor(target)

        img, desc = self.__read_file(index)
        return transform(img), torch.tensor(desc)
