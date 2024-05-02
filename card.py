import torchvision.transforms as transforms
""" Stanford Cars (Car) Dataset
Created: Nov 15,2019 - Yuchong Gu
Revised: Nov 15,2019 - Yuchong Gu
"""
import os
import pdb
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
#from utils import get_transform

def get_transform(resize, phase='train'):
    if phase == 'train':
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.RandomCrop(resize),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(brightness=0.126, saturation=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=(int(resize[0] / 0.875), int(resize[1] / 0.875))),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

class CarDataset(Dataset):
    """
    # Description:
        Dataset for retrieving Stanford Cars images and labels

    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image

        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset

        __len__(self):                  returns the length of dataset
    """

    def __init__(self, root, phase='train', resize=500):
        assert phase in ['train', 'val', 'test']
        self.root = root
        self.phase = phase
        self.resize = resize
        self.num_classes = 196

        # self.class_names = [''] * self.num_classes
        # with open(os.path.join(self.root, 'classes.txt')) as f:
        #     lines = f.readlines()
        #     for line in lines:
        #         line = line.strip()
        #         cls_idx, cls_name = line.split(' ')
        #         cls_name = cls_name.split('.')[-1]
        #         self.class_names[cls_idx] = cls_name

        if phase == 'train':
            list_path = os.path.join(self.root, 'devkit', 'cars_train_annos.mat')
            self.image_path = os.path.join(self.root, 'cars_train')
        else:
            list_path = os.path.join(self.root, 'cars_test_annos_withlabels.mat')
            self.image_path = os.path.join(self.root, 'cars_test')

        list_mat = loadmat(list_path)
        self.images = [f.item() for f in list_mat['annotations']['fname'][0]]
        self.labels = [f.item() for f in list_mat['annotations']['class'][0]]

        meta_data = loadmat(os.path.join(self.root, 'devkit', 'cars_meta.mat'))
        class_names = list(meta_data['class_names'])[0]
        self.class_names = [str(name[0]) for name in class_names]

        # transform
        self.transform = get_transform(self.resize, self.phase)

    def __getitem__(self, item):
        # image
        img_path = os.path.join(self.image_path, self.images[item])
        image = Image.open(img_path).convert('RGB')  # (C, H, W)
        image = self.transform(image)

        # return image and label
        return image, self.labels[item] - 1, img_path  # count begin from zero

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    ds = CarDataset('val')
    # print(len(ds))
    for i in range(0, 100):
        image, label = ds[i]
        # print(image.shape, label)
