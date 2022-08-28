import random, os

from PIL import Image, ImageFilter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]



def train_transform():
    transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(), 
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform
    

def val_transform():
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])
    return transform


def get_transform(_type):
    if _type == 'train':
        return train_transform()
    elif _type == 'val':
        return val_transform()


class ImageNetDB(Dataset):
    def __init__(self, data_dir, transform=None):
        self.imgs = []
        self.labels = []
        self.transform = transform
        
        for label in sorted(os.listdir(data_dir)):
            if label.isnumeric():
                label_dir = os.path.join(data_dir, label)
                for img in sorted(os.listdir(label_dir)):
                    if img.endswith('.JPEG'):
                        img_dir = os.path.join(label_dir, img)
                        self.imgs.append(img_dir)
                        self.labels.append(int(label))

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        if self.transform: img = self.transform(img)
            
        label = self.labels[idx]
        return idx, img, label

    def __len__(self):
        return len(self.imgs)