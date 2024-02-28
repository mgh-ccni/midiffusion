import glob
import itertools
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import numpy as np
from imgaug import augmenters as iaa
import os
import skimage.io as skio
import skimage

class MultimodalDataset(Dataset):
    def __init__(self, pathA, pathB, logA=False, logB=False, transform=None):
        self.transform = transform

        if not isinstance(pathA, list):
            pathA = [pathA]
        if not isinstance(pathB, list):
            pathB = [pathB]
        self.pathA = pathA
        self.pathB = pathB
        self.filenamesA = [glob.glob(path) for path in pathA]
        self.filenamesA = list(itertools.chain(*self.filenamesA))
        self.filenamesB = [glob.glob(path) for path in pathB]
        self.filenamesB = list(itertools.chain(*self.filenamesB))

        self.channels = [None, None]

        filename_index_pairs = filenames_to_dict(self.filenamesA, self.filenamesB)

        filenames = [self.filenamesA, self.filenamesB]
        log_flags = [logA, logB]

        dataset = {}
        for mod_ind in range(2):
            # Read all files from modality
            for filename, inds in filename_index_pairs.items():
                pathname = filenames[mod_ind][inds[mod_ind]]

                filename = os.path.basename(pathname)

                if filename not in dataset.keys():
                    dataset[filename] = [None, None]

                img = skio.imread(pathname)
                img = skimage.img_as_float(img)

                if log_flags[mod_ind]:
                    img = np.log(1. + img)

                if img.ndim == 2:
                    img = img[..., np.newaxis]

                if self.channels[mod_ind] is None:
                    self.channels[mod_ind] = img.shape[2]

                dataset[filename][mod_ind] = img

        self.images = []
        for image_set in dataset:
            try:
                self.images.append(
                    np.block([
                        dataset[image_set][0],
                        dataset[image_set][1]
                    ]).astype(np.float32)
                )
            except ValueError:
                print(
                    f"Failed concatenating set {image_set}. Shapes are {dataset[image_set][0].shape} and {dataset[image_set][1].shape}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, augment=True):
        if augment and self.transform:
            return self.transform(self.images[idx])
        return self.images[idx]

class ImgAugTransformNMRI:
    def __init__(self, config, testing=False):
        if not testing:
            self.aug = iaa.Sequential([
                iaa.Fliplr(0.5),
                #iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 2.0))),
                    #iaa.CenterCropToFixedSize(config.data.image_size, config.data.image_size),
                    iaa.Resize({"height": config.data.image_size, "width": config.data.image_size})
            ])
        else:
            self.aug = iaa.Sequential([
                #iaa.CropToFixedSize(config.data.image_size, config.data.image_size),
                iaa.Resize({"height": config.data.image_size, "width": config.data.image_size}),
            ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class ImgAugTransform:
    def __init__(self, testing=False):
        if not testing:
            self.aug = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 2.0))),
                iaa.CenterCropToFixedSize(128, 128),
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128, 128),
            ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img)

class ImgAugTransformNormalized:
    def __init__(self, testing=False):
        if not testing:
            self.aug = iaa.Sequential([
                iaa.Fliplr(0.5),
                iaa.Affine(rotate=(-180, 180), order=[0, 1, 3], mode="symmetric"),
                iaa.Sometimes(
                    0.5,
                    iaa.GaussianBlur(sigma=(0, 2.0))),
                iaa.CenterCropToFixedSize(128, 128),
            ])
        else:
            self.aug = iaa.Sequential([
                iaa.CropToFixedSize(128, 128),
            ])

    def __call__(self, img):
        img = (np.array(img) - np.array(img).min()) / (np.array(img).max() - np.array(img).min())
        return self.aug.augment_image(img)

def filenames_to_dict(filenamesA, filenamesB):
    d = {}
    for i in range(len(filenamesA)):
        basename = os.path.basename(filenamesA[i])
        d[basename] = (i, None)
    for i in range(len(filenamesB)):
        basename = os.path.basename(filenamesB[i])
        # filter out files only in B
        if basename in d:
            d[basename] = (d[basename][0], i)

    # filter out files only in A
    d = {k: v for k, v in d.items() if v[1] is not None}
    return d
