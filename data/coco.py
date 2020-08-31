import os
import numpy as np
import torch.utils.data as data
from pycocotools.coco import COCO
from PIL import Image


class COCODataset(data.Dataset):
    def __init__(self, root_dir, set_name, transform=None):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(
            os.path.join(self.root_dir, 'annotations',
                         'instances_' + self.set_name + '.json'))

        self.img_ids = self.coco.getImgIds()

        self.load_classes()

    def __getitem__(self, idx):

        img, path = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return {
            'img': sample['img'],
            'annot': sample['annot'],
            'path': path,
            'size': (img.size)  # W, H
        }

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.img_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name,
                            image_info['file_name'])
        # print(path)
        img = Image.open(path).convert('RGB')
        return img, path

    def load_annotations(self, image_index):
        # get ground truth annotations
        annt_ids = self.coco.getAnnIds(imgIds=self.img_ids[image_index],
                                       iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annt_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annt_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def __len__(self, ):
        return len(self.img_ids)

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def num_classes(self):
        return 80


if __name__ == "__main__":
    pass