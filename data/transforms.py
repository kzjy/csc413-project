import albumentations as A

transform = A.Compose([
    A.RandomCrop(width=256, height=256),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))

def train_transform(image, bboxes, label):
    transformed = transform(image=image, bboxes=bboxes, class_labels=label)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_class_labels = transformed['class_labels']
    return transformed_image, transformed_bboxes, transformed_class_labels