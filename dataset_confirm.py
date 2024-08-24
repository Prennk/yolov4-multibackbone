import torch
import matplotlib.pyplot as plt
import random
from PIL import Image
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def load_annotations(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    annotations = []
    for line in lines:
        parts = line.strip().split(' ')
        img_path = parts[0]
        boxes = []
        for annotation in parts[1:]:
            # Split setiap anotasi dengan koma
            annotation_parts = annotation.split(',')
            xmin = float(annotation_parts[0])
            ymin = float(annotation_parts[1])
            xmax = float(annotation_parts[2])
            ymax = float(annotation_parts[3])
            class_id = int(annotation_parts[4])
            boxes.append((xmin, ymin, xmax, ymax, class_id))

        annotations.append((img_path, boxes))
        
    return annotations

train_file = "train.txt"
test_file = "test.txt"

train_annotations = load_annotations(train_file)
test_annotations = load_annotations(test_file)

print(f'Jumlah data train: {len(train_annotations)}')
print(f'Jumlah data test: {len(test_annotations)}')

def show_random_image(annotations, classes):
    random_image = random.choice(annotations)
    img_path, boxes = random_image
    img = Image.open(img_path)
    plt.figure(figsize=(15, 15))
    plt.imshow(img)
    ax = plt.gca()
    for box in boxes:
        xmin, ymin, xmax, ymax, class_id = box
        rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
        plt.text(xmin, ymin, classes[class_id], bbox=dict(facecolor='yellow', alpha=0.5))
    plt.show()

classes = ['pn', 'pne', 'i5', 'p11', 'pl40', 'po', 'pl50', 'pl80', 'io', 'pl60', 
           'p26', 'i4', 'pl100', 'pl30', 'pl5', 'il60', 'i2', 'p5', 'w57', 'p10', 
           'ip', 'pl120', 'il80', 'p23', 'pr40', 'w59', 'ph4.5', 'p12', 'p3', 'w55', 
           'pm20', 'pl20', 'pg', 'pl70', 'pm55', 'il100', 'p27', 'w13', 'p19', 'ph4', 
           'ph5', 'wo', 'p6', 'pm30', 'w32']
show_random_image(train_annotations, classes)

def plot_category_distribution(annotations, classes):
    category_counts = Counter()
    for _, boxes in annotations:
        for _, _, _, _, class_id in boxes:
            category_counts[class_id] += 1

    categories = list(category_counts.keys())
    counts = list(category_counts.values())

    plt.figure(figsize=(15, 15))
    plt.bar(categories, counts)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=60)
    plt.title('Category Distribution')
    plt.show()

plot_category_distribution(train_annotations, classes)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, annotations, transform=None):
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path, boxes = self.annotations[idx]
        img = Image.open(img_path).convert('RGB')  # Pastikan gambar dalam format RGB

        if self.transform:
            img = self.transform(img)

        return img, boxes

transformations = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.ToTensor()
])

train_dataset = CustomDataset(train_annotations, transform=transformations)
test_dataset = CustomDataset(test_annotations, transform=transformations)

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

print(f'Jumlah train_dataloader: {len(train_dataloader)}')
print(f'Jumlah test_dataloader: {len(test_dataloader)}')

for imgs, boxes in train_dataloader:
    print(f'Shape gambar batch pertama: {[img.shape for img in imgs]}')
    break