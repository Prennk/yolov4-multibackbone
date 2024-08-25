import glob
import json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:, 0] * cluster[:, 1]
    iou = intersection / (area1 + area2 - intersection)

    return iou

def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])

def kmeans(box, k):
    row = box.shape[0]
    distance = np.empty((row, k))
    last_clu = np.zeros((row, ))

    np.random.seed()
    cluster = box[np.random.choice(row, k, replace=False)]

    iter = 0
    while True:
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)
        
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break
        
        for j in range(k):
            cluster[j] = np.median(box[near == j], axis=0)

        last_clu = near
        if iter % 5 == 0:
            print('iter: {:d}. avg_iou:{:.2f}'.format(iter, avg_iou(box, cluster)))
        iter += 1

    return cluster, near

def load_data(path):
    data_dir = "TT100K/data"
    data = []
    image_width = 2048
    image_height = 2048

    with open(path, "r") as f:
        annotations = json.load(f)

    image_list = [img for img in annotations["imgs"] if annotations["imgs"][img]["path"].startswith("train")]

    for image_id in image_list:
        image_info = annotations["imgs"][image_id]
        image_path = os.path.join(data_dir, image_info["path"])

        for obj in image_info["objects"]:
            bbox = obj["bbox"]
            xmin = bbox["xmin"]
            ymin = bbox["ymin"]
            xmax = bbox["xmax"]
            ymax = bbox["ymax"]
        
            xmin /= image_width
            ymin /= image_height
            xmax /= image_width
            ymax /= image_height
            
            data.append([xmax - xmin, ymax - ymin])
    
    return np.array(data)

if __name__ == '__main__':
    np.random.seed(0)
    input_shape = [416, 416]
    anchors_num = 9
    
    path = 'TT100K/data/annotations.json'
    
    print('Load JSON annotations.')
    data = load_data(path)
    print('Load JSON annotations done.')
    
    print('K-means boxes.')
    cluster, near = kmeans(data, anchors_num)
    print('K-means boxes done.')
    data = data * np.array([input_shape[1], input_shape[0]])
    cluster = cluster * np.array([input_shape[1], input_shape[0]])

    for j in range(anchors_num):
        plt.scatter(data[near == j][:, 0], data[near == j][:, 1])
        plt.scatter(cluster[j][0], cluster[j][1], marker='x', c='black')
    plt.savefig("kmeans_for_anchors.jpg")
    plt.show()
    print('Save kmeans_for_anchors.jpg in root dir.')

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_iou(data, cluster)))
    print(cluster)

    with open("tt100k_anchors.txt", 'w') as f:
        for i in range(cluster.shape[0]):
            if i == 0:
                x_y = "%d,%d" % (cluster[i][0], cluster[i][1])
            else:
                x_y = ", %d,%d" % (cluster[i][0], cluster[i][1])
            f.write(x_y)
