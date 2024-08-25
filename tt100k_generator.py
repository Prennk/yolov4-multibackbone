import json
import os

data_dir = "TT100K/data"
annotatons_file = os.path.join(data_dir, "annotations.json")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

classes = ['pn', 'pne', 'i5', 'p11', 'pl40', 'po', 'pl50', 'pl80', 'io', 'pl60', 
           'p26', 'i4', 'pl100', 'pl30', 'pl5', 'il60', 'i2', 'p5', 'w57', 'p10', 
           'ip', 'pl120', 'il80', 'p23', 'pr40', 'w59', 'ph4.5', 'p12', 'p3', 'w55', 
           'pm20', 'pl20', 'pg', 'pl70', 'pm55', 'il100', 'p27', 'w13', 'p19', 'ph4', 
           'ph5', 'wo', 'p6', 'pm30', 'w32']

def create_txt(annotations_file, train_dir, test_dir, classes):
    with open(annotations_file, "r") as f:
        annotations = json.load(f)

    def create_file(output_files, image_list, annotations):
        with open(output_files, "w") as f:
            for image_id in image_list:
                image_info = annotations["imgs"][image_id]
                image_path = os.path.join(data_dir, image_info["path"])
                line = image_path

                for obj in image_info["objects"]:
                    category = obj["category"]
                    if category in classes:
                        bbox = obj["bbox"]
                        xmin = int(bbox["xmin"])
                        ymin = int(bbox["ymin"])
                        xmax = int(bbox["xmax"])
                        ymax = int(bbox["ymax"])
                        line += f" {xmin},{ymin},{xmax},{ymax},{classes.index(category)}"
                        
                print(line)
                f.write(line + "\n")

    train_images = open(data_dir + "/train/ids.txt").read().splitlines()
    print(f"train images count: {len(train_images)}")
    test_images = open(data_dir + "/test/ids.txt").read().splitlines()
    print(f"test images count: {len(test_images)}")

    create_file("train.txt", train_images, annotations)
    create_file("test.txt", test_images, annotations)

create_txt(annotatons_file, train_dir, test_dir, classes)