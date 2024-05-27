import os

GT_PATH = "datasets/daytime_gt"
IMG_PATH = "datasets/daytime_img"

writer = open("datasets/daytime.csv", "w")
writer.write("image,gt\n")

filenames = os.listdir(GT_PATH)

for filename in filenames:
    if "json" not in filename:
        continue

    gt_path = os.path.join(GT_PATH, filename)
    img_path = os.path.join(IMG_PATH, filename.replace("gt_panoptic.json", "rgb.jpg"))

    if not os.path.exists(gt_path) or not os.path.exists(img_path):
        continue

    writer.write(img_path + "," + gt_path + "\n")

writer.flush()
writer.close()
