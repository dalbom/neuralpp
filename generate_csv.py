import os


def generate_csv(
    gt_dir, img_dir, csv_path, inference=False, replace_from=None, replace_to=None
):

    writer = open(csv_path, "w")
    writer.write("image,gt\n")

    filenames = os.listdir(gt_dir)

    if inference:
        for filename in filenames:
            if "png" not in filename:
                continue

            gt_path = os.path.join(gt_dir, filename)
            img_path = os.path.join(img_dir, filename)

            if not os.path.exists(gt_path) or not os.path.exists(img_path):
                continue

            writer.write(img_path + "," + gt_path + "\n")
    else:
        if replace_from is None:
            replace_from = "gt_panoptic.json"

        if replace_to is None:
            replace_to = "rgb.jpg"

        for filename in filenames:
            if "json" not in filename:
                continue

            gt_path = os.path.join(gt_dir, filename)
            img_path = os.path.join(img_dir, filename.replace(replace_from, replace_to))

            if not os.path.exists(gt_path) or not os.path.exists(img_path):
                continue

            writer.write(img_path + "," + gt_path + "\n")

    writer.flush()
    writer.close()


def main():
    # generate_csv("datasets/daytime_gt", "datasets/daytime_img", "datasets/daytime.csv")
    # generate_csv(
    #     "datasets/inference/masks",
    #     "datasets/inference/images",
    #     "datasets/inference.csv",
    #     inference=True,
    # )
    generate_csv(
        "datasets/pedestrian_gt",
        "datasets/pedestrian_img",
        "datasets/pedestrian.csv",
        replace_from="jpg.json",
        replace_to="jpg",
    )


if __name__ == "__main__":
    main()
