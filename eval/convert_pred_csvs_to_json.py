import os
import glob
import json
import argparse


def main(dataset_root_path, subset,mode):
    """ Convert csv annotations into a single JSON and save it,
        to match the format with the algorithm submission output.

    Parameters:
    -----------
    dataset_root_path: str
        path to the dataset. (e.g. /home/user/ocelot2023_v0.1.1)
    
    subset: str
        `train` or `val` or `test`.
    """

    #assert os.path.exists(f"{dataset_root_path}/annotations/{subset}")
    #gt_paths = sorted(glob.glob(f"{dataset_root_path}/annotations/{subset}/cell/*.csv"))
    pred_paths = sorted(glob.glob(f"{os.path.join(dataset_root_path,mode,'csv')}/*.csv"))
    num_images = len(pred_paths)
    print("images num:{}".format(num_images))

    pred_json = {
        "type": "Multiple points",
        "num_images": num_images,
        "points": [],
        "version": {
            "major": 1,
            "minor": 0,
        }
    }
    
    for idx, gt_path in enumerate(pred_paths):
        with open(gt_path, "r") as f:
            lines = f.read().splitlines()

        for line in lines:
            x, y, c = line.split(",")
            point = {
                "name": f"image_{idx}",
                "point": [int(x), int(y), int(c)],
                "probability": 1.0,  # dummy value, since it is a GT, not a prediction
            }
            pred_json["points"].append(point)

    with open(os.path.join(dataset_root_path,mode,f'cell_{mode}_{subset}.json'), "w") as g:
        json.dump(pred_json, g)
        print(f"JSON file saved in {os.path.join(dataset_root_path,mode,f'cell_{mode}_{subset}.json')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_root_path", type=str, required=True,
                        help="Path to the pred  csv. ")
    parser.add_argument("-s", "--subset", type=str, required=True, 
                        choices=["train", "val", "test"],
                        help="Which subset among (trn, val, test)?")
    parser.add_argument("-m", "--mode", type=str, required=True, 
                        choices=["predict", "gt"])
    args = parser.parse_args()
    main(args.dataset_root_path, args.subset, args.mode)
    