server = "PC"


def get_path_dict_ImageDehazing():

    if server == "PC":
        path_dict = {
            "OTS":
                {
                    "train": "E:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/OTS/",
                    "val": "E:/CXF_Code/dataset/processed_dataset/dehazing_dataset/RESIDE/SOTS/outdoor/"
                },

            "4KDehazing":
                {
                    "train": "E:/CXF_Code/dataset/processed_dataset/dehazing_dataset/4KDehazing/train/",
                    "val": "E:/CXF_Code/dataset/processed_dataset/dehazing_dataset/4KDehazing/val/"
                },
        }

    else:
        path_dict = None
    return path_dict