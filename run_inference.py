from inference_aqc import YOLACTEdgeInference
import os
import multiprocessing as mp
from multiprocessing.pool import ThreadPool
from time import time
from pathlib import Path
from tqdm import tqdm
import cv2 as cv
import numpy as np
import json

class Run(object):
    def __init__(self, weights: str, model_config: str, dataset: str, calib_images: str or None, args: str) -> None:
        self.weights = weights
        self.model_config= model_config
        self.dataset = dataset
        self.calib_images = calib_images
        self.args = args

    def model_init(self) -> None:
        self.model = YOLACTEdgeInference(self.weights, self.model_config, self.dataset, self.calib_images)

def post_process_output(res):
    json_file = []
    for i in range(len(res)):
        aux_dict = {}
        aux_dict["image_id"] = os.path.basename(str(res[i][0]))
        aux_dict["annotations"] = []
        if len(res[i][-1]) > 0:
            for j in range(len(res[i][1])):
                mask = np.array(res[i][-1][j]*255.).astype(np.uint8)
                contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                area = -99999
                for cnt in contours:
                    area_cnt = cv.contourArea(cnt)
                    if area_cnt > area:
                        area = area_cnt
                        best_contour = cnt

                epsilon = 0.01*cv.arcLength(best_contour,True)
                approx = cv.approxPolyDP(best_contour,epsilon,True)
                # Bonding box
                x,y,w,h = cv.boundingRect(best_contour)
                aux_dict["annotations"].append({
                    "category_id": res[i][1][j],
                    "score": res[i][2][j],
                    "bbox": [x, y, w, h],
                    "segmentation": approx.ravel().tolist()
                })
        json_file.append(aux_dict)
    return json_file

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
        

def main(multithread=False):
    args = {"score_threshold": 0.15}
    weights = "./yolact_edge/weights/yolact_plus_resnet50_qualitex_custom_2_121_115000.pth"
    config = "yolact_resnet50_qualitex_custom_2_config"
    dataset = "qualitex_dataset"
    calib_images = None
    score = args["score_threshold"]

    # Creating inference object
    inference = Run(weights=weights, model_config=config, dataset=dataset, calib_images=calib_images, args= args)

    # Initializing inference
    inference.model_init()

    # Launch a prediction
    input_path = "./eval_4k"
    paths = list(Path(input_path).glob('*'))
    n_workers = 8 #len(paths)//2

    if multithread:
        paths = [[str(p), score] for p in paths]
        print("Launching inference on batch")
        start_time = time()
        try:
            with ThreadPool(n_workers) as pool:
                    res = pool.starmap(inference.model.predict_simple, paths)
        except:
            raise
            return 
        stop_time = time()
        total_time = stop_time - start_time
        print(f"Predictions done in {total_time:.2f} seconds.")
    else:
        print("Launching inference on batch")
        res = []
        start_time = time()
        for path in tqdm(paths):
            res.append(inference.model.predict_simple(str(path), score))
        stop_time = time()
        total_time = stop_time - start_time
        print(f"Predictions done in {total_time:.2f} seconds.")

    # Post process
    start_time = time()
    res = post_process_output(res)
    output_path = "./results/output.json"
    with open(output_path, "w") as j:
        json.dump(res, j, cls=NpEncoder)
    stop_time = time()
    total_time = stop_time - start_time
    print(f"Postprocess done in {total_time:.2f} seconds.")


if __name__ == '__main__':
    main(multithread=True)