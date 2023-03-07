import json
import multiprocessing as mp
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import sleep, time
from typing import List

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from letxbe.provider import Provider
from letxbe.provider.type import LabelPrediction, Prediction, Task
from letxbe.session import LXBSession
from pydantic import ValidationError
from tqdm import tqdm

from inference_aqc import YOLACTEdgeInference
from utils.crop_images import crop_image
from utils.preprocess import clahe

# TODO Change to .env file - Docker secrets
AQC_CLIENT_ID = ""
AQC_CLIENT_SECRET = ""
BASE_URL = "https://staging-unfold.onogone.com"

AQC_LXB = LXBSession(AQC_CLIENT_ID, AQC_CLIENT_SECRET, BASE_URL)
PROVIDER = Provider(AQC_LXB, "aqc")


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


class Run(object):
    def __init__(self, weights: str, model_config: str, dataset: str, calib_images: str or None, args: str) -> None:
        self.weights = weights
        self.model_config = model_config
        self.dataset = dataset
        self.calib_images = calib_images
        self.args = args

    def model_init(self) -> None:
        self.model = YOLACTEdgeInference(self.weights, self.model_config, self.dataset, self.calib_images)


def preprocess_crop(images, preprocess):
    save_path = ".tmp/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    im_name = images[0][0][:-4]
    image_bytes = images[0][1]
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    im = cv.imdecode(file_bytes, cv.IMREAD_COLOR)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    if preprocess:
        im = clahe(im)
    crop_image(im, save_path, im_name)
    return


def postprocess_output(res):
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

                epsilon = 0.01*cv.arcLength(best_contour, True)
                approx = cv.approxPolyDP(best_contour, epsilon, True)
                # Bonding box
                x, y, w, h = cv.boundingRect(best_contour)
                aux_dict["annotations"].append({
                    "category_id": res[i][1][j],
                    "score": res[i][2][j],
                    "bbox": [x, y, w, h],
                    "segmentation": approx.ravel().tolist()
                })
        json_file.append(aux_dict)
    return json_file


def merge_masks(mask, cropped_mask, x, y, size=512):
    mask[y*size:(y+1)*size, x*size:(x+1)*size] = cropped_mask
    return mask


def extract_contour_and_bbox(mask, apply=False, width=4096, height=1024):  # TODO What to do with weird masks??
    contour = []
    bbox = []
    mask = np.array(mask*255).astype(np.uint8)
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if apply:
        max_area = -9999
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area > max_area:
                best_contour = cnt
        epsilon = 0.01*cv.arcLength(best_contour, True)
        approx = cv.approxPolyDP(best_contour, epsilon, True)
        approx = [tuple([x[0][0]/width*100, x[0][1]/height*100]) for x in approx]
        # Bonding box
        x, y, w, h = cv.boundingRect(best_contour)
        contour.append(approx)
        bbox.append([x, y, w, h])
    else:
        for cnt in contours:
            epsilon = 0.01*cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)
            approx = [tuple([x[0][0]/width*100, x[0][1]/height*100]) for x in approx]
            # Bonding box
            x, y, w, h = cv.boundingRect(cnt)
            contour.append(approx)
            bbox.append([x, y, w, h])
    return contour, bbox


def get_iou(a, b, epsilon=1e-5):
    """
    Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0]+a[2], b[0]+b[2])
    y2 = min(a[1]+a[3], b[1]+b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width < 0) or (height < 0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2]) * (a[3])
    area_b = (b[2]) * (b[3])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


def postprocess_masks(res):
    categories = list(range(0, 13))
    image_names = np.unique(["_".join(os.path.basename(x[0]).split("_")[:-2]) for x in res])
    post_result = []
    for name in image_names:
        aux_dict = {}
        aux_dict["image_id"] = name + "." + os.path.basename(res[0][0].split(".")[-1])
        aux_dict["annotations"] = []
        crop_data = [x for x in res if name in x[0]]
        for cat in categories:
            mask = np.zeros((1024, 4096))
            # Generate merged mask
            for data in crop_data:
                x = int(os.path.basename(data[0]).split("_")[-2])
                y = int(os.path.basename(data[0]).split("_")[-1][:-4])
                idx = np.where(data[1] == cat)[0]
                if len(idx) > 0:
                    cropped_mask = data[-1][idx][0, :, :]
                    mask = merge_masks(mask, cropped_mask, x, y)

            if mask.max() > 0:
                # Apply filling with ones or dilation to help merging the masks
                if cat == 9:  # only borders
                    loc = np.where(mask == 1)
                    min_x = min(loc[1])
                    max_x = max(loc[1])
                    min_y = min(loc[0])
                    max_y = max(loc[0])
                    mask[min_y:max_y, min_x:max_x] = 1
                else:
                    kernel = np.ones((1, 1), np.uint8)  # 20
                    mask = cv.dilate(mask, kernel)
                # Contours and bounding boxes on entire mask
                contours, bboxes = extract_contour_and_bbox(mask)

                # Merging scores on merged masks
                scores = []
                for data in crop_data:
                    extended_mask = np.zeros((1024, 4096))
                    x = int(os.path.basename(data[0]).split("_")[-2])
                    y = int(os.path.basename(data[0]).split("_")[-1][:-4])
                    idx = np.where(data[1] == cat)[0]
                    if len(idx) > 0:
                        cropped_mask = data[-1][idx][0, :, :]
                        cropped_mask = merge_masks(extended_mask, cropped_mask, x, y)
                        # kernel = np.ones((20,20),np.uint8) #20
                        # cropped_mask = cv.dilate(cropped_mask, kernel)
                        _, crop_bboxes = extract_contour_and_bbox(cropped_mask, apply=True)
                        ious = [(i, j, get_iou(bboxes[i], crop_bboxes[j])) for i in range(len(bboxes)) for j in range(len(crop_bboxes))]
                        for iou in ious:
                            if iou[-1] > 0:
                                score = data[2][idx][iou[1]]
                                scores.append((iou[0], score))
                # Saving result
                for i in range(len(bboxes)):
                    final_score = np.mean([x[1] for x in scores if x[0] == i])
                    aux_dict["annotations"].append({
                        "category_id": cat+1,
                        "score": final_score,
                        "bbox": bboxes[i],
                        "segmentation": contours[i]
                    })
        post_result.append(aux_dict)

    return post_result


def main(multithread: bool = False, preprocess: bool = False, images: List = []) -> None:
    args = {"score_threshold": 0.2}
    weights = "./yolact_edge/weights/yolact_plus_resnet50_qualitex_custom_2_121_115000.pth"
    config = "yolact_resnet50_qualitex_custom_2_config"
    dataset = "qualitex_dataset"
    calib_images = None
    score = args["score_threshold"]

    # Preprocess - crop images to 512x512px
    # eval_path = "./eval_lectra"
    # eval_paths = list(Path(eval_path).glob('*'))
    # #* Selecting one image
    # eval_paths = [x for x in eval_paths if str(x).endswith("F0_R115652_C0.jpg")]
    # eval_paths = [x for x in eval_paths if str(x).endswith("F0_R111438_C2.jpg")]
    print("Cropping images")
    preprocess_crop(images, preprocess)

    # Creating inference object
    inference = Run(weights=weights, model_config=config, dataset=dataset, calib_images=calib_images, args=args)

    # Initializing inference
    inference.model_init()

    # Launch a prediction
    n_workers = 4  # len(paths)//2
    paths = list(Path(".tmp").glob('*'))  # path to cropped images

    # Model Inference
    if multithread:
        paths = [[str(p), score] for p in paths]
        print("Launching inference on batch")
        start_time = time()
        with ThreadPool(n_workers) as pool:
            res = pool.starmap(inference.model.predict_simple, paths)
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

    # Postprocess masks: merging to original size
    res = postprocess_masks(res)

    # # Postprocess predictions
    # if not os.path.exists("./results/"):
    #     os.mkdir("./results/")

    # # res = postprocess_output(res)
    # output_path = "./results/output.json"
    # with open(output_path, "w") as j:
    #     json.dump(res, j, cls=NpEncoder)
    # print("Postprocess finished, output created.")

    # Clean .tmp images
    os.system("rm -r .tmp/*.png")
    return res


def execute_task(task_slug):
    images = PROVIDER.download_images(task_slug)
    prediction = Prediction()

    # Run prediction on images
    result = main(multithread=True, preprocess=True, images=images)  # False demo Lectra
    defects = []
    scores = []
    segmentations = []
    for anno in result[0]["annotations"]:
        category = anno["category_id"]
        score = anno["score"]
        segmentation = ",".join([str(tuple(x)) for x in anno["segmentation"]])
        if score > 0 and category == 1:
            scores.append(score)
            segmentations.append(segmentation)
            # defects.append(LabelPrediction(value=f"?value={segmentation}",score=score))

    defects = [LabelPrediction(value=f"?value={y}", score=x) for x, y in sorted(zip(scores, segmentations), reverse=True) if x > 0.4]
    if len(defects) == 0:
        x = sorted(zip(scores, segmentations), reverse=True)
        try:
            defects = [LabelPrediction(value=f"?value={x[0][1]}", score=x[0][0])]
        except Exception:
            defects = []
    print(len(defects))
    # prediction.result["defect"] = LabelPrediction(value=f"?value={segmentation}",score=score
    prediction.result["defects"] = defects
    saved = (prediction,)
    # PROVIDER._save(task_slug, prediction.dict())
    PROVIDER.save_and_finish(task_slug, saved)
    return


def loop_execution():
    while True:
        try:
            task = PROVIDER.take_charge()

        except ValidationError:
            sleep(5)
            continue
        execute_task(task.slug)


if __name__ == '__main__':
    loop_execution()
    # execute_task("dcdbcd0c-8343-4e54-bcab-fa38339b7833")
