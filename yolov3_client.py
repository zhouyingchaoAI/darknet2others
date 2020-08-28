import requests
import base64
import json
import cv2
import time
import numpy as np
import colorsys
import random
import datetime
import core.utils as utilsa
from PIL import Image

URL = "http://192.168.60.228:8501/v1/models/switchcard:predict"
headers = {"content-type": "application/json"}



# video_path      = "./docs/images/road.mp4"
# video_path      = "./videos/IMG_4198.MOV"
video_path      = 0
vid = cv2.VideoCapture(video_path)

# cap = cv2.VideoCapture("rtsp://admin:ut0000@192.168.60.220:554/axis-media/media.amp")
# vid = cap


def draw_bbox(image, image_size, bboxes, labels, scores, thread = 0.5):

    num_classes = 80
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    fontScale = 0.5

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for k, score in enumerate(scores):
        if score > thread:
            box = bboxes[k]
            class_ind = int(labels[k])
            bbox_color = colors[class_ind]
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (int(box[1]*image_size[1]), int(box[0]*image_size[0])), (int(box[3]*image_size[1]), int(box[2]*image_size[0]))

            cv2.rectangle(image, c1, c2, bbox_color, 2)

            if 1:
                bbox_mess = '%s: %.2f' % (labels[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image

def padd_resize(image, desired_size=416):

    desired_size = 416
    old_size = image.shape[:2]  # old_size is in (height, width) format

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    # new_size should be in (width, height) format

    image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_frame = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)
    return new_frame

def main():

    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.medianBlur(frame, 3)
            frame = padd_resize(frame)
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        print(frame_size)
        image_jp = cv2.imencode('.jpg', frame)[1]
        image_content = base64.b64encode(image_jp.tostring()).decode("utf-8")
        body = {
            "instances": [{"b64": image_content}]
        }
        prev_time = time.time()
        time_now = datetime.datetime.now()
        r = requests.post(URL, data=json.dumps(body), headers=headers)
        print("infer time cost {}".format((datetime.datetime.now() - time_now).total_seconds()))

        json_pre = json.loads(r.text)
        json_text = r.text
        print(json_text)


        predictions = json_pre['predictions']
        boxes = predictions[0]['detection_boxes']
        classes = predictions[0]['detection_classes']
        scores = predictions[0]['detection_scores']

        image = draw_bbox(frame, frame_size, boxes, classes, scores, 0.4)

        curr_time = time.time()
        exec_time = curr_time - prev_time
        result = np.asarray(image)
        info = "time: %.2f ms" %(1000*exec_time)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        print(info)
        if cv2.waitKey(1) & 0xFF == ord('q'): break


if __name__ == '__main__':
    main()