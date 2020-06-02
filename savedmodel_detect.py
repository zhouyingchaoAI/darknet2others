import tensorflow as tf
import cv2
import numpy as np
import random
import colorsys
# video_path      = "./docs/images/road.mp4"
video_path      = 0

path = '/home/zyc/Desktop/onnx_tflite_yolov3-dependabot-pip-tensorflow-gpu-1.15.2/weights/model/1'

def draw_bbox(image, image_size, bboxes, labels, scores, thread = 0.1):

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
            bbox_color = colors[(class_ind)]
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


with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], path)
    graph = tf.get_default_graph()

    vid = cv2.VideoCapture(video_path)
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = cv2.resize(frame, (1920, 1080))
        else:
            raise ValueError("No image!")
        frame_size = frame.shape[:2]
        print(frame_size)
        data_set = []
        image_jp = cv2.imencode('.jpg', frame)[1]
        image_jp = np.squeeze(image_jp, 1).tostring()
        data_set.append(image_jp)
        x = sess.graph.get_tensor_by_name('encoded_image_tensor:0')
        y1 = sess.graph.get_tensor_by_name('strided_slice_19:0')
        y2 = sess.graph.get_tensor_by_name('Cast_2:0')
        y3 = sess.graph.get_tensor_by_name('strided_slice_21:0')
        y4 = sess.graph.get_tensor_by_name('Tile:0')

        scores, classes, boxes, m4 = sess.run([y1, y2, y3, y4],
                          feed_dict={x: data_set})
        image = draw_bbox(frame, frame_size, boxes[0], classes[0], scores[0], 0.5)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        data_set.clear()



