import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import shutil
import argparse
from utils.parse_config import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='/home/zyc/Desktop/weights_space/cfg/yolov3_fire.cfg', help='cfg file path')
    parser.add_argument('--weights', type=str, default='/home/zyc/Desktop/weights_space/weights/yolov3_fire.weights',
                        help='path to weights file')
    parser.add_argument('--savedmodel', type=str, default='/home/zyc/Desktop/weights_space/weights/savedmodels/yolov3_fire', help='savedmode path')
    parser.add_argument('--trt', action='store_true', default=True, help='half precision FP16 inference')
    parser.add_argument('--spp', action='store_true', default=False, help='spp ')

    # parser.add_argument('--fp16', action='store_true', help='enable fp16')
    opt = parser.parse_args()

    tmp_patch = opt.weights[0:(opt.weights[0:(opt.weights).rfind('/')].rfind('/'))]

    if os.path.exists(opt.savedmodel):
        shutil.rmtree(opt.savedmodel, ignore_errors=True)
    #python detect.py --cfg cfg/yolov3-crane-excavator.cfg --weights weights/yolov3-crane-excavator_best.weights --onnx weights/crane.onnx
    #python onnx2tf.py --onnx weights/crane.onnx --tf weights/crane.pb
    #python tf_opti.py --tf weights/tools_spp.pb --optf weights/tools_spp_opti.pb
    #python pb2savedmodel_batch.py --pb_dir weights/tools_spp_opti.pb --output_dir weights/model

    weights2onnx_cmd = "python detect.py --cfg %s --weights %s --onnx %s" % \
                  (opt.cfg, opt.weights,  tmp_patch + '/tmp.onnx' )
    onnx2tf_cmd = "python onnx2tf.py --onnx %s --tf %s" % \
                  (tmp_patch + '/tmp.onnx', tmp_patch + '/tmp.pb')

    class_num = 3
    modelsss = parse_model_cfg(opt.cfg)
    for m in modelsss:
        if m['type'] == 'yolo':
            class_num = m['classes']
            break

    if opt.trt:
        if opt.spp:
            tf2opti_cmd = "python tf_opti.py --tf %s --optf %s --spp" % \
                          (tmp_patch + '/tmp.pb', tmp_patch + '/tmp_opti.pb')
            poti2savedmodel_cmd = "python pb2savedmodel_trt.py --pb_dir %s --output_dir %s --output_tensor %s --class_num %s" % \
                          (tmp_patch + '/tmp_opti.pb',  opt.savedmodel, 'Concat_383:0', class_num)
        else:
            tf2opti_cmd = "python tf_opti.py --tf %s --optf %s" % \
                          (tmp_patch + '/tmp.pb', tmp_patch + '/tmp_opti.pb')
            poti2savedmodel_cmd = "python pb2savedmodel_trt.py --pb_dir %s --output_dir %s --output_tensor %s  --class_num %s" % \
                          (tmp_patch + '/tmp_opti.pb',  opt.savedmodel, 'Concat_376:0', class_num)
    else:
        if opt.spp:
            tf2opti_cmd = "python tf_opti.py --tf %s --optf %s --spp" % \
                          (tmp_patch + '/tmp.pb', tmp_patch + '/tmp_opti.pb')
            poti2savedmodel_cmd = "python pb2savedmodel_batch.py --pb_dir %s --output_dir %s --output_tensor %s  --class_num %s" % \
                          (tmp_patch + '/tmp_opti.pb',  opt.savedmodel, 'Concat_383:0', class_num)
        else:
            tf2opti_cmd = "python tf_opti.py --tf %s --optf %s" % \
                          (tmp_patch + '/tmp.pb', tmp_patch + '/tmp_opti.pb')
            poti2savedmodel_cmd = "python pb2savedmodel_batch.py --pb_dir %s --output_dir %s  --output_tensor %s  --class_num %s" % \
                          (tmp_patch + '/tmp_opti.pb',  opt.savedmodel, 'Concat_376:0', class_num)

    conver_cmd_list = [weights2onnx_cmd, onnx2tf_cmd, tf2opti_cmd, poti2savedmodel_cmd]


    for cmd in conver_cmd_list:
        print(cmd)
        stream = os.popen(cmd)
        print(stream.read())
