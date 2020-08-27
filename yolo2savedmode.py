import os
import shutil
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgpatch', type=str, default='/home/zyc/Desktop/weights_space/cfg', help='cfg file path')
    parser.add_argument('--weightspatch', type=str, default='/home/zyc/Desktop/weights_space/weights',
                        help='path to weights file')
    parser.add_argument('--trt', action='store_true', default=True, help='half precision FP16 inference')

    # parser.add_argument('--fp16', action='store_true', help='enable fp16')
    opt = parser.parse_args()

    print(opt)
    print(os.getcwd())

    if os.path.exists(opt.weightspatch + "/savedmodels"):
        os.rmdir(opt.weightspatch + "/savedmodels")
    os.mkdir(opt.weightspatch + "/savedmodels", 0o0755)

    weights_path = os.walk(r"%s" % opt.weightspatch)

    weights_list = [file_list for path, dir_list, file_list in weights_path][0]

    conver_cmd_list = []
    for weights in weights_list:
        cfgfile = opt.cfgpatch + '/' + weights.split('.', -1)[0] + '.cfg'
        weightfile = opt.weightspatch + '/' + weights
        conver_cmd = "python weights2savedmodel.py --weights %s --cfg %s --savedmodel %s %s %s" % \
                     (weightfile, cfgfile, opt.weightspatch + '/savedmodels/' + weights.split('.', -1)[0], \
                      '--trt' if opt.trt else '', '--spp' if weights.find('spp') is not -1 else '')
        conver_cmd_list.append(conver_cmd)
        print(conver_cmd)

    for cmd in conver_cmd_list:
        stream = os.popen(cmd)
        print(stream.read())


        #python weights2savedmodel.py --cfg cfg/yolov3_416_digitalled.cfg --savedmodel weights/models --trt --spp