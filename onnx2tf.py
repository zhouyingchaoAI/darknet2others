import onnx
from onnx_tf.backend import prepare
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', type=str, default='/home/zyc/Desktop/weights_space/tmp.onnx', help='onnx file path')
    parser.add_argument('--tf', type=str, default='/home/zyc/Desktop/weights_space/tmp.pb', help='tf file path')
    opt = parser.parse_args()


    onnx_model = onnx.load(opt.onnx)  # load onnx model
    # output = prepare(onnx_model).run(input)  # run the loaded model
    # no strict to be faster
    output = prepare(onnx_model, strict=False)

    file = open(opt.tf, "wb")
    file.write(output.graph.as_graph_def().SerializeToString())
    file.close()