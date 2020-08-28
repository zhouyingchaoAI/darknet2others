from tensorflow.tools.graph_transforms import TransformGraph
from tensorflow.python import ops
import tensorflow as tf
import os
import argparse

def get_graph_def_from_file(graph_filepath):
    tf.reset_default_graph()
    with ops.Graph().as_default():
        with tf.gfile.GFile(graph_filepath, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            return graph_def

def optimize_graph(model_dir, graph_filename, transforms, output_names, outname='optimized_model.pb'):
    input_names = ['input_image',] # change this as per how you have saved the model
    graph_def = get_graph_def_from_file(os.path.join(model_dir, graph_filename))
    optimized_graph_def = TransformGraph(
    graph_def,
    input_names,
    output_names,
    transforms)
    tf.train.write_graph(optimized_graph_def,
    logdir=model_dir,
    as_text=False,
    name=outname)
    print('Graph optimized!')

transforms = ['remove_nodes(op=Identity)',
              'merge_duplicate_nodes',
              'strip_unused_nodes',
              'fold_constants(ignore_errors=true)',
              'fold_batch_norms',
              'quantize_weights']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', type=str, default='/home/zyc/Desktop/weights_space/tmp.pb', help='onnx file path')
    parser.add_argument('--optf', type=str, default='/home/zyc/Desktop/weights_space/tmp_opti.pb', help='tf file path')
    parser.add_argument('--spp', action='store_true', default=False, help='spp ')
    opt = parser.parse_args()

    if opt.spp:
        output_node_names = ["input", "Concat_383"]
    else:
        output_node_names = ["input", "Concat_376"]

    optimize_graph('', opt.tf, transforms, output_node_names, outname=opt.optf)
