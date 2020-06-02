from tensorflow.tools.graph_transforms import TransformGraph
import tensorflow as tf
import os
from tensorflow.python import ops

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
output_node_names = ["input", "Concat_376"]

path = 'weights/yolov3.pb'
opti_path = 'weights/yolov3_opti.pb'

optimize_graph('', path, transforms, output_node_names, outname=opti_path)
