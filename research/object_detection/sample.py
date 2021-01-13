import tensorflow as tf
from object_detection.data_decoders import tf_example_decoder
from object_detection.utils import dataset_util
from PIL import Image

#from tensorflow_core.python.platform import evaluation
from tensorflow.python.training import evaluation
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook



imgpath = "/data/test_splitimages_21.jpg"
with tf.io.gfile.GFile(imgpath, 'rb') as fid:
    encoded_jpg = fid.read()
    im = Image.open(imgpath)
    width, height = im.size


tf_example = tf.train.Example(features=tf.train.Features(feature={
    'image/height': dataset_util.int64_feature(height),
    'image/width': dataset_util.int64_feature(width),
    'image/filename': dataset_util.bytes_feature("myfile.jpg".encode("utf8")),
    'image/source_id': dataset_util.bytes_feature("myfile.jpg".encode("utf8")),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg),
    'image/format': dataset_util.bytes_feature(b'jpg'),
    'image/object/bbox/xmin': dataset_util.float_list_feature([50]),
    'image/object/bbox/xmax': dataset_util.float_list_feature([70]),
    'image/object/bbox/ymin': dataset_util.float_list_feature([20]),
    'image/object/bbox/ymax': dataset_util.float_list_feature([70]),
    'image/object/class/text': dataset_util.bytes_list_feature(['shark'.encode('utf8')]),
    'image/key/sha256': dataset_util.bytes_feature('Daniel Moore'.encode('utf8')),
    'image/labelbox/datarow_id': dataset_util.bytes_feature('Work will you'.encode('utf8')),
    'images/labelbox/view_url': dataset_util.bytes_feature('Testing this thing'.encode('utf8')),
    'image/object/class/label': dataset_util.int64_list_feature([2]),
}))

decoder = tf_example_decoder.TfExampleDecoder(
    load_instance_masks=False,
    load_multiclass_scores=False,
    load_context_features=False,
    instance_mask_type=1,
    label_map_proto_file='/data/mini-shark/tfrecord/ckcgqorltvxoi08974xshx1wi_2021-01-07T1806.pbtxt',
    use_display_name=False,
    num_additional_channels=0,
    num_keypoints=0,
    expand_hierarchy_labels=False,
    load_dense_pose=False,
    load_track_id=False)

#print(tf.train.Example.FromString(tf_example.SerializeToString()))

def fn():
    #coll = tf.get_collection(ops.GraphKeys.EVAL_STEP)
    coll = tf.get_collection(ops.GraphKeys.LOCAL_VARIABLES)
    return len(coll)

tensor = decoder.decode(tf_example.SerializeToString())
key_tensor = tensor['key']


#eval_step = evaluation._get_or_create_eval_step()
eval_step = variable_scope.get_variable(
        'eval_step',
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.zeros_initializer(),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP])
with tf.control_dependencies([tf.variables_initializer([eval_step]), eval_step]):
    update_eval_step = state_ops.assign_add(eval_step, 1, use_locking=True)
    fn_op = tf.py_func(fn, [],  tf.int64)

    eval_step_value = evaluation._get_latest_eval_step_value([key_tensor])
    with ops.control_dependencies([key_tensor]):
        graph = ops.get_default_graph()
        curr_eval_step = graph.get_collection(ops.GraphKeys.EVAL_STEP)[0]
        eval_step_value = array_ops.identity(curr_eval_step.read_value())

    op_list = [key_tensor, fn_op, update_eval_step]

#RESUME replicate the py_func environment? or at least the "string decode inside graph stuff"
# maybe try in TF 2, eager execution
sess = tf.compat.v1.Session()
print(sess.run(op_list))
sess.close()


#session = tf.Session()
#with session.as_default():
#    res = session.run(result)
#    print(res)

   