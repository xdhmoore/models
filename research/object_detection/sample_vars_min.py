import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope


def fn(a):
    return a;
    #return len(tf.get_collection(ops.GraphKeys.EVAL_STEP))

eval_step = variable_scope.get_variable(
        'eval_step',
        shape=[],
        dtype=dtypes.int64,
        initializer=init_ops.constant_initializer(7),
        trainable=False,
        collections=[ops.GraphKeys.LOCAL_VARIABLES, ops.GraphKeys.EVAL_STEP])

eval_tensor = tf.get_default_graph().get_tensor_by_name('eval_step:0')

with tf.control_dependencies([tf.variables_initializer([eval_step])]):
    fn_op = tf.py_func(fn, [eval_tensor],  tf.int64)

sess = tf.compat.v1.Session()
print(sess.run(fn_op))
print(tf.get_collection(ops.GraphKeys.EVAL_STEP[0]))
