
import tensorflow.compat.v1 as tf
import tf_slim as slim
import pprint

from object_detection.builders.model_builder import _build_ssd_feature_extractor
from object_detection.anchor_generators import multiple_grid_anchor_generator as ag
from object_detection.builders import anchor_generator_builder
from object_detection.models.ssd_mobilenet_v1_feature_extractor import SSDMobileNetV1FeatureExtractor
from object_detection.utils import config_util
from object_detection.utils import shape_utils
from object_detection.utils.config_util import get_configs_from_pipeline_file




def build_anchor_boxes():
    pipeline_config_path: str = "/home/tensorflow/models/research/object_detection/dockerfiles/tf1/ssd_mobilenet_v1_sharks.config"

    configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
    model_config = configs['model']
    # TODO use a higher level function that parses 'ssd' or other meta architecture
    ssd_config = model_config.ssd
    anchor_config = ssd_config.anchor_generator
    anchor_generator = anchor_generator_builder.build(anchor_config)

    #feature_extractor = SSDMobileNetV1FeatureExtractor(
    #    is_training=False,
    #    depth_multiplier=1.0,
    #    min_depth=16,
    #    pad_to_multiple=1,
    #    conv_hyperparams_fn=lambda x: print ('nope'),
    #)

    dummy_image = tf.ones([1,300,300,3])

    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_config.feature_extractor,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        is_training=False
    )

    # Idk if this is needed. Copied from ssd_meta_arch.py:L617
    with slim.arg_scope([slim.batch_norm], is_training=False,
                        updates_collections='update_ops'):
        feature_maps = feature_extractor.extract_features(dummy_image)
    
    feature_map_spatial_dims = _get_feature_map_spatial_dims(feature_maps)

    #feature_map_spatial_dims = [ (w, h), (w2, h2), (w3, h3) ]; //see ssd_meta_arch.py L583


    boxlist_list = anchor_generator.generate(
        feature_map_spatial_dims,
        im_height=300,
        im_width=300
    )

    """
    [ boxlist_list
        [ boxlist.get_whatever()
            [ box.tolist()
                h, w, x, y
            ],
            [
            ]
        ],
        [

        ]
    ]
    """

    pp = pprint.PrettyPrinter(indent=4)
    first = True
    #with tf.Session() as sess:
    result = []
    #pp.pprint(boxlist_list[-1].get_center_coordinates_and_sizes()[-1].numpy().tolist())
    for idx, boxlist in enumerate(boxlist_list):
        layer = []
        #boxcoordslist = sess.run(boxlist.get_center_coordinates_and_sizes())
        # TODO ? use boxlist.get() instead to match ssd_meta_arch.py:L609
        # TODO use tf.transpose & tf.unstack like faster_rcnn_box_coder.py:L104
        t_ycenters, t_xcenters, t_heights, t_widths  = boxlist.get_center_coordinates_and_sizes()
        layer = [box for box in zip(
            t_ycenters.numpy().tolist(),
            t_xcenters.numpy().tolist(),
            t_heights.numpy().tolist(),
            t_widths.numpy().tolist()
        )]

        result += [layer]


    #print(result)
    pp.pprint(result)


#        for listidx, boxlist in enumerate(boxlist_list):
#            print(f"BoxList #k{listidx} ############################################")
#            coordslist = boxlist.get_center_coordinates_and_sizes()
#            for boxidx, box in enumerate(coordslist):
#                sess.run(box)
#                print(f"Box #{boxidx}::{box.eval()}")

    return result

# Shamelessly stolen from research/object_detection/meta_architectures/ssd_meta_arch.py
def _get_feature_map_spatial_dims(feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
        feature_maps: a list of tensors where the ith tensor has shape
            [batch, height_i, width_i, depth_i].

    Returns:
        a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
        shape_utils.combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]

def main():
    build_anchor_boxes()


if __name__ == '__main__':
    main()
