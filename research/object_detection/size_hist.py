
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from object_detection.builders.anchor_generator_builder import build
np.random.seed(82951051)

from matplotlib import colors
from matplotlib.ticker import PercentFormatter # TODO ?

from object_detection.anchor_visualizer import build_anchor_boxes
import os
from typing import List

import tensorflow.compat.v1 as tf
tf.enable_eager_execution()

class Label():
    def __init__(self, idx: int, parsed) -> None: # type: ignore
        self.idx = idx
        self.parsed = parsed

    def get_att(self, name: str) -> float:
        return tf.sparse.to_dense(self.parsed[name]).numpy()[self.idx] # type: ignore

    xmin = property(fget=lambda self: self.get_att('image/object/bbox/xmin'))
    xmax = property(fget=lambda self: self.get_att('image/object/bbox/xmax'))
    ymin = property(fget=lambda self: self.get_att('image/object/bbox/ymin'))
    ymax = property(fget=lambda self: self.get_att('image/object/bbox/ymax'))



class Record():
    def __init__(self, parsed) -> None: #type: ignore
        self.parsed = parsed 

    # TODO there's probably a more elegant way to do this
    def labels(self):
        num = 0;
        while num < len(tf.sparse.to_dense(self.parsed['image/object/bbox/xmin']).numpy()):
            yield Label(num, self.parsed)
            num += 1

def parse_fn(example_proto) -> None: #type: ignore
    return tf.io.parse_single_example( #type: ignore
        serialized=example_proto,
        features={
            # TODO might need these if I output results transformed to NN's scale 300x300 0-1
            #'image/height':
            #'image/width' :

            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32)
        }
    ) 


def main(infilename: str, outimage: str, hist_size:float) -> None:



    # READ TFR label sizes
    indataset = tf.data.TFRecordDataset(infilename).map(parse_fn) # type:ignore
    widths = []
    heights = []
    for record in [Record(row) for row in indataset]: #type: ignore
        print("Record ---------------------------------------")
        for label in record.labels():
            #print(f"Label.xmin:{label.xmin}")
            widths += [label.xmax - label.xmin]
            heights += [label.ymax - label.ymin]

    print(f"{len(widths)} widths found and {len(heights)} heights")

    colors = 'bgrcmykw'
    fig, axes = plt.subplots()
    axes.set_aspect(aspect=1)
    # TODO add legend for these colors
    axes.hist2d(widths, heights, 100, [(0, hist_size), (0, hist_size)])
    # TODO instead maybe plot a where the original image 1:1 was.
    axes.plot([0, hist_size], [0, hist_size], alpha = 0.8)
    axes.set_xlabel('box widths (0 to 1)')
    axes.set_ylabel('box heights (0 to 1)')
    #axes.hist2d(widths, heights, 100, [(0, 1), (0, 1)])
    # Read anchor box sizes

    # TODO count number of equal anchor aspect ratios and then display the # of them
    # that the dot represents
    #with open(anchor_infile, "r", encoding="utf-8") as anchorfile:
        #anchorlayerslist = json.load(anchorfile)
    anchorlayerslist = build_anchor_boxes()
    for layer_idx, layer in enumerate(anchorlayerslist):
        anchorwidths = []
        anchorheights = []
        for anchor in layer:
            # [ycenter, xcenter, height, width]
            # Maybe? I think this is?? xmin, ymin, xmax, ymax? but why are there negative numbers...
            #width = anchor[2] - anchor[0]
            #height = anchor[3] - anchor[1]
            #anchorwidths += [width]
            #anchorheights += [height]
            anchorwidths += [anchor[3]]
            anchorheights += [anchor[2]]
        # TODO add legend for these colors also
        axes.plot(anchorwidths, anchorheights, '.' + colors[layer_idx % len(colors)])

    # 32 ^2
    # 96 ^2
    # 4096 X 2160
    # TODO this isn't going to work for everything. calculate avg img height/width
    img_height: int = 2160
    img_width: int = 4096
    small_threshold_x = 32 / img_width
    small_threshold_y = 32 / img_height
    med_threshold_x = 96 / img_width
    med_threshold_y = 96 / img_height
    large_threshold_x = 1e5 / img_width
    large_threshold_y = 1e5 / img_width
    axes.plot((small_threshold_x, small_threshold_x), (0, small_threshold_y), '-r')
    axes.plot((0, small_threshold_x), (small_threshold_y, small_threshold_y), '-r')
    axes.plot((med_threshold_x, med_threshold_x), (0, med_threshold_y), '-r')
    axes.plot((0, med_threshold_x), (med_threshold_y, med_threshold_y), '-r')
    axes.plot((large_threshold_x, large_threshold_x), (0, large_threshold_y), '-r')
    axes.plot((0, large_threshold_x), (large_threshold_y, large_threshold_y), '-r')
    #axes.plot(anchorwidths, anchorheights, '.' + colors[layer_idx % len(colors)])
    #axes.plot(anchorwidths, anchorheights, '.' + colors[layer_idx % len(colors)])

    #fig, ax = plt.subplots(tight_layout=True)
    #hist = ax.hist2d(x, y)

    # This works
    #n, bins, patches = plt.hist(widths, 100, (0, 1))

    # This also works
    # TODO make line black or white or some color not on the plot?
    # plt.plot([0, 0.1], [0, 0.1], alpha = 0.8)
    # plt.hist2d(widths, heights, 100, [(0, 0.1), (0, 0.1)])
    # plt.gca().set_aspect(aspect=1)

    # TODO try adding 1d hists to top and side like:
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/scatter_hist.html#sphx-glr-gallery-lines-bars-and-markers-scatter-hist-py
    # THIS WORKS
    #fig, axes = plt.subplots()
    #axes.hist2d(widths, heights, 100, [(0, 0.1), (0, 0.1)])

# This works
    #axes.hist2d(widths, heights, 100, [(0, 1), (0, 1)])
# RESUME try to plot different layers with different colors or shapes (or numbers?)
# RESUME do a plot against image coordinates
    #axes.plot(anchorwidths, anchorheights, '.r')

    fig.savefig(outimage)

# Copies several tfrecord files into a single new one
def join_files(outfilename: str, infilenames: List[str]) -> None:
    
    total=0
    with tf.io.TFRecordWriter(outfilename) as writer: #type:ignore
        for infilename in infilenames:
            indataset = tf.data.TFRecordDataset(infilename) # type:ignore
            for rec_idx, rec in enumerate(indataset): #type: ignore
                writer.write(rec.numpy()) #type:ignore
                total+=1
            print(f"Finished writing {rec_idx+1} records to {outfilename}")

    print(f"Wrote a total of {total} records.")

    # TODO It would be nice if there was a better way of naming output files in
    # all of these scripts than just appending stuff to the end of the name
    # after the extension. See https://stackoverflow.com/a/45353565/356887
    os.rename(outfilename, outfilename + f"-{total}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot a histogram of ground truth box sizes')
    parser.add_argument('infile', metavar='infile', type=str, nargs=1, help='the name of the tf record file')
    parser.add_argument('outimage', metavar='outimage', type=str, nargs=1, help='the name of the file to write image plot to')
    #parser.add_argument('--anchorfile', type=str, help='the name of the file with anchors json data')
    parser.add_argument('--size', '-s',type=float, help="The height/width of the square histogram, from [0,1]", default=1.0)
    # TODO handle multiple categories
    # TODO output in the 300x300px, 0-1 scale

    args = parser.parse_args()
    #main(args.infile[0], args.outimage[0], args.anchorfile, args.size)
    main(args.infile[0], args.outimage[0], args.size)