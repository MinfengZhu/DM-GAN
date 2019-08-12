# Code derived from tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
import scipy.misc
import math
import sys

MODEL_DIR = '/tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(images, splits=10):
    #assert (type(images) == list)
    #assert (type(images[0]) == np.ndarray)
    #assert (np.max(images[0]) > 10)
    #assert (np.min(images[0]) >= 0.0)
    inps = images
    bs = 1
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        print(" ")
        for i in range(n_batches):
            if i % 100 == 0:
                sys.stdout.write("\r[Running] [{}/{}] ...   ".format(i * bs, len(inps)))
            inp = []
            for j in range(bs):
                img = scipy.misc.imread(inps[i*bs+j])
                img = preprocess(img)
                inp.append(img)
            #inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        print()
        return np.mean(scores), np.std(scores)


# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('[Model] Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)


if softmax is None:
    _init_inception()


def preprocess(img):
    if len(img.shape) == 2:
        img = np.resize(img, (img.shape[0], img.shape[1], 3))
    img = scipy.misc.imresize(img, (299, 299, 3), interp='bilinear')
    img = img.astype(np.float32)
    #return img
    return np.expand_dims(img, 0)


def load_data(fullpath):
    print('[Data] Read data from ' + fullpath)
    images = []
    for path, subdirs, files in os.walk(fullpath):
        # import pdb; pdb.set_trace()
        for name in files:
            if name.rfind('jpg') != -1 or name.rfind('png') != -1:
                filename = os.path.join(path, name)
                if os.path.isfile(filename):
                    #img = scipy.misc.imread(filename)
                    #import pdb; pdb.set_trace()
                    #img = preprocess(img)
                    images.append(filename)
                    sys.stdout.write("\r[Data] [{}] ...   ".format(len(images)))
    print('')
    #print(images[0])
    #print('[Data] # images: {} '.format(len(images)))
    return images

def inception_score(path):
    images = load_data(path)
    mean, std = get_inception_score(images)
    return mean, std


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    # Command: CUDA_VISIBLE_DEVICES=1 python xxx.py path
    # Image Folder Path
    path = sys.argv[1]  #
    images = load_data(path)
    print('.......')
    mean, std = get_inception_score(images)
    print("[Inception Score] mean: {:.2f} std: {:.2f}".format(mean, std))
