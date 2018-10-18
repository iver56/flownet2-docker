from __future__ import print_function

import os

import caffe
import numpy as np
import tempfile
from PIL import Image
from flask import Flask, request
from flask import jsonify
from math import ceil
from scipy import misc

from image_utils import base64_png_image_to_pillow_image, get_apt_image_size, \
    get_temp_png_file_path

app = Flask(__name__)

DESIRED_WIDTH = 256
DESIRED_HEIGHT = 256
MAX_NUM_PIXELS = DESIRED_WIDTH * DESIRED_HEIGHT
DEPLOYPROTO = '/flownet2/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template'
CAFFEMODEL = '/flownet2/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5'
VERBOSE = True

# Load model
try:
    variables = {'TARGET_WIDTH': DESIRED_WIDTH, 'TARGET_HEIGHT': DESIRED_HEIGHT}
    divisor = 64.
    variables['ADAPTED_WIDTH'] = int(ceil(DESIRED_WIDTH/divisor) * divisor)
    variables['ADAPTED_HEIGHT'] = int(ceil(DESIRED_HEIGHT/divisor) * divisor)

    variables['SCALE_WIDTH'] = DESIRED_WIDTH / float(variables['ADAPTED_WIDTH'])
    variables['SCALE_HEIGHT'] = DESIRED_HEIGHT / float(variables['ADAPTED_HEIGHT'])

    tmp = tempfile.NamedTemporaryFile(mode='w', delete=True)
    proto = open(DEPLOYPROTO).readlines()
    for line in proto:
        for key, value in variables.items():
            tag = "$%s$" % key
            line = line.replace(tag, str(value))
        tmp.write(line)
    tmp.flush()

    if not VERBOSE:
        caffe.set_logging_disabled()
    caffe.set_device(0)  # use first GPU
    caffe.set_mode_gpu()
    net = caffe.Net(tmp.name, CAFFEMODEL, caffe.TEST)
except:
    print("Failed to load model")
    raise


def process(image1_path, image2_path):
    in0 = image1_path
    in1 = image2_path
    if not os.path.isfile(in0):
        raise BaseException('img0 does not exist: '+in0)
    if not os.path.isfile(in1):
        raise BaseException('img1 does not exist: '+in1)

    num_blobs = 2
    input_data = []
    img0 = misc.imread(in0)
    if len(img0.shape) < 3:
        input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(
            img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        )
    img1 = misc.imread(in1)
    if len(img1.shape) < 3:
        input_data.append(img1[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(
            img1[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        )

    input_dict = {}
    for blob_idx in range(num_blobs):
        input_dict[net.inputs[blob_idx]] = input_data[blob_idx]

    #
    # There is some non-deterministic nan-bug in caffe
    # it seems to be a race-condition
    #
    print('Network forward pass using %s.' % CAFFEMODEL)
    i = 1
    while i<=5:
        i+=1

        net.forward(**input_dict)

        containsNaN = False
        for name in net.blobs:
            blob = net.blobs[name]
            has_nan = np.isnan(blob.data[...]).any()

            if has_nan:
                print('blob %s contains nan' % name)
                containsNaN = True

        if not containsNaN:
            print('Succeeded.')
            break
        else:
            print('**************** FOUND NANs, RETRYING ****************')

    blob = np.squeeze(net.blobs['predict_flow_final'].data).transpose(1, 2, 0)

    return blob


@app.route("/estimate_flow/", methods=['POST'])
def estimate_flow():
    image1_base64 = request.json.get('image1_base64', None)
    if image1_base64 is None:
        raise Exception('image1_base64 cannot be None')
    else:
        image1 = base64_png_image_to_pillow_image(image1_base64)

    image2_base64 = request.json.get('image2_base64', None)
    if image2_base64 is None:
        raise Exception('image2_base64 cannot be None')
    else:
        image2 = base64_png_image_to_pillow_image(image2_base64)

    if image1.size != image2.size:
        raise Exception('The two images must have the same size')

    original_content_image_size = image1.size
    smaller_image_size = get_apt_image_size(image1, MAX_NUM_PIXELS)
    use_downsampled_image = smaller_image_size != original_content_image_size
    if use_downsampled_image:
        image1 = image1.resize(smaller_image_size, Image.LANCZOS)
        image2 = image2.resize(smaller_image_size, Image.LANCZOS)

    image1_path = get_temp_png_file_path()
    image1.save(image1_path)
    image2_path = get_temp_png_file_path()
    image2.save(image2_path)

    flow = process(image1_path, image2_path)

    if use_downsampled_image:
        # Now resize the image back to the original size, if applicable
        # Also remember to scale up the flow by the correct factor
        raise Exception('Upscaling result is not implemented. Try smaller input images.')
        pass# output_image = output_image.resize(original_content_image_size, Image.LANCZOS)

    return jsonify({'flow': flow.tolist()})


if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        debug=False  # debug=False avoids multiple threads
    )
