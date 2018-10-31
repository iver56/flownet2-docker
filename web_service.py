from __future__ import division
from __future__ import print_function

import caffe
import numpy as np
import tempfile
from PIL import Image
from flask import Flask, request
from flask import jsonify
from math import ceil

from image_utils import base64_png_image_to_pillow_image

app = Flask(__name__)

DESIRED_WIDTH = 512
DESIRED_HEIGHT = 512
DEPLOYPROTO = '/flownet2/flownet2/models/FlowNet2/FlowNet2_deploy.prototxt.template'
CAFFEMODEL = '/flownet2/flownet2/models/FlowNet2/FlowNet2_weights.caffemodel.h5'
VERBOSE = True

# Load model
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


def process(img0, img1):
    """
    :param img0: numpy array with shape == (DESIRED_HEIGHT, DESIRED_WIDTH, 3)
        and dtype == np.uint8
    :param img1: numpy array with shape == (DESIRED_HEIGHT, DESIRED_WIDTH, 3)
        and dtype == np.uint8
    :return:
    """
    num_blobs = 2
    input_data = []

    if len(img0.shape) < 3:
        input_data.append(img0[np.newaxis, np.newaxis, :, :])
    else:
        input_data.append(
            img0[np.newaxis, :, :, :].transpose(0, 3, 1, 2)[:, [2, 1, 0], :, :]
        )

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
    while i <= 5:
        i += 1

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

    scale_x_back_factor = image1.width / DESIRED_WIDTH
    scale_y_back_factor = image1.height / DESIRED_HEIGHT
    image1 = image1.resize((DESIRED_WIDTH, DESIRED_HEIGHT), Image.LANCZOS)
    image1 = np.array(image1.convert('RGB'))
    image2 = image2.resize((DESIRED_WIDTH, DESIRED_HEIGHT), Image.LANCZOS)
    image2 = np.array(image2.convert('RGB'))

    flow = process(image1, image2)

    # Scale up the flow
    flow[:, :, 0] *= scale_x_back_factor
    flow[:, :, 1] *= scale_y_back_factor

    return jsonify(
        {
            'flow': flow.tolist(),
            'scale_x_back_factor': scale_x_back_factor,
            'scale_y_back_factor': scale_y_back_factor
        }
    )


if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        debug=False  # debug=False avoids multiple threads
    )
