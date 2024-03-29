import numpy as np


def decode_bbox(anchors, raw_outputs, variances=None):
    """
    Decode the actual bbox according to the anchors.
    the anchor value order is:[xmin,ymin, xmax, ymax]
    :param anchors: numpy array with shape [batch, num_anchors, 4]
    :param raw_outputs: numpy array with the same shape with anchors
    :param variances: list of float, default=[0.1, 0.1, 0.2, 0.2]
    :return:
    """
    if variances is None:
        variances = [0.1, 0.1, 0.2, 0.2]
    anchor_centers_x = (anchors[:, :, 0:1] + anchors[:, :, 2:3]) / 2
    anchor_centers_y = (anchors[:, :, 1:2] + anchors[:, :, 3:]) / 2
    anchors_w = anchors[:, :, 2:3] - anchors[:, :, 0:1]
    anchors_h = anchors[:, :, 3:] - anchors[:, :, 1:2]
    raw_outputs_rescale = raw_outputs * np.array(variances)
    predict_center_x = raw_outputs_rescale[:, :, 0:1] * anchors_w + anchor_centers_x
    predict_center_y = raw_outputs_rescale[:, :, 1:2] * anchors_h + anchor_centers_y
    predict_w = np.exp(raw_outputs_rescale[:, :, 2:3]) * anchors_w
    predict_h = np.exp(raw_outputs_rescale[:, :, 3:]) * anchors_h
    predict_x_min = predict_center_x - predict_w / 2
    predict_y_min = predict_center_y - predict_h / 2
    predict_x_max = predict_center_x + predict_w / 2
    predict_y_max = predict_center_y + predict_h / 2
    predict_bbox = np.concatenate([predict_x_min, predict_y_min, predict_x_max, predict_y_max], axis=-1)

    return predict_bbox


if __name__ == '__main__':
    decode_bbox(anchors=[], raw_outputs=[])
