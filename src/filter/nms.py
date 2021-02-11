import numpy as np


def single_class_non_max_suppression(b_boxes, confidences, conf_thresh=0.2, iou_thresh=0.5, keep_top_k=-1):
    """
    do nms on single class.
    Hint: for the specific class, given the bbox and its confidence,
    1) sort the bbox according to the confidence from top to down, we call this a set
    2) select the bbox with the highest confidence, remove it from set, and do IOU calculate with the rest bbox
    3) remove the bbox whose IOU is higher than the iou_thresh from the set,
    4) loop step 2 and 3, util the set is empty.
    :param b_boxes: numpy array of 2D, [num_b_boxes, 4]
    :param confidences: numpy array of 1D. [num_b_boxes]
    :param conf_thresh:
    :param iou_thresh:
    :param keep_top_k:
    :return:
    """
    if len(b_boxes) == 0:
        return []

    conf_keep_idx = np.where(confidences > conf_thresh)[0]

    b_boxes = b_boxes[conf_keep_idx]
    confidences = confidences[conf_keep_idx]

    pick = []
    x_min = b_boxes[:, 0]
    y_min = b_boxes[:, 1]
    x_max = b_boxes[:, 2]
    y_max = b_boxes[:, 3]

    area = (x_max - x_min + 1e-3) * (y_max - y_min + 1e-3)
    ids = np.argsort(confidences)

    while len(ids) > 0:
        last = len(ids) - 1
        i = ids[last]
        pick.append(i)

        # keep top k
        if keep_top_k != -1:
            if len(pick) >= keep_top_k:
                break

        overlap_x_min = np.maximum(x_min[i], x_min[ids[:last]])
        overlap_y_min = np.maximum(y_min[i], y_min[ids[:last]])
        overlap_x_max = np.minimum(x_max[i], x_max[ids[:last]])
        overlap_y_max = np.minimum(y_max[i], y_max[ids[:last]])
        overlap_w = np.maximum(0, overlap_x_max - overlap_x_min)
        overlap_h = np.maximum(0, overlap_y_max - overlap_y_min)
        overlap_area = overlap_w * overlap_h
        overlap_ratio = overlap_area / (area[ids[:last]] + area[i] - overlap_area)

        need_to_be_deleted_idx = np.concatenate(([last], np.where(overlap_ratio > iou_thresh)[0]))
        ids = np.delete(ids, need_to_be_deleted_idx)

    # if the number of final b_boxes is less than keep_top_k, we need to pad it.
    return conf_keep_idx[pick]


if __name__ == '__main__':
    single_class_non_max_suppression(b_boxes=[], confidences=[])
