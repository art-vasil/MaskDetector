import cv2
import numpy as np

from src.loader.anchor_generator import generate_anchors
from src.loader.anchor_decode import decode_bbox
from src.filter.nms import single_class_non_max_suppression
from src.loader.tensorflow_loader import load_tf_model, tf_inference
from settings import ANCHOR_RATIOS, ANCHOR_SIZES, FEATURE_MAP_SIZES, THRESH


class MaskDetector:
    def __init__(self):
        self.sess, self.graph = load_tf_model()
        # generate anchors
        anchors = generate_anchors(FEATURE_MAP_SIZES, ANCHOR_SIZES, ANCHOR_RATIOS)

        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        self.anchors_exp = np.expand_dims(anchors, axis=0)

        self.id2class = {0: 'Mask', 1: 'NoMask'}

    def detect_mask(self, frame, conf_thresh=THRESH, iou_thresh=0.4, target_shape=(160, 160), draw_result=True):
        """
        Main function of detection inference
        :param frame: 3D numpy array of image
        :param conf_thresh: the min threshold of classification probabity.
        :param iou_thresh: the IOU threshold of NMS
        :param target_shape: the model input size.
        :param draw_result: whether to daw bounding box to the image.
        :return:
        """
        output_info = []
        alarm_ret = True
        height, width, _ = frame.shape
        image_resized = cv2.resize(frame, target_shape)
        image_np = image_resized / 255.0
        image_exp = np.expand_dims(image_np, axis=0)
        y_b_boxes_output, y_cls_output = tf_inference(self.sess, self.graph, image_exp)

        # remove the batch dimension, for batch is always 1 for inference.
        y_b_boxes = decode_bbox(self.anchors_exp, y_b_boxes_output)[0]
        y_cls = y_cls_output[0]
        # To speed up, do single class NMS, not multiple classes NMS.
        bbox_max_scores = np.max(y_cls, axis=1)
        bbox_max_score_classes = np.argmax(y_cls, axis=1)

        # keep_idx is the alive bounding box after nms.
        keep_ids = single_class_non_max_suppression(y_b_boxes, bbox_max_scores, conf_thresh=conf_thresh,
                                                    iou_thresh=iou_thresh)

        for idx in keep_ids:
            conf = float(bbox_max_scores[idx])
            class_id = bbox_max_score_classes[idx]
            if class_id == 1:
                alarm_ret = False
            bbox = y_b_boxes[idx]
            # clip the coordinate, avoid the value exceed the image boundary.
            x_min = max(0, int(bbox[0] * width))
            y_min = max(0, int(bbox[1] * height))
            x_max = min(int(bbox[2] * width), width)
            y_max = min(int(bbox[3] * height), height)

            if draw_result:
                if class_id == 0:
                    color = (0, 255, 0)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                cv2.putText(frame, "%s" % (self.id2class[class_id]), (x_min + 2, y_min - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
            output_info.append([class_id, conf, x_min, y_min, x_max, y_max])

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return output_info, frame, alarm_ret


if __name__ == "__main__":
    MaskDetector().detect_mask(frame=cv2.imread(""), conf_thresh=0.5)
