import tensorflow as tf

from settings import MODEL_PATH


def load_tf_model():
    """
    Load the model.
    :return: session and graph
    """
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.Session(graph=detection_graph)
                return sess, detection_graph


def tf_inference(sess, detection_graph, img_arr):
    """
    Receive an image array and run inference
    :param sess: tensorflow session.
    :param detection_graph: tensorflow graph.
    :param img_arr: 3D numpy array, RGB order.
    :return:
    """
    image_tensor = detection_graph.get_tensor_by_name('data_1:0')
    detection_bboxes = detection_graph.get_tensor_by_name('loc_branch_concat_1/concat:0')
    detection_scores = detection_graph.get_tensor_by_name('cls_branch_concat_1/concat:0')
    # image_np_expanded = np.expand_dims(img_arr, axis=0)
    b_boxes, scores = sess.run([detection_bboxes, detection_scores],
                               feed_dict={image_tensor: img_arr})

    return b_boxes, scores


if __name__ == '__main__':
    load_tf_model()
