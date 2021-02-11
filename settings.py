import os

CUR_DIR = os.path.dirname(os.path.join(__file__))
MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'face_mask_detection.pb')

FEATURE_MAP_SIZES = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
ANCHOR_SIZES = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
ANCHOR_RATIOS = [[1, 0.62, 0.42]] * 5
THRESH = 0.5

WEB_CAM = True
IP_CAM_ADDRESS = ""
MASK_AUDIO_FILE_PATH = ""
NON_MASK_AUDIO_FILE_PATH = ""
