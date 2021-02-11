import cv2

import simpleaudio as sa
from src.mask.detector import MaskDetector
from settings import WEB_CAM, IP_CAM_ADDRESS, NON_MASK_AUDIO_FILE_PATH


class MaskEstimator:
    def __init__(self):
        self.mask_detector = MaskDetector()
        self.sound_ret = True

    @staticmethod
    def play_sound(audio_file_path):
        wave_obj = sa.WaveObject.from_wave_file(audio_file_path)
        play_obj = wave_obj.play()
        play_obj.wait_done()

        return

    def run(self):
        if WEB_CAM:
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(IP_CAM_ADDRESS)

        if not cap.isOpened():
            raise ValueError("Video open failed. Please set Web Camera or IP Camera")
        cnt = 0

        while cap.isOpened():
            status, img_raw = cap.read()
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)

            result_info, result_frame, alarm_ret = self.mask_detector.detect_mask(frame=img_raw, iou_thresh=0.5,
                                                                                  target_shape=(260, 260),
                                                                                  draw_result=True)

            cv2.imshow('image', result_frame)
            if result_info and self.sound_ret:
                if not alarm_ret:
                    playsound(NON_MASK_AUDIO_FILE_PATH)
                self.sound_ret = False
            if cnt > 10000:
                cnt = 0
            cnt += 1
            if cnt % 30 == 0:
                self.sound_ret = True
            if cv2.waitKey(1) & 0xff == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    MaskEstimator().run()
