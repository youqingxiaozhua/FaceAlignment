import os
from multiprocessing import current_process
import cv2

from facenet_pytorch import MTCNN


class Detector:
    """
    A face detector to detect faces in images
    It could be applied as a landmark generator for face aligning.
    """
    def __init__(self, device='cuda:0'):
        self.model = MTCNN(keep_all=True, device=device)

    def detect_one_image(self, img):
        """
        perform MTCNN on the image
        Output:
            landmarks: List[List[int]] | None, list of landmarks with the 
                probs from high to low.
        """
        if isinstance(img, str):
            img = cv2.imread(img)
        boxes, probs, landmarks = self.model.detect(img, landmarks=True)
        return boxes, probs, landmarks


workers_per_gpu = 3
devices = range(4)
devices = [f'cuda:{i}' for i in devices]

p = current_process()
if p.name == 'MainProcess':
    pass
else:   # 'SpawnPoolWorker-1'
    p_id = int(p.name.split('-')[-1])
    device_id = p_id % len(devices)
    detector = Detector(device=devices[device_id])


def get_landmarks_by_MTCNN(folder, filename):
    _, _, landmarks = detector.detect_one_image(os.path.join(folder, filename))
    return None if landmarks is None else landmarks[0]


if __name__ == '__main__':
    from align_func import Align

    aligner = Align(crop_size=112)
    aligner.align_list_of_folders(
        folder_path='data/MS1M/MS1M_Origin',
        save_path='data/MS1M/align_DetectXUE',
        landmark_func=get_landmarks_by_MTCNN,
        workers=workers_per_gpu * len(devices))

