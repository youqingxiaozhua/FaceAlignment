import os
import multiprocessing
from functools import partial

import numpy as np
import cv2

ctx = multiprocessing.get_context("spawn")


from align_trans import get_reference_facial_points, warp_and_crop_face


class Align:
    def __init__(self, crop_size=112):
        super().__init__()
        self.crop_size = crop_size
        self.scale = crop_size / 112.
        self.reference = get_reference_facial_points(default_square = True) * self.scale

    def align_one_img(self, img: np.ndarray, landmarks=None) -> np.ndarray:
        warped_face = warp_and_crop_face(np.array(img), landmarks, self.reference, crop_size=(self.crop_size, self.crop_size))
        return warped_face

    def align_one_image_warp(self, folder_path, save_path, landmark_func, image_name):
        """
        a warp function to perform multiprocessing
        """
        try:
            landmarks = landmark_func(folder_path, image_name)
        except Exception as e:
            print(f'Get landmark fails: {e}')
            return
        img = cv2.imread(os.path.join(folder_path, image_name))
        aligned = self.align_one_img(img, landmarks)
        save_path = os.path.join(save_path, image_name)
        cv2.imwrite(save_path, aligned)

    def align_one_folder(self, folder_path, save_path, landmark_func, img_buffix=('jpg', 'png'),
            workers=3):
        """
        args:
            landmark_func: a function to return landmarks with args: folder, filename
        """
        os.makedirs(save_path, exist_ok=True)
        images = os.listdir(folder_path)
        images = [i for i in images if i.split('.')[-1].lower() in img_buffix]
        # print(f'There are {len(images)} images in {folder}')

        func = partial(self.align_one_image_warp, folder_path, save_path, landmark_func)
        if workers > 1:
            with ctx.Pool(workers) as p:
                p.map(func, images)
        else:
            r = list(map(func, images))
    
    def align_list_of_folders(self, folder_path, save_path, landmark_func,
            progress_bar=True, workers=3,
            img_buffix=('jpg', 'png')):
        """
        align a list of folders with images in them.
        .
        └── folder_path/
            ├── sub_folder1/
            │   ├── image1
            │   └── image2
            └── sub_folder2/
                └── image3

        Args:
            folder: the folder which has a list of image folders
        """
        os.makedirs(save_path, exist_ok=True)
        sub_folders = os.listdir(folder_path)
        
        folder_paths = [os.path.join(folder_path, i) for i in sub_folders]
        save_paths = [os.path.join(save_path, i) for i in sub_folders]
        func = partial(self.align_one_folder, img_buffix=img_buffix,
                        landmark_func=landmark_func,
                        workers=1)
        
        if progress_bar:
            from tqdm import tqdm
            pbar = tqdm(total=len(sub_folders))
            def update(*args):
                pbar.update()
            callback = update
        else:
            callback = None

        pool = ctx.Pool(workers)

        for i in range(len(sub_folders)):
            pool.apply_async(func, args=(folder_paths[i], save_paths[i]),
                            callback=callback)
        pool.close()
        pool.join()


if __name__ == '__main__':
    from vis_landmark import landmark_n_to_5

    def ms1m_landmark_func(folder, filename):
        landmark_file = os.path.join(folder, filename + '.SDMShape2')
        with open(landmark_file, 'r') as f:
            landmarks = f.read().strip().split('\n')[1:]
        landmarks = [i.split(' ')[1:] for i in landmarks]
        landmarks = [[float(i[0]), float(i[1])] for i in landmarks]
        return landmark_n_to_5(landmarks)
    
    aligner = Align(crop_size=112)
    # aligner.align_one_folder('data/MS1M/MS1M_Origin/m.023bvv', 'data/MS1M/align_XUE/m.023bvv', ms1m_landmark_func)
    aligner.align_list_of_folders('data/MS1M/MS1M_Origin', 'data/MS1M/align_XUE', ms1m_landmark_func)

