
import numpy as np
import matplotlib.pyplot as plt


def show_landmark(img:np.ndarray, landmarks, save_path=None):
    plt.figure()
    plt.imshow(img)
    for i, landmark in enumerate(landmarks):
        point_index = i
        plt.scatter(int(landmark[0]), int(landmark[1]), 3)
        plt.text(landmark[0], landmark[1], str(point_index), fontsize=8)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def landmark_n_to_5(landmark, n=None):
    """Convert landmark with n points to 5 points.
    If you are not sure about the point index, plz use the `show_landmark ` to identity it.
    left means the left of the image
    5 points order: left eye, right eye, apex nasi, left mouth, right mouth
    """
    if n is None:
        n = len(landmark)
    if n == 74:
        return landmark[44], landmark[53], (landmark[30] + landmark[33])/2, landmark[54], landmark[60]
    elif n == 25:   # MS1M
        return landmark[0], landmark[1], landmark[7], landmark[10], landmark[11]
    raise ValueError('Unsupport n')


if __name__ == '__main__':
    import mmcv
    img_file = 'data/MS1M/MS1M_Origin/m.023bvv/29-FaceId-0_align.jpg'
    img = mmcv.imread(img_file)
    landmark_file = img_file + '.SDMShape2'
    with open(landmark_file, 'r') as f:
        landmarks = f.read().strip().split('\n')[1:]
    landmarks = [i.split(' ')[1:] for i in landmarks]
    landmarks = [[float(i[0]), float(i[1])] for i in landmarks]
    show_landmark(img, landmarks, save_path='landmark.jpg')
    from align_func import Align
    landmarks = landmark_n_to_5(landmarks)
    aligned = Align(112).align_one_img(img, landmarks)
    mmcv.imwrite(aligned, 'aligned.jpg')

