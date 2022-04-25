"""select the certain face in multiple faces with face recognition"""
import os
import random
from functools import partial
from math import ceil
import torch
from torch import mul
from tqdm import tqdm
from facenet_pytorch import MTCNN, extract_face, InceptionResnetV1
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2


from align_trans import get_reference_facial_points, warp_and_crop_face

gpu_id = 2  # task id
os.environ['CUDA_VISIBLE_DEVICES'] = str(7)
process_num = 1

ctx = torch.multiprocessing.get_context("spawn")

crop_size = 224
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale

fnt = ImageFont.truetype("DejaVuSans.ttf", size=40)


mtcnn = MTCNN(keep_all=True, device='cuda')
inception = InceptionResnetV1(device='cuda').eval()
pretrained = torch.load('20180408-102900-casia-webface.pt')
inception.load_state_dict(pretrained, strict=False)


def list_files(filepath):
    return os.listdir(filepath)


def extract_embedding(img):
    """
    img: 
    """
    if isinstance(img, Image.Image):
        img = np.array(img)
    if isinstance(img, np.ndarray):
        img = cv2.resize(img, (160, 160))
        x = torch.tensor(np.array(img)).permute(2, 0, 1).float()
    elif isinstance(img, torch.Tensor):
        assert list(img.shape) == [3, 160, 160], img.shape
        x = img
    else:
        raise ValueError("invalid image type")
    x = x.unsqueeze(0)
    x = (x - 127.5) / 128.0

    with torch.no_grad():
        x = inception(x.cuda())
    return x.detach().cpu()


def process_one_video(video_name:str):
    """
    video_name: "1-30-1280x720", no ".mp4", maybe end with "_left" or "_right"
    """
    saved_path = os.path.join('Aff-Wild2/aligned2', video_name)
    os.makedirs(saved_path, exist_ok=True)
    cropped_aligned_path = f'Aff-Wild2/cropped_aligned/{video_name}'
    ref_files = list_files(cropped_aligned_path)
    random.shuffle(ref_files)
    ref_images = ref_files[0:3]
    ref_features = []
    for ref_image in ref_images:
        ref_face = Image.open(os.path.join(cropped_aligned_path, ref_image)).resize((160, 160))
        ref_feature = extract_embedding(ref_face)
        ref_features.append(ref_feature)
    ref_features = torch.stack(ref_features)    # [3, c]

    images = list_files(cropped_aligned_path)

    images.sort()

    for im_name in images[:-2]:
        im_save_path =  os.path.join(saved_path, im_name)
        # if os.path.exists(im_save_path):
        #     continue
        if video_name.endswith('_left'):
            video_name_true = video_name[:-5]
        elif video_name.endswith('_right'):
            video_name_true = video_name[:-6]
        else:
            video_name_true = video_name
        video_name_true += '.mp4'

        img_path = os.path.join('Aff-Wild2/images', video_name_true)
        if not os.path.exists(img_path):
            video_name_true = video_name_true[:-4] + '.avi'
            img_path = os.path.join('Aff-Wild2/images', video_name_true)
        
        img_path = os.path.join(img_path, im_name)
        if not os.path.exists(img_path):
            print(img_path)
        continue
        img = Image.open(img_path)
        boxes, probs, points = mtcnn.detect(img, landmarks=True)
        if boxes is None:
            continue
        distances = np.zeros(len(boxes))
        for i, (box, point) in enumerate(zip(boxes, points)):
            crop_face = extract_face(img, box)
            crop_face_feature = extract_embedding(crop_face)
            distance = (ref_features - crop_face_feature.unsqueeze(0)).norm().sum(0).item()
            distances[i] = distance
        target_index = np.argmin(distances)
        warped_face = warp_and_crop_face(np.array(img), points[target_index], reference, crop_size=(crop_size, crop_size))
        img_warped = Image.fromarray(warped_face)
        img_warped.save(im_save_path)


def main():
    """
    align and save the most similar person with the cropped_aligned
    """
    videos = list_files('Aff-Wild2/cropped_aligned')




if __name__ == '__main__':
    os.makedirs('Aff-Wild2/aligned2', exist_ok=True)
    videos = list_files('Aff-Wild2/cropped_aligned')
    # patch_size = len(videos) // 7 + 1
    # sub_videos = videos[gpu_id * patch_size : (gpu_id + 1) * patch_size]
    # sub_videos = sub_videos[::-1]

    sub_videos = videos
    print('sub video num:', len(sub_videos))

    # process_one_video('6-30-1920x1080_left')

    with ctx.Pool(process_num) as p:
        p.map(process_one_video, sub_videos)

