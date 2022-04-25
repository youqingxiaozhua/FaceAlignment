
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

import face_recognition

from align_trans import get_reference_facial_points, warp_and_crop_face

crop_size = 224
scale = crop_size / 112.
reference = get_reference_facial_points(default_square = True) * scale

fnt = ImageFont.truetype("DejaVuSans.ttf", size=40)


mtcnn = MTCNN(keep_all=True)
inception = InceptionResnetV1().eval()
pretrained = torch.load('20180408-102900-casia-webface.pt')
inception.load_state_dict(pretrained, strict=False)


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
        x = inception(x)
    return x.detach().cpu()


def align_one_image(im_path):
    ref_face = Image.open('00007.jpg').resize((160, 160))
    ref_feature = extract_embedding(ref_face)
    # ref_face = face_recognition.load_image_file("00007.jpg")
    # ref_face = face_recognition.load_image_file("03052.jpg")
    # ref_feature = face_recognition.face_encodings(ref_face)[0]

    img = Image.open(im_path)
    boxes, probs, points = mtcnn.detect(img, landmarks=True)
    print(probs)
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    for i, (box, point) in enumerate(zip(boxes, points)):
        draw.rectangle(box.tolist(), width=3)
        for p in point:
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=3)
        crop_face = extract_face(img, box)
        crop_face_feature = extract_embedding(crop_face)
        distance = (ref_feature - crop_face_feature).norm().item()

        # crop_face = crop_face.permute(1, 2, 0).type(torch.uint8).numpy()
        # bbox = list(map(int, box.tolist()))
        # # convert to (top, right, bottom, left)
        # bbox = [bbox[1], bbox[2], bbox[3], bbox[0]]
        # crop_face_feature = face_recognition.face_encodings(crop_face, known_face_locations=[bbox])
        # if len(crop_face_feature) == 0:
        #     print(crop_face_feature)
        #     continue
        # print(crop_face_feature[0][:3])
        # distance = np.linalg.norm(ref_feature - crop_face_feature[0])

        print('distance', distance)

        draw.text(box[:2], str(distance), font=fnt)
    img_draw.save('annotated_faces.jpg')

    # align
    warped_face = warp_and_crop_face(np.array(img), points[0], reference, crop_size=(crop_size, crop_size))
    img_warped = Image.fromarray(warped_face)
    img_warped.save('aligned_face.jpg')



if __name__ == '__main__':
    align_one_image('Aff-Wild2/images/5-60-1920x1080-4.mp4/00375.jpg')
    # align_one_image('Aff-Wild2/images/6-30-1920x1080.mp4/02696.jpg')  # left, right

