'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw
import math
import torch.nn.functional as F
import json

import opts
from tqdm import tqdm

import multiprocessing as mp
import threading

from tools.colormap import colormap

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def main(args):
    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1")

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    split = args.split
    # save path
    output_dir = args.output_dir
    save_path_prefix = os.path.join(output_dir, split)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    save_visualize_path_prefix = os.path.join(output_dir, split + '_images')
    if args.visualize:
        if not os.path.exists(save_visualize_path_prefix):
            os.makedirs(save_visualize_path_prefix)

    # load data
    # root = Path(args.ytvos_path)  # data/ref-youtube-vos
    # root = "/ssd1/luojingnan/ytvos"
    # img_folder = os.path.join(root, split, "JPEGImages")
    # meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    # with open(meta_file, "r") as f:
    #     data = json.load(f)["videos"]
    # valid_test_videos = set(data.keys())
    # # for some reasons the competition's validation expressions dict contains both the validation (202) &
    # # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # # the validation expressions dict:
    # test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    # with open(test_meta_file, 'r') as f:
    #     test_data = json.load(f)['videos']
    # test_videos = set(test_data.keys())
    # valid_videos = valid_test_videos - test_videos
    # video_list = sorted([video for video in valid_videos])
    # assert len(video_list) == 202, 'error: incorrect number of validation videos'

    data_root_path = "/ssd1/luojingnan/LaSOTTest/LaSOTTest"
    output_path = "/ssd1/luojingnan/LaSOT_Referformer_output_frame_batch_50/Referformer_tracking_result"

    video_list = [video for video in os.listdir(data_root_path) if video not in [i[:-4] for i in os.listdir(output_path)]]
    # video_list = ['basketball-1']
    print(video_list)
    print(len(video_list))

    # create subprocess
    thread_num = args.ngpu
    # thread_num = 1
    global result_dict
    result_dict = mp.Manager().dict()

    processes = []
    lock = threading.Lock()

    video_num = len(video_list)
    per_thread_video_num = video_num // thread_num

    start_time = time.time()
    print('Start inference')
    for i in range(thread_num):
        if i == thread_num - 1:
            sub_video_list = video_list[i * per_thread_video_num:]
        else:
            sub_video_list = video_list[i * per_thread_video_num: (i + 1) * per_thread_video_num]

        p = mp.Process(target=sub_processor, args=(lock, i, args,
                                                   save_path_prefix, save_visualize_path_prefix,
                                                   sub_video_list, data_root_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    end_time = time.time()
    total_time = end_time - start_time

    result_dict = dict(result_dict)
    num_all_frames_gpus = 0
    for pid, num_all_frames in result_dict.items():
        num_all_frames_gpus += num_all_frames

    print("Total inference time: %.4f s" % (total_time))


def sub_processor(lock, pid, args, save_path_prefix, save_visualize_path_prefix, video_list, data_root_path):
    frame_batch_size = 50
    data_num = len(video_list)

    text = 'processor %d' % pid
    with lock:
        progress = tqdm(
            total=data_num,
            position=pid,
            desc=text,
            ncols=0
        )
    torch.cuda.set_device(pid)

    # model
    model, criterion, _ = build_model(args)
    device = args.device
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if pid == 0:
        print('number of params:', n_parameters)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    # start inference
    num_all_frames = 0
    model.eval()

    # 1. For each video
    for sub_path in video_list:
        print(f'{sub_path} start!')
        metas = []  # list[dict], length is number of expressions

        data_path = os.path.join(data_root_path, sub_path)
        video_path = os.path.join(data_path, "img")
        expression_path = os.path.join(data_path, "nlp.txt")

        with open(expression_path, 'r') as cf:
            expression = cf.readline()

        # print(expression)
        # 一个视频的目标有一个expression
        video_len = len(os.listdir(video_path))
        frame_path = os.listdir(video_path)
        frame_path.sort()

        # store images

        batch_idx = 0

        pred_logits = None
        pred_boxes = None
        pred_masks = None
        pred_ref_points = None

        # print(video_len)
        while batch_idx * frame_batch_size < video_len:
            imgs = []
            for frame_name in frame_path[batch_idx * frame_batch_size: (batch_idx + 1) * frame_batch_size]:
                # print(frame_name)
                img_path = os.path.join(video_path, frame_name)
                img = Image.open(img_path).convert('RGB')
                origin_w, origin_h = img.size
                imgs.append(transform(img))  # list[img]

            imgs = torch.stack(imgs, dim=0).to(args.device)  # [video_len, 3, h, w]
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
            # target即图片高度、宽度
            target = {"size": size}

            with torch.no_grad():
                # outputs是dict, keys包括[pred_logits，pred_boxes，pred_masks，reference_points]
                outputs = model([imgs], [expression], [target])

            # pred_logits.shape：batch_size * video_len * query_num * 1
            # pred_boxes.shape：batch_size * video_len * query_num * 4
            # pred_masks.shape：batch_size * video_len * query_num * h * w
            # reference_points.shape：batch_size * video_len * query_num * 2

            if pred_logits is None:
                pred_logits = outputs["pred_logits"][0]
                pred_boxes = outputs["pred_boxes"][0]
                # pred_masks = outputs["pred_masks"][0]
                # pred_ref_points = outputs["reference_points"][0]
            else:
                pred_logits = torch.vstack((pred_logits, outputs["pred_logits"][0]))
                pred_boxes = torch.vstack((pred_boxes, outputs["pred_boxes"][0]))
                # pred_masks = torch.vstack((pred_masks, outputs["pred_masks"][0]))
                # pred_ref_points = torch.vstack((pred_ref_points, outputs["reference_points"][0]))

            # print(pred_logits.shape)
            # print(pred_boxes.shape)
            # print(pred_masks.shape)
            # print(pred_ref_points.shape)

            batch_idx += 1

        # print(pred_logits.shape)
        # print(pred_boxes.shape)
        # print(pred_masks.shape)
        # print(pred_ref_points.shape)

        # according to pred_logits, select the query index
        pred_scores = pred_logits.sigmoid()  # [t, q, k]
        pred_scores = pred_scores.mean(0)  # [q, k]
        max_scores, _ = pred_scores.max(-1)  # [q,]
        _, max_ind = max_scores.max(-1)  # [1,]
        max_inds = max_ind.repeat(video_len)
        # pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
        # pred_masks = pred_masks.unsqueeze(0)

        # 上采样恢复原图大小
        # pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
        # pred_masks = (pred_masks.sigmoid() > args.threshold).squeeze(0).detach().cpu().numpy()

        # store the video results
        all_pred_logits = pred_logits[range(video_len), max_inds]
        all_pred_boxes = pred_boxes[range(video_len), max_inds]
        # all_pred_ref_points = pred_ref_points[range(video_len), max_inds]
        # all_pred_masks = pred_masks

        if args.visualize:
            for t, frame in enumerate(frame_path):
                # original
                img_path = os.path.join(video_path, frame)
                source_img = Image.open(img_path).convert('RGBA')  # PIL image

                draw = ImageDraw.Draw(source_img)
                draw_boxes = all_pred_boxes[t].unsqueeze(0)
                draw_boxes = rescale_bboxes(draw_boxes.detach(), (origin_w, origin_h)).tolist()

                score = all_pred_logits[t].unsqueeze(0)

                # draw boxes
                xmin, ymin, xmax, ymax = draw_boxes[0]
                # print(f'{xmin}, {ymin}, {xmax}, {ymax}')
                # print(f'{xmin}, {ymin}, {xmax - xmin}, {ymax - ymin}')
                # print("------------------------------------")
                draw.rectangle(((xmin, ymin), (xmax, ymax)), outline=tuple(color_list[0 % len(color_list)]),
                               width=2)

                # # draw reference point
                # ref_points = all_pred_ref_points[t].unsqueeze(0).detach().cpu().tolist()
                # draw_reference_points(draw, ref_points, source_img.size, color=color_list[0 % len(color_list)])
                #
                # # draw mask
                # source_img = vis_add_mask(source_img, all_pred_masks[t], color_list[0 % len(color_list)])

                # save
                # save_visualize_path_dir = os.path.join(save_visualize_path_prefix, sub_path)
                # if not os.path.exists(save_visualize_path_dir):
                #     os.makedirs(save_visualize_path_dir)
                # save_visualize_path = os.path.join(save_visualize_path_dir, frame[:-4] + '.png')
                # source_img.save(save_visualize_path)

                save_box_path_dir = os.path.join(args.output_dir, "Referformer_tracking_result")
                if not os.path.exists(save_box_path_dir):
                    os.makedirs(save_box_path_dir)
                save_box_path = os.path.join(save_box_path_dir, sub_path + '.txt')
                box_str = "%.2f,%.2f,%.2f,%.2f\n" % (xmin, ymin, xmax - xmin, ymax - ymin)
                with open(save_box_path, 'a') as fw:
                    fw.write(box_str)

                save_score_path_dir = os.path.join(args.output_dir, "Referformer_tracking_score")
                if not os.path.exists(save_score_path_dir):
                    os.makedirs(save_score_path_dir)
                save_score_path = os.path.join(save_score_path_dir, sub_path + '.txt')
                score_str = f'{score.item()}\n'
                with open(save_score_path, 'a') as fw:
                    fw.write(score_str)

            # # save binary image
            # save_path = os.path.join(save_path_prefix, sub_path)
            # if not os.path.exists(save_path):
            #     os.makedirs(save_path)
            # for j in range(video_len):
            #     frame_name = frame_path[j][:-4]
            #     mask = all_pred_masks[j].astype(np.float32)
            #     mask = Image.fromarray(mask * 255).convert('L')
            #     save_file = os.path.join(save_path, frame_name + ".png")
            #     mask.save(save_file)

        print(f'{sub_path} completed!')
        with lock:
            progress.update(1)
    result_dict[str(pid)] = num_all_frames
    with lock:
        progress.close()


# visuaize functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu() * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


# Visualization functions
def draw_reference_points(draw, reference_points, img_size, color):
    W, H = img_size
    for i, ref_point in enumerate(reference_points):
        init_x, init_y = ref_point
        x, y = W * init_x, H * init_y
        cur_color = color
        draw.line((x - 10, y, x + 10, y), tuple(cur_color), width=4)
        draw.line((x, y - 10, x, y + 10), tuple(cur_color), width=4)


def draw_sample_points(draw, sample_points, img_size, color_list):
    alpha = 255
    for i, samples in enumerate(sample_points):
        for sample in samples:
            x, y = sample
            cur_color = color_list[i % len(color_list)][::-1]
            cur_color += [alpha]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2),
                         fill=tuple(cur_color), outline=tuple(cur_color), width=1)


def vis_add_mask(img, mask, color):
    origin_img = np.asarray(img.convert('RGB')).copy()
    color = np.array(color)

    mask = mask.reshape(mask.shape[0], mask.shape[1]).astype('uint8')  # np
    mask = mask > 0.5

    origin_img[mask] = origin_img[mask] * 0.5 + color * 0.5
    origin_img = Image.fromarray(origin_img)
    return origin_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
