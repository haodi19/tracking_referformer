import os
import numpy as np
import torch

# l = os.listdir("/ssd1/luojingnan/ytvos/test")
# print(l)

# a = np.zeros((2,10,100))
# b = np.zeros((5,10,100))
# c = np.vstack((a,b))
#
# a2 = torch.zeros((2,10,100))
# b2 = torch.zeros((5,10,100))
# c2 = torch.vstack((a2,b2))
# print(c2.shape)
# print(type(c2))
#
# box_str = "%.2f,%.2f,%.2f,%.2f\n" % (1.777,6,8,5)
# print(box_str)
# with open("a.txt", 'a') as fw:
#     fw.write(box_str)

data_root_path = "/ssd1/luojingnan/LaSOTTest/data"
output_path = "/ssd1/luojingnan/LaSOT_Referformer_output_frame_batch_50/Referformer_tracking_result"

print(len(os.listdir(output_path)))


#
# video_list = [video for video in os.listdir(data_root_path) if video not in os.listdir(output_path)]
#
# print(len(os.listdir(data_root_path)))
# print(len(video_list))
# print("book-10" in os.listdir(data_root_path))
# print("book-10" in video_list)

def check_result_integrity():
    result_path = "/ssd1/luojingnan/LaSOT_Referformer_output_frame_batch_50/Referformer_tracking_result"
    gt_path = "/ssd1/luojingnan/LaSOTTest/LaSOTTest"
    current_videos = os.listdir(result_path)
    current_videos.sort()

    incomplete_results = []

    for current_video in current_videos:
        current_video_result_path = os.path.join(result_path, current_video)
        with open(current_video_result_path, 'r') as f1:
            result_lines = len(f1.readlines())

        current_video_gt_path = os.path.join(gt_path, current_video[:-4], "groundtruth.txt")
        with open(current_video_gt_path, 'r') as f2:
            gt_lines = len(f2.readlines())
            # gt_lines = gt_lines // 60 if gt_lines % 60 == 0 else gt_lines // 60 + 1

        if result_lines != gt_lines:
            incomplete_results.append((current_video, result_lines, gt_lines))

    print(incomplete_results)


check_result_integrity()
