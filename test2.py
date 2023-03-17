import os

gt_path = '/ssd1/luojingnan/TNL2K_train_subset/train_data/Arrow_Video_ZZ04_done/groundtruth.txt'

with open(gt_path, 'r') as lf:
    # (x1,y1,width,hidth)
    gt = lf.readlines()
    print(gt[0].rstrip().split(','))