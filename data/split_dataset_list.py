import argparse
import glob
import os.path as osp

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='A tool for proportionally randomizing dataset to produce file lists.')
    parser.add_argument('dataset_root', help='the dataset root path', type=str)
    parser.add_argument('--split', help='', nargs=3, type=float, default=[0.7, 0.3, 0])
    
    return parser.parse_args()


def get_files(path, file_format):
    pattern = '*.%s' % file_format.lower()
    search_files = osp.join(path, pattern)
    filenames = glob.glob(search_files)
    return sorted(filenames)


def generate_list(args):
    def get_filename_no_ext(path):
        return osp.splitext(osp.basename(path))[0]
    
    dataset_root = args.dataset_root
    if abs(sum(args.split) - 1.0) > 1e-8:
        raise ValueError("The sum of input params `--split` should be 1")
    
    image_dir = osp.join(dataset_root, "images")
    anno_dir = osp.join(dataset_root, "annotations")
    image_files = get_files(image_dir, "*")
    anno_files = get_files(anno_dir, "json")
    
    if not anno_files:
        raise FileNotFoundError("目录中无标注json文件，请先创建标注文件: {}".format(anno_dir))
    
    num_images = len(image_files)
    num_annos = len(anno_files)
    if num_images != num_annos:
        raise Exception("样本数量 = {}, 标注数量 = {}. 数量不一致, 请检查数据集!".format(num_images, num_annos))
    image_files = np.array(image_files)
    anno_files = np.array(anno_files)
    
    state = np.random.get_state()
    np.random.shuffle(image_files)
    np.random.set_state(state)  # 随机种子一致，对齐打乱后的样本与标注列表
    np.random.shuffle(anno_files)
    
    start = 0
    num_split = len(args.split)
    dataset_name = ['train', 'val', 'test']
    for i in range(num_split):
        dataset_split = dataset_name[i]
        print("Creating {}.txt...".format(dataset_split))
        if args.split[i] > 1.0 or args.split[i] < 0:
            raise ValueError("{} dataset percentage should be 0~1.".format(dataset_split))
        
        file_list = osp.join(dataset_root, dataset_split + '.txt')
        with open(file_list, "w") as f:
            num = round(args.split[i] * num_images)
            end = start + num
            if i == num_split - 1:
                end = num_images
            split_list = anno_files[start:end].tolist()
            filename_list = list(map(lambda s: str(get_filename_no_ext(s)) + "\n", split_list))
            f.writelines(filename_list)
            print(filename_list)
            start = end


if __name__ == '__main__':
    args = parse_args()
    generate_list(args)
