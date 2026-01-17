import argparse
import glob
import json
import os
import os.path as osp

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Polygon


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--task', choices=['det', 'seg'])
    parser.add_argument('input_dir', help='input annotated directory')
    parser.add_argument('output_dir', help='output annotated directory')
    return parser.parse_args()


def polygon_area(poly_points):
    """输入: [[x1,y1], [x2,y2], ..., [xn,yn]] → 返回像素面积（float）"""
    if len(poly_points) < 3:
        return 0.0
    # 确保顺时针 or 逆时针？Shapely 自动处理有向面积 → 取绝对值
    try:
        poly = Polygon(poly_points)
        if not poly.is_valid:
            # 尝试修复（如自相交）
            poly = poly.buffer(0)
        return float(poly.area) if poly.is_valid else 0.0
    except Exception:
        # 降级：用 Shoelace 公式（更轻量，无需 shapely）
        x = np.array([p[0] for p in poly_points])
        y = np.array([p[1] for p in poly_points])
        return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def get_color_map_list(num_classes):
    num_classes += 1
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = color_map[3:]
    return color_map


def shape2mask(img_size, points):
    label_mask = Image.fromarray(np.zeros(img_size[:2], dtype=np.uint8))
    image_draw = ImageDraw.Draw(label_mask)
    points_list = [tuple(point) for point in points]
    assert len(points_list) > 2, 'Polygon must have points more than 2'
    image_draw.polygon(xy=points_list, outline=1, fill=1)
    return np.array(label_mask, dtype=bool)


def shape2label(img_size, shapes, class_name_mapping):
    label = np.zeros(img_size[:2], dtype=np.int32)
    for shape in shapes:
        points = shape['points']
        class_name = shape['label']
        shape_type = shape.get('shape_type', None)
        class_id = class_name_mapping[class_name]
        label_mask = shape2mask(img_size[:2], points)
        label[label_mask] = class_id
    return label


def get_class_names(labelme_json_dir):
    # collect and save class names
    class_names = ['_background_']
    for label_file in glob.glob(osp.join(labelme_json_dir, '*.json')):
        with open(label_file) as f:
            data = json.load(f)
            for shape in data['shapes']:
                cls_name = shape['label']
                if cls_name not in class_names:
                    class_names.append(cls_name)
    
    return class_names


def get_bbox_loc(loc_list):
    if len(loc_list) not in [2, 4]:  # 2点或4点定位的四边形
        raise ValueError("坐标点数量应为2或4")
    xs, ys = zip(*loc_list)
    return [round(min(xs), 2), round(min(ys), 2), round(max(xs), 2), round(max(ys), 2)]  # xyxy


def write_mask_img(origin_img_path, masks_img_path, shapes, class_name_to_id, color_map):
    # imdecode 解决中文路径
    img = np.asarray(cv2.imdecode(np.fromfile(origin_img_path, dtype=np.uint8), cv2.IMREAD_COLOR_RGB))
    if img is None:
        raise ValueError('无法读取图像数据，检查文件是否存在: {}!'.format(origin_img_path))
    lbl = shape2label(
        img_size=img.shape,
        shapes=shapes,
        class_name_mapping=class_name_to_id
    )
    
    # Assume label ranges [0, 255] for uint8,
    if lbl.min() >= 0 and lbl.max() <= 255:
        lbl_pil = Image.fromarray(lbl.astype(np.uint8), mode='P')
        lbl_pil.putpalette(color_map)
        lbl_pil.save(masks_img_path)
    else:
        raise ValueError(
            '[%s] Cannot save the pixel-wise class label as PNG. '
            'Please consider using the .npy format.' %
            masks_img_path)


def main(args):
    # prepare
    output_dir = args.output_dir
    
    if not osp.exists(output_dir):
        os.makedirs(osp.join(output_dir, 'annotations'), exist_ok=True)
        print('Creating directory:', output_dir)
    
    if args.task == 'seg':
        os.makedirs(osp.join(output_dir, 'masks'), exist_ok=True)  # 创建mask图存放目录
        print("Created mask directory")
        # 创建label与颜色映射
    
    # collect and save class names
    class_names = get_class_names(args.input_dir)
    # ====================== 将类别写入TXT文件 =======================
    class_name_to_id = {}
    for i, class_name in enumerate(class_names):
        class_id = i  # starts with 0
        class_name_to_id[class_name] = class_id
    print('class_names:', class_names)
    
    out_class_names_file = osp.join(output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)
    
    # ===================== 生成标注json文件 =======================
    anno_list = []
    for label_file in glob.glob(osp.join(args.input_dir, '*.json')):
        obj = {}
        with open(label_file, 'r') as f:
            data = json.load(f)
            k = os.path.split(os.path.splitext(data["imagePath"])[0])[1]  # 不带后缀文件名做ID
            base_obj = {
                "id": k,
                "image_name": data["imagePath"],
                "image_width": data["imageWidth"],
                "image_height": data["imageHeight"],
            }
            if args.task == 'det':
                det = {'det': {"bbox": [], "cls": [], "area": []}}
                for item in data["shapes"]:
                    det["det"]["cls"].append(item["label"])
                    det["det"]["bbox"].append(get_bbox_loc(item["points"]))  # xyxy
                    det["det"]["area"].append(polygon_area(item["points"]))
                obj = base_obj | det
            elif args.task == 'seg':
                seg = {'seg': {"mask_path": "", "area": -1}}
                color_map = get_color_map_list(256)
                write_mask_img(
                    osp.join(args.input_dir, data['imagePath']),
                    osp.join(output_dir, 'masks', base_obj['id'] + '.png'),
                    data['shapes'],
                    class_name_to_id, color_map
                )
                seg['seg']['mask_path'] = base_obj['id'] + '.png'
                det["seg"]["area"].append(polygon_area(item["points"]))
                obj = base_obj | seg
            else:
                raise ValueError("task 只能为 [det,seg]，实际为: {}".format(args.task))
        anno_list.append({obj['id']: obj})
    
    # ===================== 批量写入json标注文件 =======================
    for anno in anno_list:
        for k, v in anno.items():
            with open(os.path.join(output_dir, 'annotations', (k + '.json')), 'w') as f:
                json.dump(v, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    args = parse_args()
    main(args)
