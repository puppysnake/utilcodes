# !/bin/python
import os, sys
import glob
import os.path as osp

import math
import numpy as np


def get_orig_imgslist(imgs_dirpath, img_suffix='jpg', name_sep=None):
    imgs_lststr = osp.join(imgs_dirpath, '*.' + img_suffix)
    imgs_paths = glob.glob(imgs_lststr)

    n_imgs = len(imgs_paths)
    img_names = [osp.basename(img_path).split('.'+img_suffix)[0] \
                 for img_path in imgs_paths]

    if name_sep:
        img_prefixs = [img_name.split(name_sep)[0] for img_name in img_names]
        prefix_to_nms = dict()

        for img_prefix, img_name in zip(img_prefixs, img_names):
            if not img_prefix in list(prefix_to_nms.keys()):
                prefix_to_nms[img_prefix] = [img_name]
            else:
                prefix_to_nms[img_prefix].append(img_name)

        return imgs_paths, img_names, prefix_to_nms

    return imgs_paths, img_names


def get_trts_imgslist(img_names, split_ratio=0.8):
    n_elements = len(img_names)
    n_trn_elements = np.round(split_ratio * n_elements).astype(int)

    smpl_idxs = np.arange(n_elements)
    np.random.shuffle(smpl_idxs)

    trn_img_nms = [img_names[idx] for idx in smpl_idxs[:n_trn_catgrs]]
    tst_img_nms = [img_names[idx] for idx in smpl_idxs[n_trn_catgrs:]]

    return trn_img_nms, tst_img_nms


def get_trts_catgrlist(prefix_to_nms, split_ratio=0.8):
    n_all_catgrs = len(list(prefix_to_nms.keys()))
    n_trn_catgrs = np.round(split_ratio * n_all_catgrs).astype(int)

    smpl_idxs = np.arange(n_all_catgrs)
    np.random.shuffle(smpl_idxs)

    trn_catgr_idxs = smpl_idxs[:n_trn_catgrs]
    tst_catgr_idxs = smpl_idxs[n_trn_catgrs:]

    catgr_nms_list = list(prefix_to_nms.values())
    trn_img_nms = np.concatenate([catgr_nms_list[catrgr_idx] for catrgr_idx in trn_catgr_idxs])
    tst_img_nms = np.concatenate([catgr_nms_list[catrgr_idx] for catrgr_idx in tst_catgr_idxs])

    return trn_img_nms, tst_img_nms


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='resplit training / testing images.')
    parser.add_argument('imgsdir', type=str, help='path of the training / testing images')
    parser.add_argument('--suffix', type=str, default='jpg', help='image suffix string')
    parser.add_argument('--trratio', type=float, default=0.8, help='training images quantity ratio')
    parser.add_argument('--nmcatgrchar', type=str, default=None, help='image name prefix separator')

    args = parser.parse_args()

    if args.nmcatgrchar:
        imgs_paths, img_names, prefix_to_nms = get_orig_imgslist(args.imgsdir, args.suffix, args.nmcatgrchar)
        tr_imnms, ts_imnms = get_trts_catgrlist(prefix_to_nms, args.trratio)

    else:
        imgs_paths, img_names = get_orig_imgslist(args.imgsdir, args.suffix, args.nmcatgrchar)
        tr_imnms, ts_imnms = get_trts_catgrlist(img_names, args.trratio)

    import pdb; pdb.set_trace()
    # command: python dataproc_split_traintestimgs.py ./dataset/transform_for_maskrcnn/JPEGImages/ --suffix 'jpg' --nmcatgrchar '_'
