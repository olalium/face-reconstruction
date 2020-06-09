import argparse
import os
import random
from os import path

import face3d
import numpy as np
import skimage.transform
from face3d.face3d.morphable_model import MorphabelModel
from Dataset.dataset_generator.facegen_to_posmap import generate_posmap_facegen_bfm
from Dataset.dataset_generator.pos_map_code import process_uv, run_posmap_300W_LP
from skimage import io

from Dataset.dataset_reader.DatasetReader import DatasetReader


def get_front_and_side_poses(face):
    front_poses = []
    side_poses = []
    for face_pose in face.face_poses:
        if face_pose.pose > 45. or face_pose.pose < -45.:
            side_poses.append(face_pose)
        elif face_pose.pose < 45. and face_pose.pose > -45.:
            front_poses.append(face_pose)
        else:
            continue
    return front_poses, side_poses


def generate_300WLP_dataset(root_300wlp_folder, save_dataset_folder, save_datasetlabel):
    uv_coords = face3d.morphable_model.load.load_uv_coords('../Data/BFM/Out/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    bfm = MorphabelModel('../Data/BFM/Out/BFM.mat')

    datasetReader = DatasetReader(root_300wlp_folder)
    face_list = datasetReader.getFacesFromDataset()

    fp_label = open(save_datasetlabel, "w")
    len_face_list = len(face_list)
    for i, face in enumerate(face_list):
        print('generating data for face: %s\t(%d/%d)' % (face.face_id, i + 1, len_face_list))
        save_folder = os.path.join(save_dataset_folder, face.dataset_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        face.sort_face_poses()

        front_poses, side_poses = get_front_and_side_poses(face)

        if not front_poses or not side_poses:
            print('passing, no fitting pose pairs')
            continue

        for face_pose in front_poses:
            side_pose = random.choice(side_poses)
            save_img_path_side = side_pose.img_path.replace(root_300wlp_folder, save_dataset_folder)
            mat_path = face_pose.mat_path
            img_path = face_pose.img_path
            save_img_path = img_path.replace(root_300wlp_folder, save_dataset_folder)
            save_npy_path = save_img_path.replace('jpg', 'npy')

            if os.path.exists(save_img_path) and os.path.exists(save_npy_path):
                fp_label.writelines(save_img_path + ' ' + save_img_path_side + ' ' + save_npy_path + '\n')
                print('passing, posmap already generated')
                continue

            fp_label.writelines(save_img_path + ' ' + save_img_path_side + ' ' + save_npy_path + '\n')
            run_posmap_300W_LP(bfm, uv_coords, img_path, mat_path, save_folder)
    fp_label.close()


def get_face_save_path(face, root_dataset_folder, save_dataset_folder):
    face_save_path = face.img_path.replace(root_dataset_folder, save_dataset_folder)
    face_path_list = face_save_path.split('/')
    face_save_path = os.path.join(face_path_list[0], face_path_list[1], face_path_list[2], face_path_list[4])
    return face_save_path


def get_random_subfolder(folder_path, is_image=False):
    dirs = np.array(os.listdir(folder_path))
    if is_image:
        valid_dirs_index = [i for i, item in enumerate(dirs) if 'jpg' in item or 'png' in item]
        dirs = dirs[valid_dirs_index]
    random_dir = np.random.choice(dirs)
    random_dir_path = os.path.join(folder_path, random_dir)
    return random_dir_path


def apply_random_background(image):
    dtd_path = '../Data/dtd/images'  # texture dataset
    random_dir_path = get_random_subfolder(dtd_path)  # random category
    random_img_path = get_random_subfolder(random_dir_path, is_image=True)  # random image within category

    bg_img = io.imread(random_img_path)
    if (bg_img.shape[0] < 256 or bg_img.shape[
        1] < 256):  # should not happend according to texture dataset specifications
        return image  # return image without background image if this is the case
    image_with_bg = image[:, :, :3].copy()
    cropped_bg_img = bg_img[:256, :256, :3].copy()  # only use the top left 256x256 pixels
    background_mask = np.array(image_with_bg <= [0, 0, 0])  # if rgb values are 0, background should be shown
    image_with_bg[background_mask] = cropped_bg_img[background_mask]

    return image_with_bg


def generate_facegen_dataset(root_dataset_folder, save_dataset_folder, save_dataset_label):
    bfm = MorphabelModel('../Data/BFM/Out/BFM.mat')
    uv_coords = face3d.morphable_model.load.load_uv_coords('../Data/BFM/Out/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    side_img_cropping_tform = skimage.transform.AffineTransform(scale=(0.8, 0.8))
    image_h, image_w = 256, 256

    datasetReader = DatasetReader(root_dataset_folder, dataset_name='facegen')
    face_list = datasetReader.getFacesFromDataset()
    len_face_list = len(face_list)
    fp_label = open(save_dataset_label, "w")
    for i, face in enumerate(face_list):
        print('generating data for face: %s\t(%d/%d)' % (face.face_id, i + 1, len_face_list))
        save_folder = os.path.join(save_dataset_folder, face.dataset_name)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # set faces and set save paths for the images
        front_face = face.face_poses[0]
        left_face = face.face_poses[1]  # left
        right_face = face.face_poses[2]  # right
        front_face_save_path = get_face_save_path(front_face, root_dataset_folder, save_dataset_folder)
        left_face_save_path = get_face_save_path(left_face, root_dataset_folder, save_dataset_folder)
        right_face_save_path = get_face_save_path(right_face, root_dataset_folder, save_dataset_folder)

        # get the base mesh path and dataset posmap path
        obj_path = front_face.img_path.replace('_front.png', '.obj')
        pos_map_path = front_face_save_path.replace('png', 'npy')

        # if posmap already exists, skip
        if os.path.exists(front_face_save_path) and os.path.exists(left_face_save_path) and os.path.exists(
                right_face_save_path) and os.path.exists(pos_map_path):
            fp_label.write(front_face_save_path + ' ' + left_face_save_path + ' ' + pos_map_path + '\n')
            fp_label.write(front_face_save_path + ' ' + right_face_save_path + ' ' + pos_map_path + '\n')
            print('passing')
            continue

        # generate posmap and get cropping transforms
        front_img_cropping_tform = generate_posmap_facegen_bfm(bfm, uv_coords, obj_path, pos_map_path, save_image=True)

        # read image
        front_image = io.imread(front_face.img_path)
        left_image = io.imread(left_face.img_path)
        right_image = io.imread(right_face.img_path)

        # crop images
        front_image_cropped = skimage.transform.warp(front_image, front_img_cropping_tform.inverse,
                                                     output_shape=(image_h, image_w), preserve_range=True)
        left_image_cropped = skimage.transform.warp(left_image, side_img_cropping_tform,
                                                    output_shape=(image_h, image_w), preserve_range=True)
        right_image_cropped = skimage.transform.warp(right_image, side_img_cropping_tform,
                                                     output_shape=(image_h, image_w), preserve_range=True)

        front_image_cropped = front_image_cropped.astype(int)
        left_image_cropped = left_image_cropped.astype(int)
        right_image_cropped = right_image_cropped.astype(int)

        # apply random backgrounds to images and save them in target dataset
        front_image = apply_random_background(front_image_cropped)
        left_image = apply_random_background(left_image_cropped)
        right_image = apply_random_background(right_image_cropped)

        io.imsave(front_face_save_path, front_image, check_contrast=False)
        io.imsave(left_face_save_path, left_image, check_contrast=False)
        io.imsave(right_face_save_path, right_image, check_contrast=False)

        # update dataset label
        fp_label.write(front_face_save_path + ' ' + left_face_save_path + ' ' + pos_map_path + '\n')
        fp_label.write(front_face_save_path + ' ' + right_face_save_path + ' ' + pos_map_path + '\n')
    fp_label.close()


def main(args):
    if (args.dataset == 'facegen'):
        print('using facegen dataset')
        assert (path.exists(args.dataset_path_input))
        generate_facegen_dataset(args.dataset_path_input, args.dataset_path_output, args.dataset_path_label)
    elif (args.dataset == '300wlp'):
        print('using 300w-lp dataset')
        assert (path.exists(args.dataset_path_input))
        generate_300WLP_dataset(args.dataset_path_input, args.dataset_path_output, args.dataset_path_label)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset generator for 2D images to 3D position map training pairs')

    parser.add_argument('--dataset', default='facegen', type=str,
                        help='specify which image dataset to generate position maps from, facegen or 300wlp')
    parser.add_argument('--dataset_path_input', default='../Data/FACEGEN_DB_10K', type=str,
                        help='path to input dataset')
    parser.add_argument('--dataset_path_output', default='../results/facegen_train_dataset', type=str,
                        help='path to output dataset')
    parser.add_argument('--dataset_path_label', default='../results/facegen_dataset_label.txt', type=str,
                        help='path to resulting dataset label file')

    main(parser.parse_args())
    # generate_facegen_dataset('../Data/FACEGEN_DB_NEW_1K', '../results/facegen_train_dataset', '../results/facegen_dataset_label.txt')
