import os

import cv2
import dlib
import face3d
import numpy as np
import trimesh
from face3d import mesh
from face3d.morphable_model import MorphabelModel
from facegen_to_posmap import strip_obj_string, get_image_vertices_from_facegen
from pos_map_code import process_uv
from skimage import io
from skimage.io import imread, imsave
from sklearn.neighbors import KDTree

from Training.TrainData import TrainData
from Utils.write import write_obj_with_colors


def generate_keypoints(image, face_detector, shape_predictor):
    detected_faces = face_detector(image, 1)
    if len(detected_faces) == 0:
        print('warning: no detected face')
        return None

    d = detected_faces[
        0].rect  ## only use the first detected face (assume that each input image only contains one face)
    shape = shape_predictor(image, d)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = [shape.part(i).x, shape.part(i).y]

    return coords


def plot_keypoints(pos, img):
    kpt_inds = np.loadtxt('../Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
    kpts_3d = pos[kpt_inds[1, :], kpt_inds[0, :], :]
    kpts_3d = kpts_3d.round().astype('int8')

    for kpt_ind in kpts_3d:
        img[kpt_ind[1]][kpt_ind[0]] = [255, 0, 0]  # draw red at coord
        img[kpt_ind[1] + 1][kpt_ind[0] + 1] = [255, 0, 0]  # draw cross to show keypoints more clearly
        img[kpt_ind[1] + 1][kpt_ind[0] - 1] = [255, 0, 0]
        img[kpt_ind[1] - 1][kpt_ind[0] + 1] = [255, 0, 0]
        img[kpt_ind[1] - 1][kpt_ind[0] - 1] = [255, 0, 0]
    return img


def get_random_subfolder(folder_path):
    dirs = os.listdir(folder_path)
    random_dir = np.random.choice(dirs)
    random_dir_path = os.path.join(folder_path, random_dir)
    return random_dir_path


def apply_random_background(image):
    dtd_path = '../Data/dtd/images'  # texture dataset
    random_dir_path = get_random_subfolder(dtd_path)  # random category
    random_img_path = get_random_subfolder(random_dir_path)  # random image within category

    random_img = io.imread(random_img_path)
    img = image[:, :, :3].copy()
    rnd_img = random_img[:256, :256, :3].copy()
    background_mask = np.array(img == [0, 0, 0])
    img[background_mask] = rnd_img[background_mask]

    return img


def random_background():
    facegen_img_path = '../test_images_facegen/00001/00001_front.png'
    image = io.imread(facegen_img_path)
    image = apply_random_background(image)
    io.imsave(facegen_img_path.replace('_front', '_front_back'), image)


def export_vertices_from_pos(pos, out_path='pos_vertices.ply'):
    vertices = np.reshape(pos, [-1, 3])
    print(vertices.shape)
    mesh = trimesh.Trimesh(vertices=vertices)
    mesh.export(out_path)


def export_keypoints_from_fg(vertices):
    l68_ind_fg = np.loadtxt('../Data/uv-data/00011_l68.txt', dtype=int)
    kpts_fg = vertices[l68_ind_fg[:]]
    mesh = trimesh.Trimesh(vertices=kpts_fg)
    mesh.export(out_path)


def export_keypoints_from_BFM_pos(pos, uv_coords, out_path='kpt_vertices_bfm.ply'):
    bfm_kpt_ind = np.loadtxt('../Data/uv-data/bfm_kpt_ind.txt').astype(np.int32)  # same as bfm.kpt_ind

    kpts_uv = np.array(uv_coords[bfm_kpt_ind]).astype(np.int32).T[:2]
    kpts = pos[kpts_uv[1], kpts_uv[0], :]
    mesh = trimesh.Trimesh(vertices=kpts)
    mesh.export(out_path)


def export_keypoints_from_BFM_and_fg(pos_bfm, fg_obj_path, uv_coords, out_path_bfm='kpt_vertices_bfm.ply',
                                     out_path_fg='kpt_vertices_fg.ply'):
    # BFM
    bfm_kpt_ind = np.loadtxt('../Data/uv-data/bfm_kpt_ind.txt').astype(np.int32)  # same as bfm.kpt_ind
    bfm_kpt_ind = bfm_kpt_ind[3:]
    bfm_kpt_ind = np.concatenate((bfm_kpt_ind[:11], bfm_kpt_ind[14:]))
    kpts_uv = np.array(uv_coords[bfm_kpt_ind]).astype(np.int32).T[:2]
    kpts = pos_bfm[kpts_uv[1], kpts_uv[0], :]
    print(kpts.shape)
    mesh = trimesh.Trimesh(vertices=kpts)
    mesh.export(out_path_bfm)

    # FG
    with open(fg_obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -2)
    fg_kpt_ind = np.loadtxt('../Data/uv-data/00011_l68.txt', dtype=int)
    fg_kpt_ind = fg_kpt_ind[3:]
    fg_kpt_ind = np.concatenate((fg_kpt_ind[:11], fg_kpt_ind[14:]))
    vertices_fg = vertices_array[fg_kpt_ind[:]]
    print(vertices_fg.shape)
    mesh = trimesh.Trimesh(vertices=vertices_fg)
    mesh.export(out_path_fg)


def fit_facegen_with_BFM(bfm, obj_path):
    with open(obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -2)

    fg_vertices = get_image_vertices_from_facegen(vertices_array, obj_path.replace('.obj', '_front.png'))  # , True)

    # fg_kpt_ind = np.loadtxt('../Data/uv-data/00011_l68.txt', dtype=int)
    fg_kpts_ind = np.loadtxt('../Data/uv-data/kpt_ind_fg_anim.txt', dtype=int)
    fg_kpts = fg_vertices[fg_kpts_ind]
    '''
    face_detector_path = '../Data/net-data/mmod_human_face_detector.dat'
    shape_predictor_path = '../Data/net-data/shape_predictor_68_face_landmarks.dat'
    img_path = obj_path.replace('.obj', '_front.png')
    img = imread(img_path, mode='RGB')[:,:,:3]

    face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    shape_predictor = dlib.shape_predictor(shape_predictor_path)
    kpts = generate_keypoints(img, face_detector, shape_predictor)
    print(kpts)
    '''

    # kpts = to_image(kpts, 512, 512)
    # print(kpts)

    # bfm_kpt_ind = np.loadtxt('../Data/uv-data/bfm_kpt_ind.txt').astype(np.int32) # same as bfm.kpt_ind
    sp, ep, s, angles, t = bfm.fit(fg_kpts[:31, :2], bfm.kpt_ind[:31], max_iter=4)

    vertices = bfm.generate_vertices(sp, ep)

    transformed_vertices = bfm.transform(vertices, s, angles, t)

    mesh = trimesh.Trimesh(vertices=transformed_vertices)
    mesh.export('fitted_vertices_bfm.ply')

    mesh = trimesh.Trimesh(vertices=transformed_vertices[bfm.kpt_ind[:31]])
    mesh.export('fitted_vertices_kpts.ply')

    mesh = trimesh.Trimesh(vertices=fg_kpts[:31, :])
    mesh.export('kpts_fg_anim.ply')


def strip_obj_string(lines, depth, ind_start, ind_end):
    '''
    iterates over read obj lines and produces a numpy array with vertices in [x,y,z] format
    '''
    array = np.zeros((len(lines), depth))
    for i, line in enumerate(lines):
        sub_array = np.array(line[ind_start:ind_end].split(' ')).astype(np.float32)
        array[i] = sub_array
    return array


def get_l68_vertex_ind_for_keypoints(img_path, pos_path):
    face_detector_path = '../Data/net-data/mmod_human_face_detector.dat'
    shape_predictor_path = '../Data/net-data/shape_predictor_68_face_landmarks.dat'

    face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    image = imread(img_path, mode='RGB')[:, :, :3]

    pos = np.load(pos_path)

    detected_faces = face_detector(image, 1)
    d = detected_faces[0].rect
    shape = shape_predictor(image, d)

    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    vertices = np.reshape(pos, [-1, 3])[:, :2]

    v_tree = KDTree(vertices)

    for (x, y) in coords:
        dst, ind = v_tree.query([[x, y]], k=1)
        print(x, y, ind[0][0], vertices[ind[0][0]], dst[0][0])


def generate_bfm_mask(uv_face_mask_path, uv_kpt_mask_path, uv_weight_mask_path, result_path):
    uv_face_mask = imread(uv_face_mask_path, as_gray=True)
    uv_kpt_mask = imread(uv_kpt_mask_path, as_gray=True)
    uv_weight_mask = imread(uv_weight_mask_path, as_gray=True)
    final_mask = np.zeros((256, 256))

    print(uv_face_mask[128][100:128])
    print(np.max(uv_weight_mask))
    final_mask = (uv_face_mask[:, :] / 255) * (uv_weight_mask[:, :] / 255)
    final_mask += (uv_kpt_mask[:, :] / 255)
    final_mask = np.clip(final_mask, 0., 1.)
    imsave(result_path, final_mask)


def validate_TrainData():
    batch_size = 16
    data = TrainData(
        '../results/facegen_3dmm_label.txt',
        weight_mask_path='../Data/uv-data/weight_mask_final.jpg',
        # '../Data/uv-data/facegen_final_mask.png',
    )
    data.set_augmentation(True, True, True)
    np.random.shuffle(data.training_data)
    batch = data(batch_size, data.training_index, data.num_training_data, data.training_data)
    data.training_index = data.get_updated_index(data.training_index, batch_size, data.num_training_data)

    img_batch = np.array(batch[0])
    pos_batch = np.array(batch[1])
    print(pos_batch[0].shape)
    print(img_batch[0][128, 60, :3])
    cv2.imwrite('img_batch_0.png', img_batch[0][:, :, :3] * 256)
    # print(pos_batch[0])
    export_vertices_from_pos(pos_batch[0])


def generate_bfm_base_obj():
    bfm = MorphabelModel('../Data/BFM/Out/BFM.mat')
    sp = bfm.get_shape_para('zero')
    ep = bfm.get_exp_para('zero')
    tp = bfm.get_tex_para('random')
    vertices = bfm.generate_vertices(sp, ep)
    colors = bfm.generate_colors(tp)
    colors = np.minimum(np.maximum(colors, 0), 255)
    write_obj_with_colors('head_fit.obj', vertices, bfm.triangles, colors)


def plot_kpts_from_obj(obj_path, uv_coords, bfm, h=256, w=256):
    img_path = obj_path.replace('.obj', '_front.png')
    with open(obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -2)
    image_vertices = get_image_vertices_from_facegen(vertices_array, img_path)

    image_vertices[:, 1] = h - image_vertices[:, 1] - 1
    position = image_vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])

    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, h, w, c=3)
    uv_position_map = uv_position_map.astype(np.float16)

    # save position map
    np.save(obj_path.replace('.obj', '_posmap.npy'), uv_position_map)

    # visualize keypoints
    input_image = imread(img_path)[:, :, :3]
    image = plot_keypoints(uv_position_map, input_image)
    imsave(img_path.replace('.png', '_keypoints.png'), image)


if __name__ == '__main__':
    bfm = MorphabelModel('../Data/BFM/Out/BFM.mat')
    obj_path = '../Data/00001/00001.obj'
    uv_coords = face3d.morphable_model.load.load_uv_coords('../Data/BFM/Out/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)

    plot_kpts_from_obj(obj_path, uv_coords, bfm)

# fit_facegen_with_BFM(bfm, '../Data/FACEGEN_FULL_10/00011/00011.obj')
