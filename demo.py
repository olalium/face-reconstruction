import os

import cv2
import dlib
import numpy as np
from skimage.io import imread, imsave
from skimage.transform import rescale, estimate_transform, warp

from Networks.predictor import MobilenetPosPredictor
from Utils.write import write_obj_with_colors


def mask_pos(pos):
    '''
    remove neck/irrelevant regions
    '''
    mask_path = 'Data/uv-data/facegen_face_mask.png'
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    index_mask = mask[:, :] < 0.5
    masked_pos = pos.copy()
    masked_pos[index_mask] = [0, 0, 0]
    return masked_pos


def plot_vertices_on_image_from_pos(pos, l68, front_img):
    h, w, c = pos.shape
    plotted_front_img = front_img.copy().astype(np.uint8)
    h_i, w_i, c_i = plotted_front_img.shape
    max_h = np.max(pos[:, :, 1])
    max_w = np.max(pos[:, :, 0])
    min_z = int(np.min(pos[:, :, 2]))
    max_z = int(np.max(pos[:, :, 2]))

    if (max_w - w_i) > 0:
        enlarged_front_img = np.zeros((h_i, int(max_w), 3), dtype=np.uint8)
        enlarged_front_img[:, 0:w_i, :] = plotted_front_img[:, :, :]
        plotted_front_img = enlarged_front_img
        h_i, w_i, c_i = plotted_front_img.shape

    if (max_h - h_i) > 0:
        enlarged_front_img = np.zeros((int(max_h), w_i, 3), dtype=np.uint8)
        enlarged_front_img[0:h_i, :, :] = plotted_front_img[:, :, :]
        plotted_front_img = enlarged_front_img
        h_i, w_i, c_i = plotted_front_img.shape

    for h_u in range(h):
        for w_u in range(w):
            index = np.around(pos[h_u][w_u], decimals=1).astype(int)
            plotted_front_img[index[1] - 2][index[0] - 2] = [0, 255 - (max_z - index[2]), index[2]]

    for (x, y) in l68:
        plotted_front_img[y][x] = [255, 0, 0]  # draw red at coord
        plotted_front_img[y + 1][x + 1] = [255, 0, 0]  # draw cross to show keypoints more clearly
        plotted_front_img[y + 1][x - 1] = [255, 0, 0]
        plotted_front_img[y - 1][x + 1] = [255, 0, 0]
        plotted_front_img[y - 1][x - 1] = [255, 0, 0]
    return plotted_front_img


# from PRNet code
def get_cropping_transformation(image, face_detector, shape_predictor):
    detected_faces = face_detector(image, 1)
    if len(detected_faces) == 0:
        print('warning: no detected face')
        return None

    d = detected_faces[
        0].rect  ## only use the first detected face (assume that each input image only contains one face)
    left = d.left();
    right = d.right();
    top = d.top();
    bottom = d.bottom()
    old_size = (right - left + bottom - top) / 2
    center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.14])
    size = int(old_size * 1.58)

    shape = shape_predictor(image, d)
    coords = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, 255], [255, 0]])
    tform = estimate_transform('similarity', src_pts, DST_PTS)
    return coords, tform


def uncrop_pos(cropped_pos, cropping_tform):
    cropped_vertices = np.reshape(cropped_pos, [-1, 3]).T
    z = cropped_vertices[2, :].copy() / cropping_tform.params[0, 0]
    cropped_vertices[2, :] = 1
    vertices = np.dot(np.linalg.inv(cropping_tform.params), cropped_vertices)
    vertices = np.vstack((vertices[:2, :], z))
    pos = np.reshape(vertices.T, [256, 256, 3])
    return pos


def get_cropped_image(img, cropping_tform):
    float_img = img / 256.0 / 1.1
    if not cropping_tform:
        return float_img
    else:
        return warp(float_img, cropping_tform.inverse, output_shape=(256, 256))


def main():
    model_path = 'Data/net-data/trained_fg_then_real.h5'  # trained_fg_then_real.h5'
    face_detector_path = 'Data/net-data/mmod_human_face_detector.dat'
    shape_predictor_path = 'Data/net-data/shape_predictor_68_face_landmarks.dat'
    # image_folder = 'test_images/'
    image_folder = 'Data/florence_objs_with_img'
    img_type = '.PNG'  # .png #

    triangles = np.loadtxt('Data/uv-data/triangles.txt').astype(np.int32)
    face_ind = np.loadtxt('Data/uv-data/face_ind.txt').astype(np.int32)
    extra_face_ind = np.loadtxt('Data/uv-data/extra_bfm_ind.txt').astype(np.int32)
    bfm_kpt_ind = np.loadtxt('Data/uv-data/bfm_kpt_ind.txt').astype(np.int32)
    face_detector = dlib.cnn_face_detection_model_v1(face_detector_path)
    shape_predictor = dlib.shape_predictor(shape_predictor_path)

    pos_predictor = MobilenetPosPredictor(256, 256)
    mobilenet_pos_predictor = os.path.join('', model_path)  # Data/net-data/keras_mobilenet_prn_20_epochs_097.h5')
    if not os.path.isfile(mobilenet_pos_predictor):
        print("please download trained model first.")
        exit()
    pos_predictor.restore(mobilenet_pos_predictor)

    face_imgs = []
    for face_folder in os.listdir(image_folder):
        if face_folder == 'results':
            continue
        front_img = os.path.join(image_folder, face_folder,
                                 'front' + img_type)  # 'front.jpg')#face_folder + '_front.png')
        side_img = os.path.join(image_folder, face_folder, 'left' + img_type)  # 'side.jpg')#face_folder + '_left.png')
        face_imgs.append([front_img, side_img])

    for images in face_imgs:
        print(images)
        front_img = imread(images[0], mode='RGB')[:, :, :3]
        side_img = imread(images[1], mode='RGB')[:, :, :3]
        if front_img.shape != (256, 256, 3):
            max_size = max(front_img.shape[0], front_img.shape[1])
            if max_size > 1000:
                front_img = rescale(front_img, 1000. / max_size)
                front_img = (front_img * 255).astype(np.uint8)
            front_img = np.around(front_img, decimals=1).astype(np.uint8)

        if side_img.shape != (256, 256, 3):
            max_size = max(side_img.shape[0], side_img.shape[1])
            if max_size > 1000:
                side_img = rescale(side_img, 1000. / max_size)
                side_img = (side_img * 255).astype(np.uint8)
            side_img = np.around(side_img, decimals=1).astype(np.uint8)

        l68_front, cropping_tform_front = get_cropping_transformation(front_img, face_detector, shape_predictor)
        l68_side, cropping_tform_side = get_cropping_transformation(side_img, face_detector, shape_predictor)

        cropped_image_front = get_cropped_image(front_img, cropping_tform_front)
        cropped_image_side = get_cropped_image(side_img, cropping_tform_side)
        imsave(images[0].replace('front', 'side_cropped'), cropped_image_side)
        imsave(images[0].replace('front', 'front_cropped'), cropped_image_front)
        img_concat = np.concatenate((cropped_image_front, cropped_image_side), axis=2)
        cropped_pos = pos_predictor.predict(img_concat)

        pos = uncrop_pos(cropped_pos, cropping_tform_front)

        all_vertices = np.reshape(pos, [256 ** 2, -1])
        vertices = all_vertices[face_ind, :]

        save_vertices = vertices.copy()
        save_vertices[:, 1] = 256 - 1 - save_vertices[:, 1]

        [h, w, _] = front_img.shape
        vertices[:, 0] = np.minimum(np.maximum(vertices[:, 0], 0), w - 1)  # x
        vertices[:, 1] = np.minimum(np.maximum(vertices[:, 1], 0), h - 1)  # y
        ind = np.round(vertices).astype(np.int32)
        colors = front_img[ind[:, 1], ind[:, 0], :]  # n x 3

        write_obj_with_colors(images[0].replace(img_type, '.obj'), save_vertices, triangles, colors)

        masked_pos = mask_pos(pos)
        plotted_image = plot_vertices_on_image_from_pos(masked_pos, l68_front, front_img)
        imsave(images[0].replace('front', 'projected'), plotted_image)


if __name__ == '__main__':
    main()
