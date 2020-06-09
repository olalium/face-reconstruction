import xml.etree.ElementTree as et

import face3d
import numpy as np
import skimage.transform
from face3d import mesh
from face3d.morphable_model import MorphabelModel
from pos_map_code import process_uv
from scipy.spatial.transform import Rotation as R
from skimage import io
from skimage.io import imread, imsave
from sklearn.neighbors import KDTree


def apply_similarity_and_projection_transform(vertices, R, t3d, P, s=1.):
    '''
    vertices = 3D vertices you want to transform
    R = rotation matrix for vertices
    t3d = translation vertex for x,y,z
    P = perspective matrix
    s = scale
    '''

    # similarity transform
    t_vertices = s * vertices.dot(R.T) + t3d[np.newaxis, :]

    # projection
    t_vertices_h = np.hstack((t_vertices, np.ones((t_vertices.shape[0], 1))))
    projected_vertices = t_vertices_h.dot(P.T)

    return projected_vertices


def from_projected_vertices_to_image_vertices(projected_vertices, height=256, width=256, depth=256):
    '''
    Defines and applies a viewport transform
    divide the projected vertices on their w
    we normalize z to begin at 0
    '''
    viewport_matrix = np.array([
        [(width - 0) / 2, 0, 0, (0 + width) / 2],
        [0, (height - 0) / 2, 0, (0 + height) / 2],
        [0, 0, (depth - 0) / 2, (0 + depth) / 2],
        [0, 0, 0, 1]
    ])
    image_vertices_h = projected_vertices.dot(viewport_matrix.T)
    image_vertices = np.divide(image_vertices_h[:, :3], image_vertices_h[:, 3, np.newaxis])  # divide by w

    image_vertices[:, 2] = np.max(image_vertices[:, 2]) - image_vertices[:, 2]  # substract z by max z value

    return image_vertices


def strip_obj_string(lines, depth, ind_start, ind_end):
    '''
    iterates over read obj lines and produces a numpy array with vertices in [x,y,z] format
    '''
    array = np.zeros((len(lines), depth))
    for i, line in enumerate(lines):
        sub_array = np.array(line[ind_start:ind_end].split(' ')).astype(np.float32)
        array[i] = sub_array
    return array


def interpolate_position_from_distances(face_vertices, dst_list, ind_list):
    '''
    Weighted average inteprolation based on input distances
    '''
    weights = 1 / dst_list[:]
    weights = weights[:] / np.sum(weights)
    vertices = np.zeros((3, 3))
    for i, w in enumerate(weights):
        vertices[i] = w * face_vertices[ind_list[i]]
    vertex = np.sum(vertices, axis=0)
    return vertex


def get_image_vertices_from_facegen(obj_vertices, img_path, save_image=False):
    # set file paths
    xml_settings_cam_path = img_path.replace('.png', '_cam.xml')
    xml_settings_pose_path = img_path.replace('.png', '.xml')
    img_out_path = img_path.replace('.png', '_projected.png')

    # initialize element trees from xml
    cam_settings_file = et.parse(xml_settings_cam_path)
    cam_settings = cam_settings_file.getroot()
    xml_val_cam = cam_settings.find('val')

    pose_settings_file = et.parse(xml_settings_pose_path)
    pose_settings = pose_settings_file.getroot()
    xml_val_pose = pose_settings.find('val')

    # rotation
    xml_rend = xml_val_pose.find('rend')
    xml_pose = xml_rend.find('pose')
    xml_rotateToHcs = xml_pose.find('rotateToHcs')
    xml_m_comp = xml_rotateToHcs.find('m_comp')
    xml_m = xml_m_comp.find('m')
    xml_m_items = xml_m.findall('item')
    quat_x = float(xml_m_items[0].text)
    quat_y = float(xml_m_items[1].text)
    quat_z = float(xml_m_items[2].text)
    rotation_matrix = R.from_quat([quat_x, quat_y, quat_z, 1]).as_matrix()  # assumes w = 1

    # translation
    xml_modelview = xml_val_cam.find('modelview')
    xml_translation = xml_modelview.find('translation')
    xml_m = xml_translation.find('m')
    translation_vector = np.zeros(3)
    for x, v in enumerate(xml_m.iter('item')):
        translation_vector[x] = float(v.text)

    # projection
    xml_frustum = xml_val_cam.find('frustum')
    xml_m = xml_frustum.find('m')
    xml_items = xml_m.findall('item')
    l0 = float(xml_items[0].text)
    r0 = float(xml_items[1].text)
    b0 = float(xml_items[2].text)
    t0 = float(xml_items[3].text)
    n0 = float(xml_items[4].text)
    f0 = float(xml_items[5].text)
    proj_matrix = np.array([
        [2 * n0 / (r0 - l0), 0, 0, 0],
        [0, 2 * n0 / (t0 - b0), 0, 0],
        [0, 0, (n0 + f0) / (f0 - n0), -(2 * n0 * f0) / (f0 - n0)],
        [0, 0, -1, 0]
    ])

    # transform
    projected_vertices = apply_similarity_and_projection_transform(
        obj_vertices,
        rotation_matrix,
        translation_vector,
        proj_matrix
    )
    image_vertices = from_projected_vertices_to_image_vertices(projected_vertices)

    # save output image
    if (save_image):
        img_vertices_clipped = np.clip(image_vertices, 0, 256)  # clip vertice coords to fit image bounds
        img = imread(img_path) / 255.
        img_reshape = transform.resize(img, (256, 256))
        image_out = img_reshape.copy().astype(np.float32)

        for i, vertex in enumerate(img_vertices_clipped):
            ind = np.round(vertex).astype(np.int8)
            image_out[ind[1]][ind[0]] = [1, 0, ind[2] / 256, 1]
        imsave(img_out_path, image_out)

    return image_vertices


def generate_face_mask_facegen(obj_path, mask_path, width=256, height=256):
    with open(obj_path) as f:
        lines = f.readlines()
    uv_coords = [line for line in lines if line.startswith('vt ')]
    uv_coords_array = strip_obj_string(uv_coords, 2, 3, -1)
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -1)

    front_face_vert = get_image_vertices_from_facegen(vertices_array, obj_path.replace('.obj', '_front.png'))  # , True)

    posmap = np.zeros((width, height, 3))
    uv_tree = KDTree(uv_coords_array)

    for h in range(height):
        v = 1. - (float(h) / float(height))
        for w in range(width):
            u = float(w) / float(width)
            dst, ind = uv_tree.query([[u, v]], k=3)
            if (dst[0][0] > 0.022):
                vertex = [0., 0., 0.]
            else:
                vertex = [1., 1., 1.]
            posmap[h][w] = vertex
    io.imsave(mask_path, posmap)


def generate_detail_mask_facegen(face_mask_path, eye_nose_mouth_mask_path, uv_image_path, result_path):
    final_mask = np.array((256, 256))
    face_mask = imread(face_mask_path, as_gray=True)
    e_n_m_mask = imread(eye_nose_mouth_mask_path, as_gray=True)  # eye, nose, mouth mask
    uv_image = imread(uv_image_path)

    assert face_mask.shape == e_n_m_mask.shape == (256, 256)

    face_mask[:, :] = np.divide(face_mask[:, :], 2)
    e_n_m_mask[:, :] = e_n_m_mask[:, :] * 2
    final_mask = face_mask + e_n_m_mask
    final_mask = np.clip(final_mask, 0, 1)

    final_mask_rgba = np.full((256, 256, 4), 255)
    final_mask_rgba[:, :, 3] = final_mask[:, :] * 255 / 2.
    for i_h, h in enumerate(uv_image):
        for i_v, v in enumerate(h):
            a = final_mask_rgba[i_h][i_v][3] / 255.
            uv_image[i_h][i_v] = uv_image[i_h][i_v] * (1.0 - a) + final_mask_rgba[i_h][i_v] * a
    uv_image[:, :, 3] = 255

    imsave(result_path, final_mask)
    imsave(result_path.replace('.png', '_with_image.png'), uv_image)


def generate_posmap_facegen(bfm, uv_coords, obj_path, pos_map_path, width=256, height=256, save_image=False):
    '''
    generate position map (u,v) => (x,y,z)
    '''
    # get data from .obj file
    with open(obj_path) as f:
        lines = f.readlines()
    uv_coords_obj = [line for line in lines if line.startswith('vt ')]
    uv_coords_array = strip_obj_string(uv_coords_obj, 2, 3, -1)
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -1)

    # get vertices in image space
    front_face_vert = get_image_vertices_from_facegen(vertices_array, obj_path.replace('.obj', '_front.png'))  # , True)
    # right_face_vert = get_image_vertices_from_facegen(vertices_array, obj_path.replace('.obj', '_right.png'), True)
    # left_face_vert = get_image_vertices_from_facegen(vertices_array, obj_path.replace('.obj', '_left.png'), True)

    l68_ind_fg = np.loadtxt('../Data/uv-data/00011_l68.txt', dtype=int)
    l68_vertices_fg = front_face_vert[l68_ind_fg[:]][:, :2]  # .astype(int)

    sp, ep, s, angles, t = bfm.fit(l68_vertices_fg, bfm.kpt_ind)

    vertices = bfm.generate_vertices(sp, ep)

    transformed_vertices = bfm.transform(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    image_vertices[:, 1] = height - image_vertices[:, 1] - 1
    position = image_vertices.copy()
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # normalize z so that 0 is the lowest. Same as 300W-LP

    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, height, width, c=3)
    uv_position_map = uv_position_map.astype(np.float16)

    if save_image:
        io.imsave(pos_map_path.replace('.npy', '_posmap.png'), uv_position_map / np.amax(uv_position_map))

    print(pos_map_path, uv_position_map.shape)
    # save position map
    np.save(pos_map_path, uv_position_map)


def generate_posmap_facegen_bfm(bfm, uv_coords, obj_path, pos_map_path, width=256, height=256, save_image=False):
    img_path = obj_path.replace('.obj', '_front.png')
    with open(obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices_array = strip_obj_string(vertices, 3, 2, -1)
    image_vertices = get_image_vertices_from_facegen(vertices_array, img_path)
    image_vertices[:, 1] = height - image_vertices[:, 1] - 1

    # crop image with key points
    kpt = image_vertices[bfm.kpt_ind, :].astype(np.int32)
    left = np.min(kpt[:, 0])
    right = np.max(kpt[:, 0])
    top = np.min(kpt[:, 1])
    bottom = np.max(kpt[:, 1])
    center = np.array([right - (right - left) / 2.0,
                       bottom - (bottom - top) / 2.0])
    old_size = (right - left + bottom - top) / 2
    size = int(old_size * 1.5)

    # randomize the cropping size
    marg = old_size * 0.1
    t_x = np.random.rand() * marg * 2 - marg
    t_y = np.random.rand() * marg * 2 - marg
    center[0] = center[0] + t_x
    center[1] = center[1] + t_y
    size = size * (np.random.rand() * 0.2 + 0.9)

    # crop and record the transform parameters
    src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2,
                                                                       center[1] + size / 2],
                        [center[0] + size / 2, center[1] - size / 2]])
    DST_PTS = np.array([[0, 0], [0, height - 1], [width - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)

    # transform face position(image vertices) along with 2d facial image
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z

    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, height, width, c=3)
    uv_position_map = uv_position_map.astype(np.float16)

    if save_image:
        io.imsave(pos_map_path.replace('.npy', '_posmap.png'), uv_position_map / np.amax(uv_position_map))

    # save position map
    print(pos_map_path, uv_position_map.shape)
    np.save(pos_map_path, uv_position_map)

    # return cropping transform
    return tform


if __name__ == '__main__':
    bfm = MorphabelModel('../Data/BFM/Out/BFM.mat')
    uv_coords = face3d.morphable_model.load.load_uv_coords('../Data/BFM/Out/BFM_UV.mat')
    uv_coords = process_uv(uv_coords)
    # np.savetxt('../Data/uv-data/bfm_kpt_ind.txt',bfm.kpt_ind)

    generate_posmap_facegen_bfm(bfm, uv_coords, '../Data/00001/00001.obj', '../Data/00001/00001_bfm.npy',
                                save_image=True)
