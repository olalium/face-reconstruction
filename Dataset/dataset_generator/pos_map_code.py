import numpy as np
import scipy.io as sio
import skimage.transform
from skimage import io
from face3d import mesh

'''
Code based on the original implementation https://github.com/YadiraF/PRNet
'''


def process_uv(uv_coords, uv_h=256, uv_w=256):
    uv_coords[:, 0] = uv_coords[:, 0] * (uv_w - 1)
    uv_coords[:, 1] = uv_coords[:, 1] * (uv_h - 1)
    uv_coords[:, 1] = uv_h - uv_coords[:, 1] - 1
    uv_coords = np.hstack((uv_coords, np.zeros((uv_coords.shape[0], 1))))  # add z
    return uv_coords


def run_posmap_300W_LP(bfm, uv_coords, image_path, mat_path, save_folder, uv_h=256, uv_w=256, image_h=256, image_w=256):
    # load image and fitted parameters
    image_name = image_path.strip().split('/')[-1]
    image = io.imread(image_path) / 255.
    h, w, c = image.shape
    info = sio.loadmat(mat_path)
    pose_para = info['Pose_Para'].T.astype(np.float32)
    shape_para = info['Shape_Para'].astype(np.float32)
    exp_para = info['Exp_Para'].astype(np.float32)

    # generate mesh from shape and expression parameters
    vertices = bfm.generate_vertices(shape_para, exp_para)

    # transform mesh to VCC by applying scale, rotation and transformation
    s = pose_para[-1, 0]
    angles = pose_para[:3, 0]
    t = pose_para[3:6, 0]
    transformed_vertices = bfm.transform_3ddfa(vertices, s, angles, t)
    projected_vertices = transformed_vertices.copy()  # using stantard camera & orth projection as in 3DDFA
    image_vertices = projected_vertices.copy()
    image_vertices[:, 1] = h - image_vertices[:, 1] - 1

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
    DST_PTS = np.array([[0, 0], [0, image_h - 1], [image_w - 1, 0]])
    tform = skimage.transform.estimate_transform('similarity', src_pts, DST_PTS)
    cropped_image = skimage.transform.warp(image, tform.inverse, output_shape=(image_h, image_w))

    # transform face position(image vertices) along with 2d facial image
    position = image_vertices.copy()
    position[:, 2] = 1
    position = np.dot(position, tform.params.T)
    position[:, 2] = image_vertices[:, 2] * tform.params[0, 0]  # scale z
    position[:, 2] = position[:, 2] - np.min(position[:, 2])  # translate z

    # uv position map: render position in uv space
    uv_position_map = mesh.render.render_colors(uv_coords, bfm.full_triangles, position, uv_h, uv_w, c=3)
    uv_position_map = uv_position_map.astype(np.float16)

    # save files
    io.imsave('{}/{}'.format(save_folder, image_name), np.squeeze(cropped_image))
    np.save('{}/{}'.format(save_folder, image_name.replace('jpg', 'npy')), uv_position_map)
    io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_posmap.jpg')),
              (uv_position_map) / np.amax(uv_position_map))  # only for show #  / max(image_h, image_w)

    # --verify
    # uv_texture_map_rec = cv2.remap(cropped_image, uv_position_map[:,:,:2].astype(np.float32), None, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,borderValue=(0))
    # io.imsave('{}/{}'.format(save_folder, image_name.replace('.jpg', '_tex.jpg')), np.squeeze(uv_texture_map_rec))
