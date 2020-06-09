import argparse
import glob
import os

import numpy as np
from scipy.spatial import distance
from sklearn.neighbors import KDTree

from icp.icp import icp
from utils.write import write_obj_with_colors


def apply_homogenous_tform(tform, vertices):
    n, m = vertices.shape
    vertices_affine = np.ones((n, m + 1))
    vertices_affine[:, :3] = vertices.copy()
    vertices = np.dot(tform, vertices_affine.T).T
    return vertices[:, :3]


def get_vertices_from_obj(obj_path):
    with open(obj_path) as f:
        lines = f.readlines()
    vertices = [line for line in lines if line.startswith('v ')]
    vertices = strip_obj_string(vertices)
    return vertices


def strip_obj_string(lines):
    array = np.zeros((len(lines), 3))
    for i, line in enumerate(lines):
        sub_array = np.array(line[2:].split(' ')).astype(np.float32)[:3]
        array[i] = sub_array
    return array


class prediction_evaluater:
    def __init__(self):
        self.face_ind = np.loadtxt('../Data/uv-data/face_ind.txt').astype(np.int32)
        self.uv_kpt_ind = np.loadtxt('../Data/uv-data/uv_kpt_ind.txt').astype(np.int32)
        self.triangles = np.loadtxt('../Data/uv-data/triangles.txt').astype(np.int32)

    def __call__(self, predicted_vertices, ground_truth_vertices, alignment_data=None, save_vertices=False,
                 save_output='aligned_vertices.obj'):
        if alignment_data is not None:
            init_pose = alignment_data[:4]
            scale = alignment_data[4][0]
        else:
            init_pose = None
            scale = 1.0
        original_predicted_vertices = predicted_vertices.copy() * scale
        original_f_vertices = ground_truth_vertices.copy()
        if (predicted_vertices.shape[0] > ground_truth_vertices.shape[0]):
            diff = predicted_vertices.shape[0] - ground_truth_vertices.shape[0]
            predicted_vertices = predicted_vertices[diff:, :] * scale
        else:
            diff = ground_truth_vertices.shape[0] - predicted_vertices.shape[0]
            ground_truth_vertices = ground_truth_vertices[diff:, :] * scale

        tform, distances, i = icp(predicted_vertices, ground_truth_vertices,
                                  max_iterations=100, tolerance=0.0001, init_pose=init_pose)

        aligned_predicted_vertices = apply_homogenous_tform(tform, predicted_vertices)
        aligned_original_vertices = apply_homogenous_tform(tform, original_predicted_vertices)

        if save_vertices:
            colors = np.ones((aligned_original_vertices.shape))
            write_obj_with_colors(save_output, aligned_original_vertices, self.triangles, colors)

        error = self.nmse(aligned_original_vertices, original_f_vertices)
        return error

    def nmse(self, predicted_vertices, ground_truth_vertices, normalization_factor=None):
        # calculate the normalized mean squared error between a predicted and ground truth mesh
        if not normalization_factor:
            mins = np.amin(ground_truth_vertices, axis=0)
            maxes = np.amax(ground_truth_vertices, axis=0)
            bbox = np.sqrt((maxes[0] - mins[0]) ** 2 + (maxes[1] - mins[1]) ** 2 + (maxes[2] - mins[2]) ** 2)
            normalization_factor = bbox

        v_tree = KDTree(ground_truth_vertices)
        error_array = np.zeros(predicted_vertices.shape[0])
        for i, v in enumerate(predicted_vertices):
            dst, ind = v_tree.query([v], k=1)
            gt_v = ground_truth_vertices[ind[0][0]]
            error_array[i] = distance.euclidean(v, gt_v)

        nmse = np.mean(error_array) / normalization_factor
        print(nmse)
        return nmse


def evaluate_predictions(args):
    evaluater = prediction_evaluater()

    obj_paths = []
    for subject in os.listdir(args.florence_path):
        for obj_file in glob.glob(args.florence_path + '/' + subject + '/*.obj'):
            obj_name = obj_file.split('/')[-1]
            if 'front' in obj_name and 'subject' not in obj_name:
                predicted_obj_path = obj_file
            elif 'front' not in obj_name:
                gt_obj_file_path = obj_file
        alignment_data_path = glob.glob(args.florence_path + '/' + subject + '/pose.txt')[0]
        obj_paths.append([gt_obj_file_path, predicted_obj_path, alignment_data_path])

    errors = []
    for paths in obj_paths:
        print(paths[0])
        florence_vertices = get_vertices_from_obj(paths[0])
        predicted_vertices = get_vertices_from_obj(paths[1])
        alignment_data = np.loadtxt(fname=paths[2])
        error = evaluater(predicted_vertices, florence_vertices, alignment_data=alignment_data, save_vertices=True)
        errors.append(error)
    # errors.append(sum(errors) / len(errors))
    errors = np.array(errors)
    np.savetxt(args.save_file, errors)
    return
    '''

    vertices_array = get_vertices_from_obj('../Data/Florence_with_predicted/front.obj')
    vertices_array_two = get_vertices_from_obj('../Data/Florence_with_predicted/front_two.obj')
    vertices_array_two_latest = get_vertices_from_obj('../Data/Florence_with_predicted/front_two_latest.obj')
    vertices_array_gt = get_vertices_from_obj('../Data/Florence_with_predicted/florence.obj')
    init_align = np.array([[1, 0, 0, -200],
                [0, 1, 0, -200],
                [0, 0, 1, -50],
                [0, 0, 0, 1],
                [0.5, 0.5, 0.5 ,0.5]])
    evaluater(vertices_array, vertices_array_gt, save_vertices = True, alignment_data = init_align)
    evaluater(vertices_array_two, vertices_array_gt, save_vertices = True, alignment_data = init_align)
    evaluater(vertices_array_two_latest, vertices_array_gt, save_vertices = True, alignment_data = init_align)
    '''


if __name__ == '__main__':
    par = argparse.ArgumentParser(description='Network Evaluation')
    par.add_argument('--florence_path', default='../Data/Florence/files.txt', type=str,
                     help='The path to the florence dataset description file')
    par.add_argument('--save_file', default='./evaluation_results.txt', type=str,
                     help='The path to the file where the results are saved')
    evaluate_predictions(par.parse_args())
