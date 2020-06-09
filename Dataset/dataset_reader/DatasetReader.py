import glob
import os

import numpy as np
from Dataset.dataset_reader.Face import Face


class DatasetReader(object):
    def __init__(self, dataset_path, dataset_name="300w-lp"):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.faces = []
        self.face_dict = {}
        self.has_read = False

    def getFacesFromDataset(self):
        if self.has_read:
            return self.faces

        if self.dataset_name == '300w-lp':
            for sub_dataset in os.listdir(self.dataset_path):
                print('fetching faces from dataset: %s' % (sub_dataset))
                if ('IBUG' in sub_dataset):
                    self.add_IBUG_faces(sub_dataset)
                elif ('AFW' in sub_dataset):
                    self.add_AFW_faces(sub_dataset)
                elif ('LFPW' in sub_dataset):
                    self.add_LFPW_faces(sub_dataset)
                elif ('HELEN' in sub_dataset):
                    self.add_HELEN_faces(sub_dataset)
                else:
                    continue
            self.has_read = True

        elif self.dataset_name == 'facegen':
            self.add_facegen_faces()
            self.has_read = True
        else:
            print('only 300w-lp and facegen is currently supported')
            return None

        return self.faces

    def print_statistics(self):
        face_list = self.getFacesFromDataset()
        print('number of faces: %d' % (len(face_list)))
        distribution_of_face_poses = np.zeros(20)
        distribution_of_face_initial_poses = np.zeros(20)
        for face in face_list:
            distribution_of_face_poses[len(face.face_poses)] += 1
            face.sort_face_poses()
            rounded_pose_index = round(abs(face.face_poses[0].pose) / 5)
            distribution_of_face_initial_poses[rounded_pose_index] += 1

        print('facepose| faces')
        print('-----------------')
        for u, i in enumerate(distribution_of_face_poses):
            print('%d\t|\t%d' % (u, i))
        print('\ninitial\nfacepose| faces')
        print('-----------------')
        for u, i in enumerate(distribution_of_face_initial_poses):
            print('%d\t|\t%d' % (u * 5, i))

    def add_IBUG_faces(self, sub_dataset):
        img_paths = os.path.join(self.dataset_path, sub_dataset, '*.jpg')
        img_list = glob.glob(img_paths)
        for img_path in img_list:
            split_path = img_path.split('/')[-1].split('_')
            if len(split_path) == 4:
                face_number = str(split_path[2])
            elif len(split_path) == 5:
                face_number = str(split_path[2]) + '_' + str(split_path[3])
            self.populate_face_dict(img_path, face_number, sub_dataset)

    def add_AFW_faces(self, sub_dataset):
        img_paths = os.path.join(self.dataset_path, sub_dataset, '*.jpg')
        img_list = glob.glob(img_paths)
        for img_path in img_list:
            split_path = img_path.split('/')[-1].split('_')
            face_number = str(split_path[1]) + '_' + str(split_path[2])
            self.populate_face_dict(img_path, face_number, sub_dataset)

    def add_LFPW_faces(self, sub_dataset):
        img_paths = os.path.join(self.dataset_path, sub_dataset, '*.jpg')
        img_list = glob.glob(img_paths)
        for img_path in img_list:
            split_path = img_path.split('/')[-1].split('_')
            face_number = str(split_path[3]) + '_' + str(split_path[2])
            self.populate_face_dict(img_path, face_number, sub_dataset)

    def add_HELEN_faces(self, sub_dataset):
        img_paths = os.path.join(self.dataset_path, sub_dataset, '*.jpg')
        img_list = glob.glob(img_paths)
        for img_path in img_list:
            split_path = img_path.split('/')[-1].split('_')
            face_number = str(split_path[1]) + '_' + str(split_path[2])
            self.populate_face_dict(img_path, face_number, sub_dataset)

    def add_facegen_faces(self):
        for face_folder in os.listdir(self.dataset_path):
            img_path_front = os.path.join(self.dataset_path, face_folder, face_folder + '_front.png')
            img_path_left = os.path.join(self.dataset_path, face_folder, face_folder + '_left.png')
            img_path_right = os.path.join(self.dataset_path, face_folder, face_folder + '_right.png')
            self.populate_face_dict(img_path_front, face_folder, '', get_pose=False)
            self.populate_face_dict(img_path_left, face_folder, '', get_pose=False)
            self.populate_face_dict(img_path_right, face_folder, '', get_pose=False)

    def populate_face_dict(self, img_path, face_number, sub_dataset, get_pose=True, pose=0):
        face_id = face_number + '_' + sub_dataset.strip('/')
        if face_id in self.face_dict.keys():
            self.face_dict[face_id].add_face_pose(img_path, get_pose, pose)
            return
        face = Face(sub_dataset, face_id)
        face.add_face_pose(img_path, get_pose, pose)
        self.faces.append(face)
        self.face_dict[face_id] = face
