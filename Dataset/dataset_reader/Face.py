from math import degrees

from scipy.io import loadmat


class Face(object):
    def __init__(self, dataset_name, face_id):
        self.face_poses = []
        self.dataset_name = dataset_name
        self.face_id = face_id

    def add_face_pose(self, img_path, get_pose, pose):
        face_pose = Face_pose(img_path, get_pose, pose)
        self.face_poses.append(face_pose)

    def sort_face_poses(self):
        self.face_poses = sorted(self.face_poses, key=lambda face_pose: abs(face_pose.pose), reverse=False)


class Face_pose(object):
    def __init__(self, path, get_pose, pose):
        self.img_path = path
        self.mat_path = path.replace('jpg', 'mat')
        self.pose = pose
        if get_pose:
            self.get_pose()

    def get_pose(self):
        mat_file = loadmat(self.mat_path)
        self.pose = round(degrees(mat_file['Pose_Para'][0][1]), 0)  # pose param has jaw angle in radians
