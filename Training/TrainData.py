import math

import cv2
import imgaug.augmenters as iaa
import numpy as np


class TrainData(object):
    def __init__(self, train_data_file, weight_mask_path='../Data/uv-data/uv_mask_final.png', pre_path=''):
        super(TrainData, self).__init__()
        self.training_data_file = train_data_file
        self.pre_path = pre_path
        self.data_list = []
        self.training_data = []
        self.validation_data = []
        self.num_training_data = 0
        self.num_validation_data = 0
        self.read_data(0.9)
        self.training_index = 0
        self.validation_index = 0
        self.rotate = False
        self.channel_scale = False
        self.dropout = False
        self.weight_mask = np.zeros(shape=(1, 256, 256, 3)).astype('float32')
        self.generateWeightMask(weight_mask_path)

    def read_data(self, percent_train):
        with open(self.training_data_file) as fp:
            temp = fp.readlines()
            for item in temp:
                item = item.strip().split()
                self.data_list.append(item)
            data_array = np.array(self.data_list)
            # np.random.shuffle(data_array)
            split_index = int(percent_train * data_array.shape[0])
            self.training_data = data_array[:split_index]
            self.num_training_data = self.training_data.shape[0]
            self.validation_data = data_array[split_index:]
            self.num_validation_data = self.validation_data.shape[0]

    def get_updated_index(self, index, batch_num, num_data):
        if (index + batch_num) < num_data:
            index += batch_num
        else:
            index = 0
        return index

    def getBatch(self, batch_list):
        batch = []
        imgs = []
        labels = []
        hasWritten = False
        for item in batch_list:
            if len(item) == 3:
                img_name = self.pre_path + item[0]
                img_name_side = self.pre_path + item[1]
                label_name = self.pre_path + item[2]
            else:
                img_name = self.pre_path + item[0] + ' ' + item[1]
                img_name_side = self.pre_path + item[2] + ' ' + item[3]
                label_name = self.pre_path + item[4] + ' ' + item[5]
            if not hasWritten:
                hasWritten = True
            # print(img_name)
            img = cv2.imread(img_name)
            img_side = cv2.imread(img_name_side)
            # print(img_name, img_name_side)
            if img.shape != (256, 256, 3):
                img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
                img_side = cv2.resize(img_side, (256, 256), interpolation=cv2.INTER_AREA)
            label = np.load(label_name)

            img_array = np.array(img, dtype=np.float32)
            img_array_side = np.array(img_side, dtype=np.float32)
            img_concat = np.concatenate((img_array, img_array_side), axis=2)

            label_array = np.array(label, dtype=np.float32)
            img_concat, label_array = self.augment_data(img_concat, label_array)
            imgs.append(img_concat / 256.0 / 1.1)  # or /255
            labels.append(label_array / 256.0 / 1.1)  # or /255
            # print(item[0], np.max(label_array.reshape(256*256, 3).T, axis = 1), np.min(label_array.reshape(256*256, 3).T, axis = 1))
        batch.append(imgs)
        batch.append(labels)

        return batch

    def __call__(self, batch_num, index, num_data, data):
        if (index + batch_num) <= num_data:
            batch_list = data[index:(index + batch_num)]
            batch_data = self.getBatch(batch_list)

        elif index < num_data:
            batch_list = data[index:num_data]
            batch_data = self.getBatch(batch_list)
        return batch_data

    def generateWeightMask(self, weight_mask_path):
        weights = cv2.imread(weight_mask_path, cv2.IMREAD_GRAYSCALE).astype('float32')  # [256, 256]
        weights = weights / 255.0
        self.weight_mask[0, :, :, 0] = weights  # / 16.0
        self.weight_mask[0, :, :, 1] = weights  # / 16.0
        self.weight_mask[0, :, :, 2] = weights  # / 16.0

    def augment_data(self, img_concat, label):
        if self.rotate:
            img_concat, label = self.rotate_data(img_concat, label)
        if self.channel_scale:
            img_concat = self.scale_image_channel(img_concat)
        if self.dropout:
            img_concat = self.apply_dropout(img_concat)
        # cv2.imwrite('random.png', img_concat[:,:,:3])
        return img_concat, label

    def rotate_data(self, img_concat, pos, angle_range=45):
        angle_front = np.random.randint(-angle_range, angle_range)
        angle_side = np.random.randint(-angle_range, angle_range)
        radians_front = angle_front / 180. * np.pi
        radians_side = angle_side / 180. * np.pi
        front_img_array = img_concat[:, :, :3]
        side_img_array = img_concat[:, :, 3:]
        h, w, c = front_img_array.shape
        M_front = self.getRotateMatrix(radians_front, front_img_array.shape)
        M_side = self.getRotateMatrix(radians_side, side_img_array.shape)
        rotated_front_img = cv2.warpPerspective(front_img_array, M_front, (h, w))
        rotated_side_img = cv2.warpPerspective(side_img_array, M_side, (h, w))
        rotated_pos = pos.copy()
        rotated_pos[:, :, 2] = 1.
        rotated_pos = rotated_pos.reshape(w * h, c)
        rotated_pos = np.dot(rotated_pos, M_front.T)
        rotated_pos = rotated_pos.reshape(h, w, c)
        rotated_pos[:, :, 2] = pos[:, :, 2]
        concat_rotated_img = np.concatenate((rotated_front_img, rotated_side_img), axis=2)
        return concat_rotated_img, rotated_pos

    def getRotateMatrix(self, angle, image_shape):
        [image_height, image_width, image_channel] = image_shape
        t1 = np.array([[1, 0, -image_height / 2.], [0, 1, -image_width / 2.], [0, 0, 1]])
        r1 = np.array([[math.cos(angle), math.sin(angle), 0], [math.sin(-angle), math.cos(angle), 0], [0, 0, 1]])
        t2 = np.array([[1, 0, image_height / 2.], [0, 1, image_width / 2.], [0, 0, 1]])
        rt_mat = t2.dot(r1).dot(t1)
        return rt_mat.astype(np.float32)

    def scale_image_channel(self, img_concat):
        rand_colors = np.random.uniform(low=0.6, high=1.4, size=(2, 3))
        img_concat[:, :, :3] = img_concat[:, :, :3] * rand_colors[0]
        img_concat[:, :, 3:] = img_concat[:, :, 3:] * rand_colors[1]
        return img_concat

    def apply_dropout(self, img_concat):
        augmentaton = iaa.CoarseDropout(p=(0.1, 0.2), size_percent=(.02, .02))
        img_concat[:, :, 3:] = augmentaton(image=img_concat[:, :, 3:])
        img_concat[:, :, :3] = augmentaton(image=img_concat[:, :, :3])
        return img_concat

    def set_augmentation(self, rotate, channel_scale, dropout):
        self.rotate = rotate
        self.channel_scale = channel_scale
        self.dropout = dropout
