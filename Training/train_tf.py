import argparse
import math
import os
from datetime import datetime
from sys import stdout

import numpy as np
import tensorflow as tf
from TrainData import TrainData
from Networksetworks.predictor import resfcn256_6


def main(args):
    # Some arguments
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    batch_size = args.batch_size
    epochs = args.epochs
    train_data_file = args.train_data_file
    learning_rate = args.learning_rate
    model_path = args.model_path

    save_dir = args.checkpoint
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    data = TrainData(train_data_file)

    x = tf.placeholder(tf.float32, shape=[None, 256, 256, 6])
    label = tf.placeholder(tf.float32, shape=[None, 256, 256, 3])

    net = resfcn256_6(256, 256)
    x_op = net(x, is_training=True)

    loss = tf.losses.mean_squared_error(label, x_op)  # , weights=data.weight_mask)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(var_list=tf.global_variables())
    save_path = model_path

    # Begining train
    time_now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fp_log = open("training_logs/log_" + time_now + ".txt", "w")
    print("\n\nstarting training... \n \n")
    for epoch in range(epochs):
        for iters in range(int(math.ceil(1.0 * data.num_training_data / batch_size))):
            batch = data(batch_size, data.training_index, data.num_training_data, data.training_data)
            data.training_index = data.get_updated_index(data.training_index, batch_size, data.num_training_data)
            sess.run(train_step_optimizer, feed_dict={x: batch[0], label: batch[1]})
            if (iters % args.error_calculation_iters) == 0:
                loss_res = sess.run(loss, feed_dict={x: batch[0], label: batch[1]})
                stdout.flush()
                stdout.write('\riters:%d/epoch:%d,learning rate:%f,loss:%f' % (iters, epoch, learning_rate, loss_res))

        saver.save(sess=sess, save_path=save_path)
        stdout.write('\n\rcalculating validation loss...')
        validation_data = []
        for iters in range(int(math.ceil(1.0 * data.num_validation_data / batch_size))):
            batch = data(batch_size, data.validation_index, data.num_validation_data, data.validation_data, False)
            data.validation_index = data.get_updated_index(data.validation_index, batch_size, data.num_validation_data)
            validation_data.append(sess.run(loss, feed_dict={x: batch[0], label: batch[1]}))
        validation_data = np.array(validation_data)
        stdout.flush()
        stdout.write('\nvalidation loss:%f\n' % np.average(validation_data))
        fp_log.writelines('[%s] epoch:%d learning rate:%f validation loss:%f\n' % (
        datetime.now().strftime("%Y_%m_%d_%H_%M_%S"), epoch, learning_rate, np.average(validation_data)))

        if ((epoch != 0) and (epoch % 10 == 0)):  # learning rate halved every 5 epoch
            learning_rate = learning_rate / 2
    fp_log.close()


if __name__ == '__main__':
    par = argparse.ArgumentParser(
        description='Joint 3D Face Reconstruction and Dense Alignment with Position Map Regression Network')
    par.add_argument('--train_data_file', default='./results/datasetLabel.txt', type=str, help='The training data file')
    par.add_argument('--learning_rate', default=0.0001, type=float, help='The learning rate')
    par.add_argument('--epochs', default=200, type=int, help='Total epochs')
    par.add_argument('--batch_size', default=16, type=int, help='Batch sizes')
    par.add_argument('--checkpoint', default='checkpoint/', type=str, help='The path of checkpoint')
    par.add_argument('--model_path', default='checkpoint/256_256_resfcn256_weight', type=str,
                     help='The path of pretrained model')
    par.add_argument('--gpu', default='0', type=str, help='The GPU ID')
    par.add_argument('--error_calculation_iters', default=100, type=int, help='Run error calculation every x iters')

    main(par.parse_args())
