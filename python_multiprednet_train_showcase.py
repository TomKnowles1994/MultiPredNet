"""
To train a Multi-modal Predictive coding Network (MuPNet) using visual tactile data gathered from Physical and simulated WhiskEye robot
"""

import time, os, imghdr
import numpy as np
from numpy.random import permutation
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

### User-defined Parameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

n_sample = 1200                             # Number of samples in the dataset. 
                                            # If you have collected your own dataset, you will need to determine how many samples where collected in the run
                                            # Alternatively, if you are using a built-in dataset, copy the sample number as described in the datasets' README

minibatch_sz = 1                            # Minibatch size. Can be left as default for physical data, for simulated data good numbers to try are 40, 50 and 100
                                            # Datasize size must be fully divisible by minibatch size

data_path = '/your/path/here'               # Path to training data. Training data should be in .mat format:

save_path = '/your/path/here'               # Path to save trained model to (once trained)
load_path = '/your/path/here'               # Path to load trained model from (after training, or if already trained beforehand)

n_epoch = 200                               # Number of training epochs to generate model. Default is 200
                                            
shuffle_data = False                        # Do you want to shuffle the training data? Default is False


### Model Hyperparameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

load_model = False                          # If True, load a previously trained model from load_path. If False, train from scratch.

m1_inp_shape = visual_data.shape[1]         # modality 1 (default vision) input layer shape
m2_inp_shape = tactile_data.shape[1]        # modality 2 (default tactile) input layer shape

m1_layers = [1000, 300]                     # modality 1 layers shape
m2_layers = [50, 20]                        # modality 2 layers shape
msi_layers = [100]                          # multi-modal integration layers shape

m1_cause_init = [0.1, 0.1]                  # the starting value for the inference process, whereby priors are updated according to evidence; modality 1
m2_cause_init = [0.1, 0.1]                  # the starting value for the inference process, whereby priors are updated according to evidence; modality 2
msi_cause_init = [0.1]                      # the starting value for the inference process, whereby priors are updated according to evidence; multi-modal integration

reg_m1_causes = [0.0, 0.0]                  # regularised error, disabled by default; modality 1
reg_m2_causes = [0.0, 0.0]                  # regularised error, disabled by default; modality 2
reg_msi_causes = [0.0]                      # regularised error, disabled by default; multi-modal integration

lr_m1_causes = [0.0004, 0.0004]             # learning rate for the inference process; modality 1
lr_m2_causes = [0.004, 0.004]               # learning rate for the inference process; modality 2
lr_msi_causes = [0.0004]                    # learning rate for the inference process; multi-modal integration

reg_m1_filters = [0.0, 0.0]                 # filters for regularised error, disabled by default; modality 1
reg_m2_filters = [0.0, 0.0]                 # filters for regularised error, disabled by default; modality 2
reg_msi_filters = [0.0, 0.0]                # filters for regularised error, disabled by default; multi-modal integration

lr_m1_filters = [0.0001, 0.0001]            # learning rate for the inference process; modality 1
lr_m2_filters = [0.001, 0.001]              # learning rate for the inference process; modality 2
lr_msi_filters = [0.0001, 0.0001]           # learning rate for the inference process; multi-modal integration

def load_mat_data(data_path, shuffle=True):

    img = img_as_float(np.array(sio.loadmat(data_path + '/images.mat')['images'].tolist())[0]) #load from matlab and normalise
    img = img.reshape(img.shape[0], 10800) # flatten
    theta = sio.loadmat(data_path + '/theta.mat')['theta']
    xy = sio.loadmat(data_path + '/xy.mat')['xy']

    # reshape and combine whisker data
    theta = np.reshape(theta, [-1, theta.shape[-1]]).T
    xy = np.reshape(xy, [-1, xy.shape[-1]]).T
    tactile_data = preprocess_tactile_data(np.concatenate([theta, xy], axis=1))

    if shuffle:
        # shuffle sequence of data but maintain visual-tactile alignment
        img, tactile_data = shuffle_in_sync(img, tactile_data)

    return img, tactile_data


def preprocess_tactile_data(tactile_data):
    scaler = MinMaxScaler(copy=False)
    scaler.fit(tactile_data)
    scaler.transform(tactile_data)

    return tactile_data

def shuffle_in_sync(visual_data, tactile_data):
    assert visual_data.shape[0] == tactile_data.shape[0]

    shared_indices = permutation(visual_data.shape[0])
    shuffled_visual, shuffled_tactical = visual_data[shared_indices], tactile_data[shared_indices]

    return shuffled_visual, shuffled_tactical

class Network:
    def __init__(self, n_sample, minibatch_sz, m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                 lr_m2_causes, lr_msi_causes, reg_m1_filters, reg_m2_filters, reg_msi_filters, lr_m1_filters,
                 lr_m2_filters, lr_msi_filters):

        self.m1_inp_shape = m1_inp_shape
        self.m2_inp_shape = m2_inp_shape
        self.m1_layers = m1_layers
        self.m2_layers = m2_layers
        self.msi_layers = msi_layers

        # create placeholders
        self.x_m1 = tf.placeholder(tf.float32, shape=[minibatch_sz, m1_inp_shape])
        self.x_m2 = tf.placeholder(tf.float32, shape=[minibatch_sz, m2_inp_shape])
        self.batch = tf.placeholder(tf.int32, shape=[])

        # create filters and cause for m1
        self.m1_filters = []
        self.m1_causes = []
        for i in range(len(self.m1_layers)):
            filter_name = 'm1_filter_%d' % i
            cause_name = 'm1_cause_%d' % i

            if i == 0:
                self.m1_filters += [tf.get_variable(filter_name, shape=[self.m1_layers[i], self.m1_inp_shape])]
            else:
                self.m1_filters += [tf.get_variable(filter_name, shape=[self.m1_layers[i], self.m1_layers[i-1]])]

            init = tf.constant_initializer(m1_cause_init[i])
            self.m1_causes += [tf.get_variable(cause_name, shape=[n_sample, self.m1_layers[i]], initializer=init)]

        # create filters and cause for m2
        self.m2_filters = []
        self.m2_causes = []
        for i in range(len(self.m2_layers)):
            filter_name = 'm2_filter_%d' % i
            cause_name = 'm2_cause_%d' % i

            if i == 0:
                self.m2_filters += [tf.get_variable(filter_name, shape=[self.m2_layers[i], self.m2_inp_shape])]
            else:
                self.m2_filters += [tf.get_variable(filter_name, shape=[self.m2_layers[i], self.m2_layers[i-1]])]

            init = tf.constant_initializer(m2_cause_init[i])
            self.m2_causes += [tf.get_variable(cause_name, shape=[n_sample, self.m2_layers[i]], initializer=init)]

        # create filters and cause for msi
        self.msi_filters = []
        self.msi_causes = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                # add filters for m1
                filter_name = 'msi_m1_filter'
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.m1_layers[-1]])]
                # add filters for m2
                filter_name = 'msi_m2_filter'
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.m2_layers[-1]])]
            else:
                filter_name = 'msi_filter_%d' % i
                self.msi_filters += [tf.get_variable(filter_name, shape=[self.msi_layers[i],
                                                                                   self.msi_layers[i - 1]])]

            cause_name = 'msi_cause_%d' % i
            init = tf.constant_initializer(msi_cause_init[i])
            self.msi_causes += [tf.get_variable(cause_name, shape=[n_sample, self.msi_layers[i]], initializer=init)]

        # compute predictions
        current_batch = tf.range(self.batch * minibatch_sz, (self.batch + 1) * minibatch_sz)
        # m1 predictions
        self.m1_minibatch = []
        self.m1_predictions = []
        for i in range(len(self.m1_layers)):
            self.m1_minibatch += [tf.gather(self.m1_causes[i], indices=current_batch, axis=0)]
            self.m1_predictions += [tf.nn.leaky_relu(tf.matmul(self.m1_minibatch[i], self.m1_filters[i]))]

        # m2 predictions
        self.m2_minibatch = []
        self.m2_predictions = []
        for i in range(len(self.m2_layers)):
            self.m2_minibatch += [tf.gather(self.m2_causes[i], indices=current_batch, axis=0)]
            self.m2_predictions += [tf.nn.leaky_relu(tf.matmul(self.m2_minibatch[i], self.m2_filters[i]))]

        # msi predictions
        self.msi_minibatch = []
        self.msi_predictions = []
        for i in range(len(self.msi_layers)):
            self.msi_minibatch += [tf.gather(self.msi_causes[i], indices=current_batch, axis=0)]
            if i == 0:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_minibatch[i], self.msi_filters[i]))]  # m1 prediction
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_minibatch[i], self.msi_filters[i+1]))]  # m2 prediction
            else:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_minibatch[i], self.msi_filters[i+1]))]

        # add ops for computing gradients for m1 causes and for updating weights
        self.m1_bu_error = []
        self.m1_update_filter = []
        self.m1_cause_grad = []
        for i in range(len(self.m1_layers)):
            if i == 0:
                self.m1_bu_error += [tf.losses.mean_squared_error(self.x_m1, self.m1_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.m1_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_minibatch[i - 1]), self.m1_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(self.m1_layers) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_predictions[i+1]), self.m1_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.msi_predictions[0]), self.m1_minibatch[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_m1_causes[i] * (self.m1_minibatch[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_m1_causes[i])(self.m1_minibatch[i])
            self.m1_cause_grad += [tf.gradients([self.m1_bu_error[i], td_error, reg_error],
                                                          self.m1_minibatch[i])[0]]

            # ops for updating weights
            reg_error = reg_m1_filters[i] * (self.m1_filters[i] ** 2)
            m1_filter_grad = tf.gradients([self.m1_bu_error[i], reg_error], self.m1_filters[i])[0]
            self.m1_update_filter += [
                tf.assign_sub(self.m1_filters[i], lr_m1_filters[i] * m1_filter_grad)]

        # add ops for computing gradients for m2 causes and for updating weights
        self.m2_bu_error = []
        self.m2_update_filter = []
        self.m2_cause_grad = []
        for i in range(len(self.m2_layers)):
            if i == 0:
                self.m2_bu_error += [tf.losses.mean_squared_error(self.x_m2, self.m2_predictions[i],
                                                                            reduction=tf.losses.Reduction.NONE)]
            else:
                self.m2_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m2_minibatch[i - 1]), self.m2_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(self.m2_layers) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m2_predictions[i+1]), self.m2_minibatch[i],
                        reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.msi_predictions[1]), self.m2_minibatch[i],
                        reduction=tf.losses.Reduction.NONE)

            reg_error = reg_m2_causes[i] * (self.m2_minibatch[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_m2_causes[i])(self.m2_minibatch[i])
            self.m2_cause_grad += [
                tf.gradients([self.m2_bu_error[i], td_error, reg_error], self.m2_minibatch[i])[0]]

            # add ops for updating weights
            reg_error = reg_m2_filters[i] * (self.m2_filters[i] ** 2)
            m2_filter_grad = tf.gradients([self.m2_bu_error[i], reg_error], self.m2_filters[i])[0]
            self.m1_update_filter += [
                tf.assign_sub(self.m2_filters[i], lr_m2_filters[i] * m2_filter_grad)]
            #else:
                #raise NotImplementedError

        # add ops for computing gradients for msi causes
        self.msi_bu_error = []
        self.msi_reg_error = []
        self.msi_update_filter = []
        self.msi_cause_grad = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_minibatch[-1]), self.msi_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m2_minibatch[-1]), self.msi_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.msi_reg_error += [reg_msi_causes[i] * (self.msi_minibatch[i] ** 2)]
                # self.msi_reg_error += [tf.keras.regularizers.l2(reg_msi_causes[i])(self.msi_minibatch[i])]
                if len(self.msi_layers) > 1:
                    raise NotImplementedError
                else:
                    self.msi_cause_grad += [
                        tf.gradients([self.msi_bu_error[i], self.msi_bu_error[i+1], self.msi_reg_error[i]],
                                               self.msi_minibatch[i])[0]]

                # add ops for updating weights
                reg_error = reg_msi_filters[i] * (self.msi_filters[i] ** 2)
                msi_filter_grad = tf.gradients([self.msi_bu_error[i], reg_error], self.msi_filters[i])[0]
                self.msi_update_filter += [
                    tf.assign_sub(self.msi_filters[i], lr_msi_filters[i] * msi_filter_grad)]
                reg_error = reg_msi_filters[i+1] * (self.msi_filters[i+1] ** 2)
                msi_filter_grad = tf.gradients([self.msi_bu_error[i+1], reg_error], self.msi_filters[i+1])[0]
                self.msi_update_filter += [
                    tf.assign_sub(self.msi_filters[i+1], lr_msi_filters[i+1] * msi_filter_grad)]
            else:
                raise NotImplementedError

        # add ops for updating causes
        self.m1_update_cause = []
        self.m2_update_cause = []
        self.msi_update_cause = []
        with tf.control_dependencies(self.m1_cause_grad + self.m2_cause_grad + self.msi_cause_grad):
            # m1 modality
            for i in range(len(self.m1_layers)):
                self.m1_update_cause += [tf.scatter_sub(self.m1_causes[i], indices=current_batch,
                                                                  updates=(lr_m1_causes[i] * self.m1_cause_grad[i]))]

            # m2 modality
            for i in range(len(self.m2_layers)):
                self.m2_update_cause += [tf.scatter_sub(self.m2_causes[i], indices=current_batch,
                                                                  updates=(lr_m2_causes[i] * self.m2_cause_grad[i]))]

            # msi modality
            for i in range(len(self.msi_layers)):
                self.msi_update_cause += [tf.scatter_sub(self.msi_causes[i], indices=current_batch,
                                                                   updates=(lr_msi_causes[i] * self.msi_cause_grad[i]))]


def train():
    tf.compat.v1.reset_default_graph()

    # load direct from matlab objects
    visual_data, tactile_data = load_mat_data(data_path, shuffle_data)

    completed_epoch = 0

    net = Network(n_sample, minibatch_sz, m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                  lr_m2_causes, lr_msi_causes, reg_m1_filters, reg_m2_filters, reg_msi_filters, lr_m1_filters,
                  lr_m2_filters, lr_msi_filters)

    saver = tf.train.Saver()
    cause_epoch = 20
    config = tf.ConfigProto(device_count={'GPU': 1})
    with tf.Session(config=config) as sess:
        if load_model is True:
            saver.restore(sess, '%s/main.ckpt' % load_path)
        else:
            sess.run(tf.global_variables_initializer())

        if load_model is True:
            m1_epoch_loss = np.load('%s/m1_epoch_loss.npy' % load_path)
            assert completed_epoch == m1_epoch_loss.shape[0], 'Value of completed_epoch is incorrect'

            m1_epoch_loss = np.vstack([m1_epoch_loss, np.zeros((n_epoch, len(m1_layers)))])
            #m2_epoch_loss = np.vstack(
            #    [np.load('%s/m2_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(msi_layers)))])
            m2_epoch_loss = np.vstack(
                [np.load('%s/m2_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(m2_layers)))])
            msi_epoch_loss = np.vstack(
                [np.load('%s/msi_epoch_loss.npy' % load_path), np.zeros((n_epoch, len(msi_layers) + 1))])

            m1_avg_activity = np.vstack(
                [np.load('%s/m1_avg_activity.npy' % load_path), np.zeros((n_epoch, len(m1_layers)))])
            m2_avg_activity = np.vstack(
                [np.load('%s/m2_avg_activity.npy' % load_path), np.zeros((n_epoch, len(m2_layers)))])
            msi_avg_activity = np.vstack(
                [np.load('%s/msi_avg_activity.npy' % load_path), np.zeros((n_epoch, len(msi_layers)))])
        else:
            m1_epoch_loss = np.zeros((n_epoch, len(m1_layers)))
            m2_epoch_loss = np.zeros((n_epoch, len(m2_layers)))
            msi_epoch_loss = np.zeros((n_epoch, len(msi_layers) + 1))

            m1_avg_activity = np.zeros((n_epoch, len(m1_layers)))
            m2_avg_activity = np.zeros((n_epoch, len(m2_layers)))
            msi_avg_activity = np.zeros((n_epoch, len(msi_layers)))

        for i in range(n_epoch):
            current_epoch = completed_epoch + i

            n_batch = n_sample // minibatch_sz
            for j in range(n_batch):
                visual_batch = visual_data[(j*minibatch_sz):((j+1)*minibatch_sz), :]
                tactile_batch = tactile_data[(j * minibatch_sz):((j + 1) * minibatch_sz), :]

                # update causes
                for k in range(cause_epoch):
                    m1_cause, m2_cause, msi_cause, m1_grad, m2_grad, msi_reg_error = sess.run(
                        [net.m1_update_cause, net.m2_update_cause, net.msi_update_cause,
                         net.m1_cause_grad, net.m2_cause_grad, net.msi_reg_error],
                        feed_dict={net.x_m1: visual_batch, net.x_m2: tactile_batch, net.batch: j})

                # update weights
                _, _, _, m1_error, m2_error, msi_error, m1_filter, m2_filter, msi_filter = sess.run(
                    [net.m1_update_filter, net.m2_update_filter, net.msi_update_filter,
                     net.m1_bu_error, net.m2_bu_error, net.msi_bu_error,
                     net.m1_filters, net.m2_filters, net.msi_filters],
                    feed_dict={net.x_m1: visual_batch, net.x_m2: tactile_batch, net.batch: j})

                # record maximum reconstruction error on the entire data
                m1_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > m1_epoch_loss[current_epoch, l]
                                                   else m1_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(m1_error)]
                m2_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                   if np.max(np.mean(item, axis=1)) > m2_epoch_loss[current_epoch, l]
                                                   else m2_epoch_loss[current_epoch, l]
                                                   for l, item in enumerate(m2_error)]
                msi_epoch_loss[current_epoch, :] = [np.max(np.mean(item, axis=1))
                                                    if np.max(np.mean(item, axis=1)) > msi_epoch_loss[current_epoch, l]
                                                    else msi_epoch_loss[current_epoch, l]
                                                    for l, item in enumerate(msi_error)]

            # m1_epoch_loss[current_epoch, :] /= (n_sample // minibatch_sz)
            # m2_epoch_loss[current_epoch, :] /= (n_sample // minibatch_sz)
            # msi_epoch_loss[current_epoch, :] /= (n_sample // minibatch_sz)

            # track average activity in inferred causes
            m1_avg_activity[current_epoch, :] = [np.mean(item) for item in m1_cause]
            m2_avg_activity[current_epoch, :] = [np.mean(item) for item in m2_cause]
            msi_avg_activity[current_epoch, :] = [np.mean(item) for item in msi_cause]

            print('(%d) M1:%s (%s), M2:%s (%s), MSI:%s (%s)' % (
                i, ', '.join(['%.8f' % elem for elem in m1_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in m1_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in m2_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in m2_avg_activity[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in msi_epoch_loss[current_epoch, :]]),
                ', '.join(['%.8f' % elem for elem in msi_avg_activity[current_epoch, :]])))

        # create the save path if it does not exist
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # save model and stats
        saver.save(sess, '%s/main.ckpt' % save_path)
        np.save('%s/m1_epoch_loss.npy' % save_path, m1_epoch_loss)
        np.save('%s/m2_epoch_loss.npy' % save_path, m2_epoch_loss)
        np.save('%s/msi_epoch_loss.npy' % save_path, msi_epoch_loss)
        np.save('%s/m1_avg_activity.npy' % save_path, m1_avg_activity)
        np.save('%s/m2_avg_activity.npy' % save_path, m2_avg_activity)
        np.save('%s/msi_avg_activity.npy' % save_path, msi_avg_activity)


if __name__ == '__main__':
    starttime = time.time()
    train()
    endtime = time.time()

    print ('Time taken: %f' % ((endtime - starttime) / 3600))



#def preprocess_images(preproc_setting, filename):
#    final_image_sz = preproc_setting['final_image_sz'] # (45 x 80)
#
#    # read images as 32 bit floats
#    img = cv2.imread(filename).astype(np.float32)
#
#    if preproc_setting['gray_scale']:
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#    if preproc_setting['resize_image']:
#        img = cv2.resize(img, (final_image_sz[1], final_image_sz[0]))
#
#    if preproc_setting['norm_image']:
#        img = img / 255
#
#    if preproc_setting['flatten']:
#        img = np.ravel(img)
#
#    return img
#
#def convert_to_npy(data_path, image_proc_setting, num_images):
#    img_arr = np.zeros([num_images, 10800]) # to fit size of specific dataset for training 200521
#    cam1_images_n = np.zeros([0], dtype=np.int)
#    #theta_meas = np.zeros([4, 6, 0])
#    #xy_meas = np.zeros([2, 4, 6, 0])
#    theta = np.zeros([4, 6, 0])
#    xy = np.zeros([2, 4, 6, 0])
#    cur_img = 0
#
#    print("Converting CSV to NPY...")
#    for dp in data_path:
#        for dir in os.listdir(dp):
#            print(dp + '/' + dir)
#            for f in os.listdir(dp + '/' + dir):
#                if (imghdr.what(dp + '/' + dir + '/' + f) == 'png') and ('cam1' in dp + '/' + dir + '/' + f):
#                    img_arr[cur_img] = preprocess_images(image_proc_setting, dp + '/' + dir + '/' + f)
#                    cur_img = cur_img + 1
#            cam1_images_n = np.genfromtxt(dp + '/' + dir + '/cam1_n', delimiter=',', dtype=np.int)
#            #cam1_images_n = np.concatenate([cam1_images_n, x])
#            print("loaded images")
#
#            theta_meas = np.reshape(np.genfromtxt(dp + '/' + dir + '/theta_meas', delimiter=','), [4, 6, -1])
#            #theta_meas = np.concatenate([theta_meas, x], axis=2)
#            print("loaded theta_meas")
#
#            xy_meas = np.reshape(np.genfromtxt(dp + '/' + dir + '/xy_meas_clean', delimiter=','), [2, 4, 6, -1])
#            #xy_meas = np.concatenate([xy_meas, x], axis=3)
#            print("loaded xy_meas_clean")
#
#
#            sample_idx = cam1_images_n * 10
#            theta = np.concatenate([theta, theta_meas[:, :, sample_idx]], axis=2)
#            print(np.shape(theta))
#
#            xy = np.concatenate([xy, xy_meas[:, :, :, sample_idx]], axis=3)
#            print(np.shape(xy))
#    print(" done! ")
#    # create the save path if it does not exist
#    if not os.path.exists(dp + '_npy'):
#        os.makedirs(dp + '_npy')
#
#    # save loaded data
#    np.save(dp + '_npy/images', img_arr)
#    #np.save(dp + '_npy/theta_meas', theta_meas)
#    #np.save(dp + '_npy/xy_meas', xy_meas)
#    np.save(dp + '_npy/theta', theta)
#    np.save(dp + '_npy/xy', xy)
#
#    print('saved to: ' + dp + '_npy')
#
#    #return img_arr, theta, xy
#
#
#def load_npy_data(data_path):
#    img = np.load(data_path + '_npy/images.npy')
#    theta = np.load(data_path + '_npy/theta.npy')
#    xy = np.load(data_path + '_npy/xy.npy')
#
#    # reshape and combine whisker data
#    theta = np.reshape(theta, [-1, theta.shape[-1]]).T
#    xy = np.reshape(xy, [-1, xy.shape[-1]]).T
#    tactile_data = preprocess_tactile_data(np.concatenate([theta, xy], axis=1))
#
#    return img, tactile_data#
