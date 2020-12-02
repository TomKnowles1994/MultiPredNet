import os
import numpy as np
import scipy.io as sio
from skimage.util import img_as_float, img_as_ubyte
from sklearn.preprocessing import MinMaxScaler
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

### User-Defined Parameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file
                                                                    
model_path = '/your/path/here'                                      # Path to saved model created by training script
tr_data_path = '/your/path/here'                                    # Path to the training data
ts_data_path = '/your/path/here'                                    # Path to the test data
save_path = '/your/path/here'                                       # Folder to save representations and reconstructions

avail_modality = 'both'                                             # Modalities made available to the network. Can be left as 'both' for full reconstructions.
                                                                    # Can be limited to 'visual' or 'tactile' to test the effects of sensor dropout

num_test_samps = 73                                                 # Samples in test dataset. Default is 73 for provided Physical test sets, 150 for Simulated

error_criterion = np.array([1e-3, 1e-3, 1e-4, 2e-3, 1e-3, 3e-3])    # This is the desired precision of the inference

max_iter = 500                                                      # Maximum number of inference cycles before the representation is generated

### Model Hyperparameters ###

## Note: if you change any of these, ensure the corresponding value (if applicable) is changed in the python_multiprednet_gen_reps_showcase.py file

m1_inp_shape = 45 * 80 * 3                  # modality 1 (default vision) input layer shape
m2_inp_shape = 72                           # modality 2 (default tactile) input layer shape
                                            
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

def init_minmaxscaler(data_path):
    concat_theta = sio.loadmat(data_path + '/theta.mat')['theta']
    concat_theta = np.reshape(concat_theta, (-1, concat_theta.shape[-1]))
    concat_xy = sio.loadmat(data_path + '/xy.mat')['xy']
    concat_xy = np.reshape(concat_xy, (-1, concat_xy.shape[-1]))
    concat_tactile_data = np.concatenate((concat_theta, concat_xy), axis=0)
    mms = MinMaxScaler(copy=False)
    mms.fit(concat_tactile_data.T)

    return mms


def load_mat_data(data_path, idx, mms):
    img = img_as_float(np.array(sio.loadmat(data_path + 'images.mat')['images'].tolist())[0][[idx],:,:,:])
    img = img.reshape(1, 10800) #flatten
    theta = sio.loadmat(data_path + 'theta.mat')['theta'][:,:,[idx]]
    xy = sio.loadmat(data_path + 'xy.mat')['xy'][:,:,:,[idx]]

    # reshape and combine whisker data
    theta = np.reshape(theta, [-1, 1]).T
    xy = np.reshape(xy, [-1, 1]).T
    conc = np.float32(np.concatenate([theta, xy], axis=1))
    tactile_data = mms.transform(conc)

    return img, tactile_data

def init_network(model_path, available_modality='both'):
    tf.reset_default_graph()

    net = Network(m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                  lr_m2_causes, lr_msi_causes, available_modality)

    saver = tf.compat.v1.train.Saver(net.m1_filters + net.m2_filters + net.msi_filters)
    config = tf.ConfigProto(device_count={'GPU': 1})
    sess = tf.Session(config=config)
    saver.restore(sess, '%smain.ckpt' % model_path)

    # sess = tf.Session()
    return sess, net

def infer_repr(sess, net, max_iter=1, error_criterion=0.0001, visual_data=None, tactile_data=None, verbose=False,
               available_modality='both'):
    sess.run(tf.compat.v1.variables_initializer(net.m1_causes + net.m2_causes + net.msi_causes))
    iter = 1

    while True:
        # infer representations
        m1_cause, m2_cause, msi_cause, m1_error, m2_error, msi_error, m1_pred, m2_pred, msi_pred, m1_filter, m2_filter, msi_filter = sess.run(
                [net.m1_update_cause, net.m2_update_cause, net.msi_update_cause, net.m1_bu_error, net.m2_bu_error, net.msi_bu_error,
                 net.m1_predictions, net.m2_predictions, net.msi_predictions, net.m1_filters, net.m2_filters, net.msi_filters],
                 feed_dict={net.x_m1: visual_data, net.x_m2: tactile_data})
        if available_modality is 'both':
            m1_epoch_loss = [np.mean(item) for item in m1_error]
            m2_epoch_loss = [np.mean(item) for item in m2_error]
            msi_epoch_loss = [np.mean(item) for item in msi_error]
        elif available_modality is 'visual':
            m1_epoch_loss = [np.mean(item) for item in m1_error]
            m2_epoch_loss = [np.NINF, np.NINF]
            msi_epoch_loss = [np.mean(item) for item in msi_error]
        elif available_modality is 'tactile':
            m1_epoch_loss = [np.NINF, np.NINF]
            m2_epoch_loss = [np.mean(item) for item in m2_error]
            msi_epoch_loss = [np.mean(item) for item in msi_error]

        if (np.all(np.array(m1_epoch_loss + m2_epoch_loss + msi_epoch_loss) < error_criterion)) or (iter >= max_iter):
            if verbose:
                print_str = ', '.join(['%.8f' % elem for elem in m1_epoch_loss + m2_epoch_loss + msi_epoch_loss])
                print ('(%d) %s' % (iter, print_str))
            break
        else:
            iter += 1

    # reconstruct the missing modality
    recon_tac = np.dot(msi_cause[0], msi_filter[1])
    for l in range(len(m2_filter), 0, -1):
        recon_tac = np.dot(recon_tac, m2_filter[l - 1])
    recon_vis = np.dot(msi_cause[0], msi_filter[0])
    for l in range(len(m1_filter), 0, -1):
        recon_vis = np.dot(recon_vis, m1_filter[l - 1])

    return msi_cause, recon_vis, recon_tac

class Network:
    def __init__(self, m1_inp_shape, m2_inp_shape, m1_layers, m2_layers, msi_layers, m1_cause_init,
                  m2_cause_init, msi_cause_init, reg_m1_causes, reg_m2_causes, reg_msi_causes, lr_m1_causes,
                 lr_m2_causes, lr_msi_causes, available_modality='both'):
        self.m1_inp_shape = m1_inp_shape
        self.m2_inp_shape = m2_inp_shape
        self.m1_layers = m1_layers
        self.m2_layers = m2_layers
        self.msi_layers = msi_layers

        # create placeholders
        self.x_m1 = tf.placeholder(tf.float32, shape=[1, m1_inp_shape])
        self.x_m2 = tf.placeholder(tf.float32, shape=[1, m2_inp_shape])

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
            self.m1_causes += [tf.get_variable(cause_name, shape=[1, self.m1_layers[i]], initializer=init)]

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
            self.m2_causes += [tf.get_variable(cause_name, shape=[1, self.m2_layers[i]], initializer=init)]

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
            self.msi_causes += [tf.get_variable(cause_name, shape=[1, self.msi_layers[i]], initializer=init)]

        # m1 predictions
        self.m1_predictions = []
        for i in range(len(self.m1_layers)):
            self.m1_predictions += [tf.nn.leaky_relu(tf.matmul(self.m1_causes[i], self.m1_filters[i]))]

        # m2 predictions
        self.m2_predictions = []
        for i in range(len(self.m2_layers)):
            self.m2_predictions += [tf.nn.leaky_relu(tf.matmul(self.m2_causes[i], self.m2_filters[i]))]

        # msi predictions
        self.msi_predictions = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_causes[i], self.msi_filters[i]))]  # m1 prediction
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_causes[i], self.msi_filters[i+1]))]  # m2 prediction
            else:
                self.msi_predictions += [tf.nn.leaky_relu(tf.matmul(self.msi_causes[i], self.msi_filters[i+1]))]

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
                    tf.stop_gradient(self.m1_causes[i - 1]), self.m1_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]

            # compute top-down prediction error
            if len(self.m1_layers) > (i + 1):
                # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_predictions[i+1]), self.m1_causes[i],
                    reduction=tf.losses.Reduction.NONE)
            else:
                # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.msi_predictions[0]), self.m1_causes[i],
                    reduction=tf.losses.Reduction.NONE)

            reg_error = reg_m1_causes[i] * (self.m1_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_m1_causes[i])(self.m1_minibatch[i])
            self.m1_cause_grad += [tf.gradients([self.m1_bu_error[i], td_error, reg_error],
                                                          self.m1_causes[i])[0]]

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
                    tf.stop_gradient(self.m2_causes[i - 1]), self.m2_predictions[i],
                        reduction=tf.losses.Reduction.NONE)]

            if len(self.m2_layers) > (i + 1):
            # there are more layers in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m2_predictions[i+1]), self.m2_causes[i],
                        reduction=tf.losses.Reduction.NONE)
            else:
            # this is the only layer in this modality
                td_error = tf.losses.mean_squared_error(
                    tf.stop_gradient(self.msi_predictions[1]), self.m2_causes[i],
                        reduction=tf.losses.Reduction.NONE)

            reg_error = reg_m2_causes[i] * (self.m2_causes[i] ** 2)
            # reg_error = tf.keras.regularizers.l2(reg_m2_causes[i])(self.m2_minibatch[i])
            self.m2_cause_grad += [
                tf.gradients([self.m2_bu_error[i], td_error, reg_error], self.m2_causes[i])[0]]

        # add ops for computing gradients for msi causes
        self.msi_bu_error = []
        self.msi_reg_error = []
        self.msi_update_filter = []
        self.msi_cause_grad = []
        for i in range(len(self.msi_layers)):
            if i == 0:
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m1_causes[-1]), self.msi_predictions[i],
                    reduction=tf.losses.Reduction.NONE)]
                self.msi_bu_error += [tf.losses.mean_squared_error(
                    tf.stop_gradient(self.m2_causes[-1]), self.msi_predictions[i+1],
                    reduction=tf.losses.Reduction.NONE)]

                self.msi_reg_error += [reg_msi_causes[i] * (self.msi_causes[i] ** 2)]
                # self.msi_reg_error += [tf.keras.regularizers.l2(reg_msi_causes[i])(self.msi_minibatch[i])]
                if len(self.msi_layers) > 1:
                    raise NotImplementedError
                else:
                    if available_modality is 'both':
                        self.msi_cause_grad += [
                            tf.gradients([self.msi_bu_error[i], self.msi_bu_error[i+1], self.msi_reg_error[i]],
                                                   self.msi_causes[i])[0]]
                    elif available_modality is 'visual':
                        self.msi_cause_grad += [tf.gradients([self.msi_bu_error[i], self.msi_reg_error[i]],
                                                             self.msi_causes[i])[0]]
                    elif available_modality is 'tactile':
                        self.msi_cause_grad += [tf.gradients([self.msi_bu_error[i + 1], self.msi_reg_error[i]],
                                                             self.msi_causes[i])[0]]

            else:
                raise NotImplementedError

        # add ops for updating causes
        self.m1_update_cause = []
        self.m2_update_cause = []
        self.msi_update_cause = []
        with tf.control_dependencies(self.m1_cause_grad + self.m2_cause_grad + self.msi_cause_grad):
            # m1 modality
            for i in range(len(self.m1_layers)):
                self.m1_update_cause += [tf.assign_sub(self.m1_causes[i], (lr_m1_causes[i] * self.m1_cause_grad[i]))]

            # m2 modality
            for i in range(len(self.m2_layers)):
                self.m2_update_cause += [tf.assign_sub(self.m2_causes[i], (lr_m2_causes[i] * self.m2_cause_grad[i]))]

            # msi modality
            for i in range(len(self.msi_layers)):
                self.msi_update_cause += [tf.assign_sub(self.msi_causes[i], (lr_msi_causes[i] * self.msi_cause_grad[i]))]


if __name__ == '__main__':

    sess, net = init_network(model_path, avail_modality)

    mms = init_minmaxscaler(tr_data_path)

    recon_temp_vis = np.zeros([ num_test_samps, 45, 80, 3 ])
    recon_temp_tac = np.zeros([ num_test_samps , 72 ])
    repr = np.zeros([num_test_samps, 100])


    for j in range (num_test_samps):
        visual_input, tactile_input = load_mat_data(ts_data_path, j, mms)
        if avail_modality is 'visual':
            tactile_input = np.zeros([1, 72])
        elif avail_modality is 'tactile':
            visual_input = np.zeros([1, 10800])
        reps, recon_vis, recon_tac = infer_repr(sess, net, max_iter, error_criterion, visual_input, tactile_input, True, avail_modality)
        repr[j, :] = reps[0]
        recon_temp_vis[j, :] = recon_vis.reshape(45,80,3) # reform into image
        recon_temp_tac[j, :] = recon_tac

    print('Done!')

    # create the save path if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    np.save(save_path + 'reps.npy', repr)
    sio.savemat(save_path + 'reps.mat', {'reps':repr})

    np.save(save_path + 'recon_vis.npy', recon_temp_vis)
    sio.savemat(save_path + 'recon_vis.mat', {'recon_vis':recon_temp_vis})

    np.save(save_path + 'recon_tac.npy', recon_temp_tac)
    sio.savemat(save_path + 'recon_tac.mat', {'recon_tac':recon_temp_tac})

    print('written output to ' + save_path )