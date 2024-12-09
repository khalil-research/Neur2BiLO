import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import math


class NNmodel(object):
    def __init__(self, NNArchitecture, flag_ISNN, flag_doubleX, problem_str, data_dir):
        self.Architecture = NNArchitecture
        self.num_neuron_input = NNArchitecture[0]
        self.num_neuron_output = NNArchitecture[-1]
        self.num_neuron_hidden = NNArchitecture[1:-1]
        self.num_layer_hidden = len(self.num_neuron_hidden)
        self.flag_ISNN = flag_ISNN
        self.flag_doubleX = flag_doubleX
        if flag_ISNN:
            flag_ISNN = 'non_neg'
        else:
            flag_ISNN = None

        self.problem_str = problem_str

        self.fp_nn_model_fig = f'{data_dir}nn_figs/NNmodel_{self.problem_str}.png'
        if self.flag_ISNN == False:
            self.fp_nn_params = f'{data_dir}nn_params/NNparametersGNN_{self.problem_str}.xlsx'
        else:
            self.fp_nn_params = f'{data_dir}nn_params/NNparametersISNN__{self.problem_str}.xlsx'
        
        seednumber = 8
        z = [
            tf.keras.Input(shape=(self.num_neuron_input,))
        ]
        for i in range(self.num_layer_hidden):
            if i == 0:
                z += [
                    tf.keras.layers.Dense(self.num_neuron_hidden[i],
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seednumber),
                                          kernel_constraint=flag_ISNN,
                                          use_bias=True,
                                          bias_initializer='zeros',
                                          activation='relu'
                                          )(z[i])
                ]
            else:
                z += [
                    tf.keras.layers.Dense(self.num_neuron_hidden[i],
                                          kernel_initializer=tf.keras.initializers.glorot_normal(seed=seednumber),
                                          kernel_constraint=flag_ISNN,
                                          use_bias=True,
                                          bias_initializer='zeros',
                                          activation='relu'
                                          )(tf.keras.layers.concatenate([z[i], z[0]]))
                ]
        z += [
            tf.keras.layers.Dense(self.num_neuron_output,
                                  kernel_initializer=tf.keras.initializers.ones,
                                  kernel_constraint=flag_ISNN,
                                  use_bias=True,
                                  bias_initializer='zeros'
                                  )(tf.keras.layers.concatenate([z[self.num_layer_hidden], z[0]]))
        ]
        self.model = tf.keras.Model(z[0], z[-1])
        self.model.summary()
        tf.keras.utils.plot_model(self.model, self.fp_nn_model_fig, show_shapes=True)

        self.history = None
        self.label_max = None
        self.label_min = None
        self.w = None
        self.b = None
        self.err_max = None
        self.err_min = None

    def train(self, data):
        def my_loss_upper(y_true, y_pre):
            loss1 = tf.square(y_true - y_pre) / 2
            loss2 = tf.maximum(y_true - y_pre, 0)
            return loss1 + loss2

        def my_loss_lower(y_true, y_pre):
            loss1 = tf.square(y_true - y_pre) / 2
            loss2 = tf.maximum(y_pre - y_true, 0)
            return loss1 + loss2

        # data input
        num_x = len(data[1, :]) - 1
        sample = data[:, 0:num_x]
        label = data[:, num_x]
        self.label_max = max(label)
        self.label_min = min(label)
        label_nor = (label - self.label_min) / (self.label_max - self.label_min) * 1

        # train NN
        bs = max(1, 2 ** (round(math.log2(len(label))) - 6))
        self.model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001, decay=0.001),
                           loss='mse',
                           # loss=my_loss_upper,
                           # loss=my_loss_lower,
                           metrics=[tf.keras.metrics.MAE]  # , tf.keras.metrics.MAPE
                           )
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000, restore_best_weights=True, verbose=1)
        self.history = self.model.fit(sample if self.flag_doubleX == False else np.hstack([sample, 1 - sample]), label_nor,
                                 epochs=1000,
                                 batch_size=bs,
                                 verbose=1,
                                 validation_split=0.0,
                                 # validation_data=(x_val, y_val)
                                 callbacks=[callback]
                                 )
        # self.showTraining()
        self.err_max, self.err_min = self.evaluate(sample, label)

        # save NN
        w = []
        b = []
        for i in range(int(len(self.model.weights) / 2)):
            temp = np.transpose(self.model.weights[2 * i].numpy())
            w.append(temp)
            temp = self.model.weights[2 * i + 1].numpy()
            b.append(temp.reshape(-1, 1))
        self.w = w
        self.b = b
        self.save()
        return

    def save(self):
        
        writer = pd.ExcelWriter(self.fp_nn_params)
        for i in range(int(len(self.model.weights) / 2)):
            wrt = pd.DataFrame(self.w[i])
            wrt.to_excel(writer, 'kernel' + str(i + 1), header=None, index=False)
            wrt = pd.DataFrame(self.b[i])
            wrt.to_excel(writer, 'bias' + str(i + 1), header=None, index=False)
        wrt = pd.DataFrame([self.Architecture, [self.label_max, self.label_min], [self.err_max, self.err_min]])
        wrt.to_excel(writer, 'others', header=None, index=False)
        writer.close()
        print('successfully export NN parameters as', self.fp_nn_params)
        return

    def showTraining(self):
        # loass
        plt.figure()
        plt.plot(self.history.history['loss'])
        # plt.plot(history.history['val_loss'])
        plt.ylabel('Model loss')
        plt.xlabel('Epoch')
        # plt.ylim(0, 1e-8)
        plt.ylim(0, self.history.history['loss'][-300])
        plt.legend(['Train_loss', 'Val_loss'])
        plt.show()
        # MAE
        plt.figure()
        plt.plot(self.history.history['mean_absolute_error'])
        # plt.plot(history.history['val_mean_absolute_error'])
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.ylim(0, self.history.history['mean_absolute_error'][-300])
        plt.legend(['Train_MAE', 'Val_MAE'])
        plt.show()
        return

    def evaluate(self, sample, label):
        phi_pre = self.model.predict(sample if self.flag_doubleX == False else np.hstack([sample, 1-sample])) * (self.label_max - self.label_min) + self.label_min
        err = phi_pre[:, 0] - label
        err_max = np.max(err)
        err_min = np.min(err)
        err_relative = err / label * 100
        temp = np.vstack([phi_pre[:, 0], label, err, err_relative])
        print('max_abs_error:', np.round(np.max(abs(err)), 2))
        print('max_relative_error:', np.round(np.max(abs(err_relative)), 2), '%')
        # absolute_error
        plt.figure()
        plt.boxplot(err)
        plt.ylabel('absolute_error')
        # plt.show()
        # relative_error
        plt.figure()
        plt.boxplot(err_relative)
        plt.ylabel('relative_error (%)')
        # plt.show()
        return err_max, err_min

    def readParameters(self, FileName):
        data = pd.read_excel(io=FileName, sheet_name=None, header=None)
        self.Architecture = np.int_(data['others'].values[0])
        self.num_layer_hidden = len(self.Architecture) - 2
        w = []
        b = []
        for i in range(self.num_layer_hidden + 1):
            w += [data['kernel' + str(i + 1)].values]
            b += [data['bias' + str(i + 1)].values]
        self.w = w
        self.b = b
        self.label_max = data['others'].values[1][0]
        self.label_min = data['others'].values[1][1]
        self.err_max = data['others'].values[2][0]
        self.err_min = data['others'].values[2][1]
        return

    def predict(self, x_temp):
        z = [x_temp]
        for i in range(self.num_layer_hidden + 1):
            if i == 0:
                temp = self.w[i] @ z[i] + self.b[i][:, 0]
                z += [np.maximum(temp, 0)]
            else:
                temp = self.w[i] @ np.hstack([z[i], z[0]]) + self.b[i][:, 0]
                z += [np.maximum(temp, 0)]
        return z[-1][0] * (self.label_max - self.label_min) + self.label_min


def calculateNumHidden(num_samples, num_layer, num_input, flag_ISNN):
    if flag_ISNN:
        return math.ceil((num_samples/(num_input+1)-1)/num_layer)
    else:
        m = num_layer
        while m*(m+2)/4+(m+1)*(num_input+1) < num_samples:
            m += 1
        return math.ceil(m/num_layer)


def encoder(x):
    num_x = x.shape[0]
    if num_x % 10 != 0:
        print('error number of x')
        return None
    num_seg = num_x // 10
    temp = np.zeros((10, num_x))
    for i in range(10):
        temp[i, i*num_seg:(i+1)*num_seg] = 2**np.array(range(num_seg))
    return temp @ x

