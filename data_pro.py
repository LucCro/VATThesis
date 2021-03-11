import tensorflow as tf
import numpy as np
from DataGenerator import SimpleSequence
from zca_whitening import ZCA
import params
import random

class data_processing:
    def get_data(batch_size = 128, cifar = True, svhn= False, semi = True, whiten = False):
        if cifar:
            #Data split parameters
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
            x = np.concatenate((x_train, x_test)).astype('float32')
            x /= 255
            
            if whiten:
                print('Whitening...')
                train_data_flat = x.reshape(x.shape[0], -1).T
                zca = ZCA(D=train_data_flat, n_components=train_data_flat.shape[1])
                train_data_flat = zca.transform(D=train_data_flat, whiten=True, ZCA=True)
                train_data2 = train_data_flat.T.reshape(x.shape)  
                x = train_data2
            
            import matplotlib.pyplot as plt
            def NormalizeData(data):
                return (data - np.min(data)) / (np.max(data) - np.min(data))
            plt.figure()
            plt.imshow(NormalizeData(x)[1]) 
            plt.show()
            
            y = np.concatenate((y_train, y_test))
            y = tf.keras.utils.to_categorical(y)
            
            
            random.seed(params.seedd)
            index_list = (list(range(60000)))
            random.shuffle(index_list)
            
            # train-validation-test split
            if semi == True:
              data_split = {'trainIDs': index_list[:49000], 'valIDs': index_list[49000:50000], 'testIDs': index_list[50000:60000]}
              print(data_split['trainIDs'][:5])
            else:
              data_split = {'trainIDs': index_list[:4000], 'valIDs': index_list[49000:50000], 'testIDs': index_list[50000:60000]}
            # an indicator array indicatig whether a training example is labeled
            labeled = np.ones((60000, ), dtype = bool)
            
            if semi == True:
                labeled[4000:40000, ...] = False
            
            data = {'x': x, 'y': y, 'labeled': labeled}
            
            def get_data_subset(data, split, subset):
                """
                Select training, validation or testing portion of the data.
                """
                
                return {arr: data[arr][split[subset + 'IDs']] for arr in ['x', 'y', 'labeled']}     
            
            
            train_gen = SimpleSequence(data_split['trainIDs'], batch_size,
                                  data = get_data_subset(data, data_split, 'train'))
        
            val_gen = SimpleSequence(data_split['valIDs'], batch_size,
                                  data = get_data_subset(data, data_split, 'val'))
        
            test_gen = SimpleSequence(data_split['testIDs'], batch_size,
                                  data = get_data_subset(data, data_split, 'test'))
            
            return train_gen, val_gen, test_gen
        else:
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            
            x = np.concatenate((x_train, x_test)).reshape(70000, 784).astype('float32')
            y = np.concatenate((y_train, y_test))
            
            # normalize images
            x /= 255
            
            # class numbers -> one-hot representation
            y = tf.keras.utils.to_categorical(y)
            
            # train-validation-test split
            data_split = {'trainIDs': range(50000), 'valIDs': range(50000, 60000), 'testIDs': range(60000, 70000)}
            
            # an indicator array indicatig whether a training example is labeled
            # (in the training set, first 10,000 samples are labeled)
            labeled = np.ones((70000, ), dtype = bool)
            labeled[10000:50000, ...] = False
            data = {'x': x, 'y': y, 'labeled': labeled}
    
            def get_data_subset(data, split, subset):
                """
                Select training, validation or testing portion of the data.
                """
                
                return {arr: data[arr][split[subset + 'IDs']] for arr in ['x', 'y', 'labeled']}
        
            train_gen = SimpleSequence(data_split['trainIDs'], batch_size,
                                  data = get_data_subset(data, data_split, 'train'))
        
            val_gen = SimpleSequence(data_split['valIDs'], batch_size,
                                  data = get_data_subset(data, data_split, 'val'))
        
            test_gen = SimpleSequence(data_split['testIDs'], batch_size,
                                  data = get_data_subset(data, data_split, 'test'))
            return train_gen, val_gen, test_gen