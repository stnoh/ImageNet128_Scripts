import pickle
import numpy as np

def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo)
    fo.close()
    return dict['data'], dict['labels']

def cifar_generator(filenames, batch_size, data_dir):
    def get_epoch():
        np.random.shuffle(filenames)
        for filename in filenames:        
            data, labels = unpickle(data_dir + '/' + filename)
            labels = np.array(labels)-1
            indices = np.arange(len(labels))
            np.random.shuffle(indices)
            for i in range(data.shape[0] // batch_size):
                yield data[indices[i * batch_size:(i + 1) * batch_size]], labels[indices[i * batch_size:(i + 1) * batch_size]]
    return get_epoch


def load(mode, batch_size, data_dir):
    if mode=='TRAIN':
        return cifar_generator(['train_data_batch_%i' % (i + 1) for i in range(100)], batch_size, data_dir)
    else:
        return cifar_generator(['val_data_batch_%i' % (i + 1) for i in range(10)], batch_size, data_dir)

