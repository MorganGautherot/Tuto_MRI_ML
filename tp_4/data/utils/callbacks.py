from tensorflow.keras.callbacks import Callback
import os
import pandas as pd
import datetime

class SaveHyperparameters(Callback):

    def __init__(self, path, learning_rate, batch_size, test_percentage, translation, momentum, lr_decay, comments):
        hyperParameters = pd.DataFrame({'Hyperparameters' : ['date', 'learning_rate', 'batch_size', 'test_percentage', 'translation', 'momentum', 'lr_decay', 'comments'],
                     'Values' : [datetime.datetime.now(), learning_rate, batch_size, test_percentage, translation, momentum, lr_decay, comments]})
        hyperParameters.to_csv(path+'/hyperparameters.csv', index=False, header=False)

class SaveMetrics(Callback):

    def __init__(self, path):
        self.path = path
        data_metrics = pd.DataFrame([], columns= ['epoch', 'train_loss', 'val_loss'])
        data_metrics.to_csv(self.path+'/metrics.csv', index=False)

    def on_epoch_end(self, epoch, logs=None):
        data_metrics = pd.read_csv(self.path+'/metrics.csv')
        data_metrics = data_metrics.append(pd.DataFrame([[epoch,
                                                         logs['loss'],
                                                         logs['val_loss']]], columns= ['epoch', 'train_loss', 'val_loss']))
        os.system('sudo rm -rf '+self.path+'/metrics.csv')
        data_metrics.to_csv(self.path+'/metrics.csv', index=False)
