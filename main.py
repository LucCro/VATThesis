#%% Importing Packages/Methods
import tensorflow as tf
import params
from data_pro import data_processing
from modelbuilder import ModelBuilder
from KL_divergence import KLD

#%% Parameters
batch_size = params.batch_size
epochs = params.epochs

CIFAR = params.CIFAr
dropout = params.dropout
batchNorm = params.batchNorm
activation = params.activation
lr = params.lr
whiten = params.whiten
SVHN = params.SVHn
semi = params.semi

from numpy.random import seed
seed(params.seedd)
tf.random.set_seed(params.seedd)

#%% Get Data
train_gen, val_gen, test_gen = data_processing.get_data(batch_size, cifar = CIFAR, svhn = SVHN, semi = semi, whiten = whiten)
#%% Make the model
modelbuild = ModelBuilder(dropout = dropout, batchNorm = batchNorm, activation = activation, cifar = CIFAR, svhn = SVHN)
model = modelbuild.get_model()  
#print(model.summary())

#%% Compile Model
def scheduler(epoch, lr):
    if epoch < 75:
        print(f'Learning rate: {round(lr,8)}')
        return lr
    else:              
        print(f'Learning rate: {round((lr - ((lr-0.00001)/48)*(epoch-75)),8)}')
        return lr - ((lr-0.0001)/48)*(epoch-75)
    
opt = tf.keras.optimizers.Adam(learning_rate = lr)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

# create metrics
metrics = [getattr(tf.keras.metrics, metric_class)(name = ('%s_%s' % (metric_type, metric_name)))
           for metric_type in ['sup', 'ladv', 'uladv', 'adv']
           for metric_class, metric_name in zip(['CategoricalAccuracy'], ['acc'])]

run_eagerly = params.runeager
    
model.compile(optimizer = opt, loss = KLD.kl_divergence,
    metrics = metrics, run_eagerly = run_eagerly)

#%% Train the model
print(' ')   
if CIFAR:
    print(f"Dataset: Cifar-10")
elif SVHN:
    print(f"Dataset: SVHN")
else:
    print('Dataset: MNIST')
    
print(f"Batch size: {batch_size}")
print(f"Whitening: {whiten}")
print(f"Regularization Parameter: {params.whiten_r}")
print(f"Xi: {params.xi}")
print(f'Organic VAT: {params.organic_vat}')
print(f"Semi-supervised: {semi}")
print(f'Epochs: {epochs}')
print(' ')
    
import time    
start = time.time()
 
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d_%m-%H.%M")

import os
root_dir = "/content/gdrive/My Drive/Thesis/Results/"
project_folder = f"CIFAR:{CIFAR}_SSL:{semi}_epochs:{epochs}_lr:{lr}_largemodel:{params.model_large}_iterations:{params.iterations}_onehot:{params.one_hot}_datetime:{dt_string}/"
if os.path.isdir(root_dir + project_folder) == False:
  os.mkdir(root_dir + project_folder)
  print(root_dir + project_folder + ' did not exist but was created.')
folder = root_dir+project_folder
path = f"{folder}/training_metrics.csv"

csv_logger = tf.keras.callbacks.CSVLogger(filename = path, separator=",", append = False)

mcp_save = tf.keras.callbacks.ModelCheckpoint(f'{folder}/model.hdf5', save_best_only=True, monitor='val_loss', mode='min')

callback = [lr_scheduler,csv_logger,mcp_save]

history = model.fit(x = train_gen,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = val_gen,
                    callbacks=callback)

print('Training time: %.1f seconds.' % (time.time() - start))
#%% Evaluate the model on the test set
metric_values = model.evaluate(test_gen)

print(' ')   
if CIFAR==True:
    print(f"Dataset: Cifar-10")
elif SVHN==True:
    print(f"Dataset: SVHN")
else:
    print('Dataset: MNIST')
    
print(f"Batch size: {batch_size}")
print(f"Whitening: {whiten}")
print(f"Regularization Parameter: {params.whiten_r}")
print(f"Xi: {params.xi}")
print(f"Semi-supervised: {semi}")
print(f'Epochs: {epochs}')
print(' ')

for metric_name, metric_value in zip(model.metrics_names, metric_values):
    print('%s: %.3f' % (metric_name, metric_value))
    
#%%
import pandas as pd
metric_df = pd.DataFrame()
metric_df['metric'] = model.metrics_names
metric_df['metric_values'] = metric_values
metric_df.to_csv(path_or_buf = f'{folder}test_metrics.csv', index = False)

#%%
#import pandas as pd
#metric_df = pd.read_csv('training.csv')
#metric_df.plot(x='epoch', y='val_sup_acc', style='-')
#metric_df.plot(x='epoch', y='loss', style='-')