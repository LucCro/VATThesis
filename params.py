import tensorflow as tf

batch_size = 128
num_classes = 10
epochs = 123
xi = 1e-6
alpha = 1
epsilon = 2
lr = 0.001
whiten_r = 1e-1 
dropout = 0.5
seedd = 1

CIFAr = True
SVHn = False
whiten = False 
activation = tf.keras.layers.LeakyReLU(alpha=0.1)
batchNorm = True
one_hot = True
organic_vat = True
semi = True
model_large = False
iterations = 10
step_size = 0.1
runeager = False