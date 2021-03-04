from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D 
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
from VATSetup import VATModel  
import params

# Specify model architecture
class ModelBuilder:   
    
    def __init__(self, dropout = False, batchNorm = True, activation = 'relu', cifar = True, svhn = False):
        self.dropout = dropout
        self.batchNorm = batchNorm
        self.activation = activation
        self.cifar = cifar

    def get_model(self):
        if self.cifar:
            if params.model_large == False:
                self.activation = params.activation
                
                inp = tf.keras.Input(shape=(32, 32, 3))
                x = Conv2D(96, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3))(inp)
                x = BatchNormalization()(x)
                x = Conv2D(96, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(96, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                
                x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
                
                x = Dropout(self.dropout)(x)
                
                x = Conv2D(192, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(192, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(192, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                
                x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
                
                x = Dropout(self.dropout)(x)
                
                x = Conv2D(192, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(192, (1, 1), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(192, (1, 1), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                
                x = GlobalAveragePooling2D()(x) 
                
                output = Dense(10, activation='softmax')(x)
                
                model = VATModel(inputs=[inp], outputs=[output])
                
                return model
        
            else:
                self.activation = params.activation
                
                inp = tf.keras.Input(shape=(32, 32, 3))
                x = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3))(inp)
                x = BatchNormalization()(x)
                x = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(128, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                
                x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
                
                x = Dropout(self.dropout)(x)
                
                x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(256, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                
                x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
                
                x = Dropout(self.dropout)(x)
                
                x = Conv2D(512, (3, 3), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(256, (1, 1), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                x = Conv2D(128, (1, 1), activation=self.activation, kernel_initializer='he_uniform', padding='same')(x)
                x = BatchNormalization()(x)
                
                x = GlobalAveragePooling2D()(x) 
                
                output = Dense(10, activation='softmax')(x)
                
                model = VATModel(inputs=[inp], outputs=[output])
                
                return model
        else:
            inp = tf.keras.Input(shape=(784,))
            x = tf.keras.layers.Dense(1200, activation=tf.nn.relu)(inp)
            x = tf.keras.layers.Dense(600, activation=tf.nn.relu)(x)
            x = tf.keras.layers.Dense(300, activation=tf.nn.relu)(x)
            x = tf.keras.layers.Dense(150, activation=tf.nn.relu)(x)
            output = tf.keras.layers.Dense(10, activation='softmax')(x)
            
            model = VATModel(inputs=[inp], outputs=[output])
            
            return model   