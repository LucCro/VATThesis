import tensorflow as tf

class KLD():  
    def kl_divergence(y_true, y_pred):
        return tf.keras.backend.mean(tf.keras.losses.kl_divergence(y_true, y_pred))