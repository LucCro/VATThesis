import tensorflow as tf
from KL_divergence import KLD
import params

# Adverarial attacks
class Attacks():
    def virtual_L2_attack(model, x, y):
        """
        Compute a virtual adversarial attack as in the original VAT paper.
    
        Parameters
        ----------
        x : tensor
            Clean example.
        y : tensor
            Target: should be the model prediction w.r.t. x.
    
        Returns
        -------
        tensor
            Virtual adversarial example.
    
        """
        
        noise = tf.random.normal(tf.shape(x))
        noise /= tf.norm(noise, 2, axis = 1, keepdims = True)
        
        
        xi = params.xi
        inp = x + noise*xi
        
        # this can greatly increase the effectiveness of a VAT attack
        # when uncommented
        
        num_classes = params.num_classes
        
        if params.one_hot:
            y = tf.one_hot(tf.keras.backend.argmax(y), depth = num_classes)        
        
        
        return Attacks.L2_attack(model, inp, y) - inp + x
    
    def L2_attack(model, x, y):
        """
        Compute a regular L2 attack.
    
        Parameters
        ----------
        x : tensor
            Clean example.
        y : tensor
            Target label.
    
        Returns
        -------
        x_adv : tensor
            Adversarial example.
    
        """
        
        
        with tf.GradientTape() as tape:
    
            tape.watch(x)
    
            pred = model(x)
            loss_value = KLD.kl_divergence(y, pred)
    
        grads = tf.stop_gradient(tape.gradient(loss_value, x))
        
        # clipping is important since gradients can be very small,
        # particularly with the VAT attack
        radv = grads / tf.keras.backend.clip(tf.norm(grads, 2, axis = 1, keepdims = True), 1e-6, 1e+6) 
        
        
        epsilon = params.epsilon 
        x_adv = x + epsilon * radv
    
        return x_adv