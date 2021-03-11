import tensorflow as tf
from VATAttacks import Attacks
import params 

class VATModel(tf.keras.Model):
    #def __init__(self):
     #   super(GradModel, self).__init__()
    
    def compile(self, optimizer, loss, metrics = [], run_eagerly = False):
        """
        Compile the model.

        Parameters
        ----------
        optimizer : a keras optimizer
            A keras optimizer. See tf.keras.optimizers. 
        loss : TF function
            A loss function to be used for supervised and unsupervised terms.
        metrics : a list of keras metrics, optional
            Metrics to be computed for labeled and unlabeled adversarial or clean examples.
            See self.update_metrics to see how they are handled.
        run_eagerly : bool, optional
            If True, this Model's logic will not be wrapped in a tf.function;
            one thus can debug it more easily (e.g. print inside train_step).
            The default is False.

        Returns
        -------
        None.

        """
        super(VATModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss
        self.loss_trackers = [tf.keras.metrics.Mean(name = 'loss'),
                              tf.keras.metrics.Mean(name = 'loss_sup'),
                              tf.keras.metrics.Mean(name = 'loss_vat')]
        self.vat_metrics = metrics
        
        self._run_eagerly = run_eagerly
    
    
    def unpack_data(self, data):
        """
        Convert data from the genrator into a more convenient form for SSL.

        Parameters
        ----------
        data : tuple
            Tuple: x: images, y: labels, labeled: binary array indicating labeled images.
        
        """
        # x: images, y: labels, labeled: binary array indicating labeled images.
        # This format was chosen to fit dimensions keras considers valid.
        x, y, labeled = data
        
        n_labeled = tf.math.count_nonzero(labeled)
        n = tf.shape(x)[0]
        
        # select labeled images
        xl = x[:n_labeled, ...]
        yl = y[:n_labeled, ...]

        xul = x[n_labeled:, ...]
        
        return x, xl, yl, xul, n_labeled, n
    
    def compute_loss(self, data):
        """
        Compute total VAT loss:
            supervised + supervised adversarial + unsupervised adversarial.

        Parameters
        ----------
        data : tuple
            The output of the generator.

        Returns
        -------
        loss_value : scalar
            Total loss.
        loss_sup : scalar
            Supervised loss.
        loss_vat : scalar
            Adversarial loss (supervised + unsupervised).
        pred : tensor
            Predictions on clean examples (for computing metrics).
        pred_adv : tensor
            Predictions on adversarial examples (= 'adversarial predictions')
            (for computing metrics).

        """
        
        x, xl, yl, xul, n_labeled, n = self.unpack_data(data)

        # compute predictions on clean exaples
        pred = self(x)
        predl  = pred[:n_labeled, ...]
        predul = pred[n_labeled:, ...]
        # supervised loss
        loss_sup = self.loss(yl, predl)   
                       
        # supervised adversarial examples
        if params.organic_vat:
            predl = tf.stop_gradient(predl)
            xl_adv = Attacks.virtual_L2_attack(self, xl, predl)
        else:
            xl_adv = Attacks.L2_attack(self, xl, yl)
                       
        # virtual adversarial examples
        predul = tf.stop_gradient(predul)             
        xul_adv = Attacks.virtual_L2_attack(self, xul, predul)
                
        x_adv = tf.concat((xl_adv, xul_adv), axis = 0)
        y_adv = tf.concat((yl, predul), axis = 0)
            
        pred_adv = self(x_adv)
        loss_vat = self.loss(y_adv, pred_adv)
        
        # total loss        
        alpha = params.alpha             
        loss_value = loss_sup + alpha * loss_vat
        
        return loss_value, loss_sup, loss_vat, pred, pred_adv
    
    def update_metrics(self, loss_values, yl, pred, pred_adv):
        """
        Updates loss trackers and metrics so that they return the current moving average.

        """
        
        # update all the loss trackers with current batch loss values
        for loss_tracker, loss_value in zip(self.loss_trackers, loss_values):
            loss_tracker.update_state(loss_value)

        # separate predictions into labeled and unlabeled subsets
        # for metric computation        
        n_labeled = tf.shape(yl)[0]
        predl, predul = pred[:n_labeled, ...], pred[n_labeled:, ...]
        predl_adv, predul_adv = pred_adv[:n_labeled, ...], pred_adv[n_labeled:, ...]
        
        # for every metric type
        # sup:      metrics on the labeled subset measuring GT vs clean prediction fidelity
        # ladv:     metrics on the labeled subset measuring GT vs adversarial prediction fidelity
        # uladv:    metrics on the unlabeled subset measuring clean vs adversarial prediction fidelity
        # adv:      metrics on the entire batch measuring clean vs adversarial prediction fidelity
        for metric_type, y_true, y_pred in zip(['sup', 'ladv', 'uladv', 'adv'],
                                             [yl, yl, predul, pred],
                                             [predl, predl_adv, predul_adv, pred_adv]):
            
            for metric in self.vat_metrics:
                
                # if metric name contains the type name
                if metric_type in metric.name.split('_'):
                    metric.update_state(y_true, y_pred)
            
        return {m.name: m.result() for m in self.metrics}
    
    def train_step(self, data):
        """
        This method is called by model.fit() for every batch.
        It should compute gradients, update model parameters and metrics.

        Parameters
        ----------
        data : tuple
            Batch received from the generator.

        Returns
        -------
        metric_values : dictionary
            Current values of all metrics (including loss terms).

        """
        
        x, xl, yl, xul, n_labeled, n = self.unpack_data(data)

        # compute gradient wrt parameters
        with tf.GradientTape() as tape:
            loss_value, loss_sup, loss_vat, pred, pred_adv = self.compute_loss(data)

        grads = tape.gradient(loss_value, self.trainable_variables)
        
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))        
        
        metric_values = self.update_metrics([loss_value, loss_sup, loss_vat], yl, pred, pred_adv)

        return metric_values
        
    def test_step(self, data):
        """
        This method is called by model.fit() during the validation step
        and by model.evaluate().

        """
        
        x, xl, yl, xul, n_labeled, n = self.unpack_data(data)
        
        loss_value, loss_sup, loss_vat, pred, pred_adv = self.compute_loss(data)
        
        metric_values = self.update_metrics([loss_value, loss_sup, loss_vat], yl, pred, pred_adv)
        
        return metric_values

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return self.loss_trackers + self.vat_metrics