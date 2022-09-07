import numpy as np
from tensorflow.python.ops.math_ops import _truediv_python3

# Scale inputs using global class
class ScalingOperations:

  # save scaled and unscaled versions of array & max and min upon init
  def __init__(self, unscaled_arr):
    self.unscaled_arr = np.array(unscaled_arr)
    self.ds_size,_ = unscaled_arr.shape
    self.arr_min = np.min(self.unscaled_arr, axis=0)
    self.arr_max = np.max(self.unscaled_arr, axis=0)
    self.scaled_arr = ScalingOperations.map_to_01(self, unscaled_arr)

    # just mae things :*
    min_over_delta = (self.arr_min) / (self.arr_max - self.arr_min)
    self.mae_scaled_avg = np.mean(self.scaled_arr, axis=0) + min_over_delta
    self.mae_unscaled_avg = np.mean(self.unscaled_arr, axis=0)

    scaled_mean_repeat = np.repeat([np.mean(self.scaled_arr, axis=0)], self.ds_size, axis=0)
    self.scaled_variance = np.mean(np.square(self.scaled_arr - scaled_mean_repeat), axis=0) #(avg of (pred-truth)^2)/(variance)
   
    unscaled_mean_repeat = np.repeat([np.mean(self.unscaled_arr, axis=0)], self.ds_size, axis=0)
    self.unscaled_variance = np.mean(np.square(self.unscaled_arr - unscaled_mean_repeat), axis=0)
    
  # map dataset to 01
  def map_to_01(self, unscaled_arr):

    ds_size,_ = unscaled_arr.shape
    # expand dimensions to match larger array
    max_repeat = np.repeat([self.arr_max], ds_size, axis=0)
    min_repeat = np.repeat([self.arr_min], ds_size, axis=0)

    return (unscaled_arr - min_repeat) / (max_repeat - min_repeat)

  def map_to_original(self, scaled):

    # check that the array corresponds to this object
    # more complicated than expected
    # if verify==True:
    #   print("Verification not yet implemented")
    ds_size,_ = scaled.shape

    # expand dimensions to match larger array
    max_repeat = np.repeat([self.arr_max], ds_size, axis=0)
    min_repeat = np.repeat([self.arr_min], ds_size, axis=0)

    return (scaled * (max_repeat - min_repeat)) + min_repeat

  def normalized_mae(self, preds, truth, scaled=False):

    # fix dims if only one sample present
    if len(preds.shape) == 1:
      preds = np.array([preds])
      truth = np.array([truth])

    # choose the correct average for mae, depending on input data
    ds_average = self.mae_unscaled_avg
    if scaled:
      ds_average = self.mae_scaled_avg

    # actual computation of nmae for this subdataset
    nmae = np.average(np.abs((preds-truth) / ds_average), axis=0)
    return nmae

  def r_squared(self, preds, truth, scaled=True):
    if scaled == True:
      return 1 - (np.mean(np.square(preds-truth), axis=0) / self.scaled_variance)
    else:
      return 1 - (np.mean(np.square(preds-truth), axis=0) / self.unscaled_variance)






