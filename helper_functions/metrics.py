import numpy as np

def normalized_mae(preds, truth, average):
  """
  evaluates NMAE (from paper)
  preds: predictions from NN
  truth: ground truth labels
  average: average of dataset
  output: NMAE value (depends on used average)
  """

  # fix dims if only one sample present
  if len(preds.shape) == 1:
    preds = np.array([preds])
    truth = np.array([truth])

  # actual computation of nmae for this subdataset
  nmae = np.average(np.abs((preds-truth) / average), axis=0)
  return nmae



def r_squared(preds, truth, variance):
  """
  evaluate the R^2 metric for some predictions and ground truth
  preds: predictions from NN
  truth: labels for the preds
  variance: variance of dataset
  output: R^2 metric (depends on used variance)
  """ 
  return 1 - (np.mean(np.square(preds-truth), axis=0) / variance)