import math
import numpy as np
import tensorflow as tf
import random
import metrics
from scipy import linalg as la
from scipy import signal


class Processing:
  """
  Class that encompasses all processing for data
  """

  def __init__(
    self, 
    imgs, 
    labs, 
    splits, 
    custom_test=None,
    custom_valid=None,
    batch_size=64, 
    conv_behavior=True, 
    scale_labels=True, 
    symmetric=True, 
    twopoint=False, 
    shuffle=True, 
    pca=False,
    pca_variance=0.9,
    pca_dims=None
    ):
    """
    Initialization of object, some processing
    """

    if np.max(imgs) > 2:
      imgs = imgs / imgs.shape[1]
    print("Max of ds is", np.max(imgs))

    # images and labels that are after init processed according to settings
    self.images = np.array(imgs)
    self.labels = np.array(labs)

    
    # images and labels corresponding to original data
    self.raw_images = np.copy(self.images)
    self.raw_labels = np.copy(self.labels)

    # determine size of one side
    sq = math.sqrt(self.raw_images.shape[1])
    print("WARNING: Only 2D samples are supported")
    assert sq == int(sq), "Images are not square. Support not yet implemented"
    self.sample_size = int(sq)

    self.batch_size = batch_size
    self.dataset_size = self.images.shape[0]
    self.custom_test = custom_test
    self.custom_valid = custom_valid

    # two point correlation transformation
    if twopoint:
      # print(twopoint)
      self.images = Processing.two_point_dataset(self)
    
    if pca:
      self.images, dim_number = Processing.pca_dataset(self.images, frac_variance=pca_variance, number_dims=pca_dims)
      self.sample_size = int(math.sqrt(dim_number))

    # scaling vars
    self.arr_min = np.min(self.raw_labels, axis=0)
    self.arr_max = np.max(self.raw_labels, axis=0)

    assert np.sum(splits) == 1, "splits do not add to 1"
    self.splits = (np.array(splits) * self.dataset_size).astype(int)
    

    # boolean indicator vars
    self.conv_behavior = conv_behavior
    self.scale_labels = scale_labels
    self.symmetric = symmetric
    self.shuffle = shuffle

    # test some properties
    assert self.images.shape[0] == self.labels.shape[0], "Number of labels and images not equal"
    assert len(splits) == 3, "More than three splits given"
    assert self.batch_size > 0, "Non-positive batch size given"

    # shuffle data if required
    if self.shuffle:
      Processing.shuffle_data(self)
    
    # scale labels, symmetrize and create dataset
    if self.scale_labels:
      self.labels = Processing.scale_labels_to_01(self, self.labels)
    if self.symmetric:
      self.labels = Processing.nine_to_six(self, self.labels)
    Processing.create_datasets(self)
    return



  def shuffle_data(self):
    """
    shuffles the images and labels such that the corresponding image and label
    pairs are still at the same position in the shuffled arrays
    """
    permutation = np.random.permutation(self.dataset_size)

    # for standard imgs to use
    np.take(self.images,permutation,axis=0,out=self.images)
    np.take(self.labels,permutation,axis=0,out=self.labels)

    # for background original imgs for comparison
    np.take(self.raw_images,permutation,axis=0,out=self.raw_images)
    np.take(self.raw_labels,permutation,axis=0,out=self.raw_labels)
    return
  


  def create_datasets(self, reshuffle=False):
    """
    creates train, val and test tensorflow datasets
    self: object
    reshuffle: to reshuffle data if you want to
    """

    # not happy with the distribution?
    # don't worry, reshuffle is here for you
    if reshuffle:
      Processing.shuffle_data(self)

    split1 = self.splits[0]
    split2 = split1 + self.splits[1]

    train_imgs = self.images[:split1]
    valid_imgs = self.images[split1:split2]
    test_imgs = self.images[split2:]

    train_labs = self.labels[:split1]
    valid_labs = self.labels[split1:split2]
    test_labs = self.labels[split2:]

    if self.custom_test != None:
      test_imgs, test_labs = self.custom_test
      if self.symmetric:
       test_labs = Processing.nine_to_six(self, test_labs)

    if self.custom_valid != None:
      valid_imgs, valid_labs = self.custom_test
      if self.symmetric:
        valid_labs = Processing.nine_to_six(self, valid_labs)

    # handy dandy tf.split splits data into three arrays
    # train_imgs, valid_imgs, test_imgs = tf.split(self.images, self.splits, axis=0)
    # train_labs, valid_labs, test_labs = tf.split(self.labels, self.splits, axis=0)


    merge = np.concatenate((train_imgs, valid_imgs))
    print(merge.shape[0], np.unique(merge, axis=0).shape[0])
    # assert merge.shape[0] == np.unique(merge, axis=0).shape[0]

    self.train_labs = train_labs
    self.valid_labs = valid_labs
    self.test_labs = test_labs

    # for shorter code, predefine long function
    create_ds = tf.data.Dataset.from_tensor_slices

    raw_train_labs = create_ds(train_labs)
    raw_valid_labs = create_ds(valid_labs)
    raw_test_labs = create_ds(test_labs)
    
    # depending on the desired format the datasets have different shapes
    if self.conv_behavior:
      # create ds and reshape pure images
      raw_train_imgs = create_ds(train_imgs).map(lambda x: x.reshape((self.sample_size, self.sample_size, 1)))
      raw_valid_imgs = create_ds(valid_imgs).map(lambda x: x.reshape((self.sample_size, self.sample_size, 1)))
      raw_test_imgs = create_ds(test_imgs).map(lambda x: x.reshape((self.sample_size, self.sample_size, 1)))

    else:
      # only create ds
      raw_train_imgs = create_ds(train_imgs)
      raw_valid_imgs = create_ds(valid_imgs)
      raw_test_imgs = create_ds(test_imgs)
      print("WARNING: Datasets for non-convolutional networks not tested yet")

    # create datasets
    self.ds_train = tf.data.Dataset.zip((raw_train_imgs, raw_train_labs))
    self.ds_val = tf.data.Dataset.zip((raw_valid_imgs, raw_valid_labs))
    self.ds_test = tf.data.Dataset.zip((raw_test_imgs, raw_test_labs))
      
    
    # batch datasets
    self.ds_train = self.ds_train.batch(self.batch_size)
    self.ds_val = self.ds_val.batch(self.batch_size)
    self.ds_test = self.ds_test.batch(self.batch_size)
    

    # create datasets for autoencoder
    self.datasets_auto = (
      tf.data.Dataset.zip((raw_train_imgs, raw_train_imgs)).batch(self.batch_size),
      tf.data.Dataset.zip((raw_valid_imgs, raw_valid_imgs)).batch(self.batch_size),
      tf.data.Dataset.zip((raw_test_imgs, raw_test_imgs)).batch(self.batch_size)
    )

    # its all saved as attributes, but you can have it here too :)
    return self.ds_train, self.ds_val, self.ds_test



  def get_input_shape(self):
    """
    get the input_shape to pass to NN
    """
    if self.conv_behavior:
      return (self.sample_size, self.sample_size, 1)
    else:
      return (None, self.batch_size, self.sample_size * self.sample_size)
  
  def get_output_shape(self):
    """
    get the output_shape to pass to NN
    """
    if self.symmetric:
      return 6
    else:
      return 9

  def nine_to_six(self, matrices):
    """
    from array of matrices only save symmetric matrix in vector6
    matrices: array of 3x3 matrices or vectors of size 9
    output: array of [C11, C22, C33, C12, C13, C23] shaped vectors
    """

    set_size = matrices.shape[0]

    # Back to normal vectors, and create nine arrays of length dataset
    matrices = np.array(matrices).reshape(set_size, 9).T

    # Extract the relevant dimensions
    reduced = matrices[[True, True, True, False, True, True, False, False, True]].T

    # reorder the relevant dimensions
    reduced = reduced.T[[0,3,5,1,2,4]].T
    return reduced



  def six_to_nine(self, matrices, output_as_matrix=False):
    """
    from array of vector6 create full C matrix
    matrices: array of [C11, C22, C33, C12, C13, C23] shaped vectors
    output_as_matrix: Do you want 3x3?
    output: 
    """
    
    matrices = np.array(matrices)
    set_size = matrices.shape[0]
    
    # expand the entries to original 9D vectors
    expanded = matrices.T[[0,3,4,3,1,5,4,5,2]].T

    # do you want matrices or vectors?
    if output_as_matrix:
      return expanded.reshape(set_size, 3, 3)
    else:
      return expanded



  def scale_labels_to_01(self, matrices):
    """
    scale some labels dimension-wise between [0,1] using the dimension-wise 
    maxima and minima of the entire dataset
    matrices: input as optional subset
    output: scaled labels
    """

    ds_size = matrices.shape[0]
    # expand dimensions to match larger array
    max_repeat = np.repeat([self.arr_max], ds_size, axis=0)
    min_repeat = np.repeat([self.arr_min], ds_size, axis=0)

    return (matrices - min_repeat) / (max_repeat - min_repeat)

  

  def scale_labels_to_original(self, matrices):
    """
    scale labels between [0,1] dimension-wise back to the original
    using dimension-wise max and min of the entire dataset
    matrices: input as optional subset
    output: unscaled labels
    """

    # check that the array corresponds to this object
    # more complicated than expected
    # if verify==True:
    #   print("Verification not yet implemented")
    ds_size = matrices.shape[0]

    # expand dimensions to match larger array
    max_repeat = np.repeat([self.arr_max], ds_size, axis=0)
    min_repeat = np.repeat([self.arr_min], ds_size, axis=0)

    return (matrices * (max_repeat - min_repeat)) + min_repeat



  def process_for_eval(self, matrices):
    """
    Processes a matrix output by the nn to the shape required for evaluation
    as per the global initial settings
    matrices: matrices to be processed
    output: Array of 9D vectors in original scales
    """
    matrices = np.array(matrices)
    if self.symmetric:
      matrices = Processing.six_to_nine(self, matrices)
    if self.scale_labels:
      matrices = Processing.scale_labels_to_original(self, matrices)
    
    return matrices



  def get_one_pair(self, test_set=False, processed_label=True):
    """
    finds a pair according to the params set and returns it
    test_set: whether the pair should be in the test set
    processed_label: label in raw form or in processed form
    output: (image, label) tuple
    """
    if test_set == True:
      single_set = self.ds_test.unbatch().batch(1).shuffle(self.splits[2]).take(1)
      img, label = single_set.as_numpy_iterator().next()
    else:
      index = random.randrange(0, self.dataset_size+1)
      img = self.images[index]
      label = self.labels[index]
    
    if not processed_label:
      label = Processing.process_for_eval(self, [label])
      label = np.squeeze(label)

    return img, label

  

  def get_eval_label_average(self):
    """
    Gets the dimension-wise label average for unprocessed labels
    output: Dimension-wise label average (unprocessed)
    """
    return np.average(self.raw_labels, axis=0)

  

  def get_eval_label_variance(self):
    """
    Gets the dimension-wise label variance for unprocessed labels
    output: Dimension-wise label variance (unprocessed)
    """
    return np.var(self.raw_labels, axis=0)

  def eval_test_nmae(self, preds):
    """
    functionality hiding, directly evaluates nmae for the test set
    preds: predictions of the test set
    output: dimension-wise nmae
    """
    eval_test = Processing.process_for_eval(self, self.test_labs)
    return metrics.normalized_mae(
      preds, 
      eval_test, 
      Processing.get_eval_label_average(self)
      )



  def eval_test_r_squared(self, preds):
    """
    functionality hiding, directly evaluates R^2 for the test set
    preds: predictions of the test set
    output: dimension-wise R^2
    """

    eval_test = Processing.process_for_eval(self, self.test_labs)
    return metrics.r_squared(
      preds,
      eval_test,
      Processing.get_eval_label_variance(self)
      )

  @staticmethod
  def two_point_correlation(image):
    s = image.shape[0]
    f = np.zeros(s)
    r = 0
    inv_image = 1 - image
    for r in range(0,s):
      fhr = 0
      for y in range(0,s):
        fhr = fhr + image[y] * inv_image[(r+y)%s]
      f[r] = fhr / s
    
    return f
  
  def fast_2pc(image):

    # image = image.reshape(self.sample_size, self.sample_size)
    inv_image = image
    # print(image.shape)

    # old sample size
    s = image.shape[0]

    """
    # old method:
    double_image = np.concatenate((image, image))[:-1]
    res = np.convolve(double_image, inv_image, 'valid') / s
    """
    # new method using scipy.signal
    two_pc = signal.correlate2d(image, inv_image, mode='same') / s
    
    # reshape back to 1D
    res_shape = two_pc.shape
    # two_pc = two_pc.reshape(self.sample_size ** 2)

    # save new sample size
    # self.sample_size = res_shape[0]

    return two_pc

  def two_point_dataset(self):
    
    # make square
    imgs = self.images.reshape(self.dataset_size, self.sample_size, self.sample_size)
    
    # apply to each element
    result = Processing.fast_2pc(imgs[0])
    count = 1
    for x in imgs[1:]:
      r = Processing.fast_2pc(x)
      result = np.concatenate((result, r))

      count += 1
      if (100 * count / self.dataset_size) % 10 == 0:
        print (count, " out of ", self.dataset_size)


    # result = np.apply_along_axis(Processing.fast_2pc, 2, imgs)

    # update datapoint size
    res_shape = result.shape
    self.sample_size = res_shape[1]

    # unmake square
    result = result.reshape(self.dataset_size, self.sample_size ** 2)

    return result



  # Source: https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python
  @staticmethod
  def pca_dataset(data, frac_variance=0.9, number_dims=None):
    """
    returns: data transformed in 2 dims/columns + regenerated original data
    pass in: data as 2D NumPy array
    """
    
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]

    
    eval_total = np.sum(evals)
    eval_sum = 0
    dimcount = 0
    for x in evals:
      eval_sum += x
      dimcount += 1
      if eval_sum / eval_total >= frac_variance:
        break

    # next larger square
    dimcount = int(math.ceil(math.sqrt(dimcount)) ** 2)
    assert dimcount < n, "Something is wrong in PCA..."
    print("Keeping ", dimcount, "out of ", n, "dimensions")
    
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    red_evecs = evecs[:, :dimcount]

    if number_dims != None:
      red_evecs = evecs[:,:number_dims]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(red_evecs.T, data.T).T, dimcount


      



