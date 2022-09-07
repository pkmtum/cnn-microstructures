import numpy as np

# if as_vector is True, mat_shape reshape is applied
# else no reshape is done
def test_psd_eigenval(matrices, as_vector=True, mat_shape=(3,3), tol=1e-12):
  
  matrices = np.array(matrices)
  if as_vector:
    arr_shape = matrices.shape
    if len(arr_shape) == 1:
      matrices = matrices.reshape(mat_shape)
    if len(arr_shape) == 2:
      matrices = matrices.reshape((arr_shape[0], mat_shape[0], mat_shape[1]))

  eigenval,_ = np.linalg.eig(matrices)
  print(eigenval)
  bool_num = np.all(eigenval > -tol, axis=1)
  bool_zero = np.all(eigenval > 0, axis=1)
  not_psd_count_num = np.size(bool_num) - np.count_nonzero(bool_num)
  not_psd_count_zero = np.size(bool_zero) - np.count_nonzero(bool_zero)

  # not_psd_count_zero tests strictly pd
  return not_psd_count_num, not_psd_count_zero

def test_psd_cholesky(matrices, as_vector=True, mat_shape=(3,3)):
  matrices = np.array(matrices)
  if as_vector:
    arr_shape = matrices.shape
    if len(arr_shape) == 1:
      matrices = matrices.reshape(mat_shape)
    if len(arr_shape) == 2:
      matrices = matrices.reshape((arr_shape[0], mat_shape[0], mat_shape[1]))

  num_not_psd = 0
  for mat in matrices:
    try:
      np.linalg.cholesky(mat)
    except np.linalg.LinAlgError:
      num_not_psd = num_not_psd + 1

  return num_not_psd


def is_symmetric(matrices, tol=0):

  transposed = np.transpose(matrices, axes=(0,2,1))
  diff = np.absolute(matrices - transposed)
  result = int(np.count_nonzero(diff > tol) / 2)

  return (result == 0), result
