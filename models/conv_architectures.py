import tensorflow as tf

# larger convmodel with ~500'000 params
class GeneralNet(tf.keras.Model):
  def __init__(self, dataproc):
    super().__init__()
    self.dataproc = dataproc
    self.batch_size = dataproc.batch_size
    self.inputshape = dataproc.get_input_shape()
    self.outputshape = dataproc.get_output_shape()
    self.model = tf.keras.Sequential(
      layers=[]
    )
  
  def call(self, inputs, training=None):
    return self.model(inputs, training=training)

  def summary(self):
    return self.model.summary()


class ConvModel1(GeneralNet):
  def __init__(self, dataproc):
    super().__init__(dataproc)
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(16, 5, activation='relu', input_shape=self.inputshape),
            tf.keras.layers.Conv2D(32, 5, activation='relu'),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.AvgPool2D(2),
            tf.keras.layers.Conv2D(32, 1, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class ConvModel2(GeneralNet):
  # large16
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(16, 3, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(128, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class ConvModel3(GeneralNet):
  # medium16
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(8, 3, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class ConvModel4(GeneralNet):
  # small16
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(4, 3, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(8, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class ConvModel5(GeneralNet):
  # tiny16 (not recognized by the UN)
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(2, 3, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(4, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(8, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, 2, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class Conv32Model1(GeneralNet):
  # large32
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(16, 5, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 5, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(128, 3, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class Conv32Model2(GeneralNet):
  # medium32
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(8, 5, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, 5, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class Conv32Model3(GeneralNet):
  # small32
  def __init__(self, dataproc, regularization=0.01):
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(4, 5, activation='relu', input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(8, 5, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(16, 3, activation='relu', kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(32, 3, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )

class ConvModelSym(tf.keras.Model):
  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(16, 5, activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(32, 5, activation='relu'),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.AvgPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(6)
          ]
    )

  def call(self, inputs):
    return self.model(inputs)

  def summary(self):
    return self.model.summary()


class Conv32ModelExxtra(GeneralNet):
  # small32
  def __init__(self, dataproc, regularization=0.01):
    actf ='gelu'
    super().__init__(dataproc)
    reg2 = tf.keras.regularizers.L2
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(32, 5, activation=actf, input_shape=self.inputshape, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(64, 3, activation=actf, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(128, 3, activation=actf, kernel_regularizer=reg2(regularization)),
            # tf.keras.layers.MaxPool2D(2),
            tf.keras.layers.Conv2D(256, 3, activation=actf, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.AvgPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=actf, kernel_regularizer=reg2(regularization)),
            tf.keras.layers.Dense(self.outputshape)
          ]
    )
