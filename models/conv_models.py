# FCN for 16x16 and 32x32 samples

import tensorflow as tf

# for 32x32
class ConvModelExp(tf.keras.Model):

  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.layer1 = tf.keras.layers.Conv2D(4, 5, activation='relu', input_shape=(batch_size, 32, 32, 1))
    self.layer2 = tf.keras.layers.Conv2D(8, 5, activation='relu')
    self.layer3 = tf.keras.layers.Conv2D(16, 5, activation='relu')
    self.layer4 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.layer5 = tf.keras.layers.Conv2D(64, 3, activation='relu')
    self.layer6 = tf.keras.layers.Conv2D(16, 1, activation='relu')
    self.layer7 = tf.keras.layers.Flatten()
    self.layer8 = tf.keras.layers.Dense(9)

  def call(self, inputs):
    x = inputs
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    return x


class ConvModelMP(tf.keras.Model):
  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(8, 5, activation='relu', input_shape=(32, 32, 1)),
            tf.keras.layers.Conv2D(16, 5, activation='relu'),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPool2D(2, strides=2),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.Conv2D(32, 1, activation='relu'),
            tf.keras.layers.AvgPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(9)
          ]
    )

  # Optim loss after 100 epochs
  # Adadelta: Better at later point? Consistent decrease
  # Adam: Decrease to 2e-4 but with repeating spikes
  # RMSprop: Not suitable in first 100 epochs
  # SGD: Similar to Adam but worse performance

  # BEST ACC: First Adam until plateau, then Adamax
  
  def call(self, inputs):
    return self.model(inputs)

  def summary(self):
    return self.model.summary()

# larger convmodel with ~500'000 params
class ConvModelLarge(tf.keras.Model):
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
            tf.keras.layers.Dense(9)
          ]
    )

  # Optim loss after 100 epochs
  # Adadelta: Better at later point? Consistent decrease
  # Adam: Decrease to 2e-4 but with repeating spikes
  # RMSprop: Not suitable in first 100 epochs
  # SGD: Similar to Adam but worse performance

  # BEST ACC: First Adam until plateau, then Adamax
  
  def call(self, inputs):
    return self.model(inputs)

  def summary(self):
    return self.model.summary()


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

# larger convmodel for 16x16
class ConvModelLarge16(tf.keras.Model):
  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.model = tf.keras.Sequential(
      layers=[
            tf.keras.layers.Conv2D(8, 5, activation='relu'),
            tf.keras.layers.Conv2D(16, 5, activation='relu', input_shape=(16, 16, 1)),
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.AvgPool2D(2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(9)
          ]
    )

  # Optim loss after 100 epochs
  # Adadelta: Better at later point? Consistent decrease
  # Adam: Decrease to 2e-4 but with repeating spikes
  # RMSprop: Not suitable in first 100 epochs
  # SGD: Similar to Adam but worse performance

  # BEST ACC: First Adam until plateau, then Adamax
  
  def call(self, inputs):
    return self.model(inputs)

  def summary(self):
    return self.model.summary()

# for 32x32
class ConvModelAP(tf.keras.Model):

  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.layer1 = tf.keras.layers.Conv2D(4, 5, activation='relu', input_shape=(batch_size, 32, 32, 1))
    self.layer2 = tf.keras.layers.Conv2D(8, 5, activation='relu')
    self.layer3 = tf.keras.layers.Conv2D(16, 5, activation='relu')
    self.layer4 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.layer5 = tf.keras.layers.Conv2D(64, 3, activation='relu')
    self.layer6 = tf.keras.layers.Conv2D(16, 1, activation='relu')
    self.layer7 = tf.keras.layers.Flatten()
    self.layer8 = tf.keras.layers.Dense(9)

  def call(self, inputs):
    x = inputs
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    x = self.layer7(x)
    x = self.layer8(x)
    return x