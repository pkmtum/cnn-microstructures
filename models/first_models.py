# FCN for 16x16 and 32x32 samples

import tensorflow as tf

# for 16x16
class SimpleModel(tf.keras.Model):

  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.layer1 = tf.keras.layers.Dense(256, activation=tf.nn.relu, input_shape=(None, batch_size, 256))
    self.layer2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    self.layer3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    self.layer4 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
    self.layer5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.layer6 = tf.keras.layers.Dense(9)

  def call(self, inputs):
    x = inputs
    x = self.layer1(x)
    # x = self.layer2(x)
    # x = self.layer3(x)
    # x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    return x

# for 32x32
class SimpleModel32(tf.keras.Model):

  def __init__(self, batch_size):
    super().__init__()
    self.batch_size = batch_size
    self.layer1 = tf.keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(None, batch_size, 1024))
    self.layer2 = tf.keras.layers.Dense(512, activation=tf.nn.relu)
    self.layer3 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
    self.layer4 = tf.keras.layers.Dense(128, activation=tf.nn.relu)
    self.layer5 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    self.layer6 = tf.keras.layers.Dense(9)

  def call(self, inputs):
    x = inputs
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    x = self.layer5(x)
    x = self.layer6(x)
    return x

def get_simple(in_shape, out_shape, REG=0.001):
  reg2 = tf.keras.regularizers.L2
  return tf.keras.Sequential(
    layers=[
      tf.keras.layers.Dense(64, activation='relu', input_shape=in_shape, kernel_regularizer=reg2(REG)),
      tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=reg2(REG)),
      tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=reg2(REG)),
      tf.keras.layers.Dense(out_shape),
  ]
    )