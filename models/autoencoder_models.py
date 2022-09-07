# FCN for 16x16 and 32x32 samples

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

STRIDES = 2
PADDING = 'same'

def create_enc(inputshape=(16,16,1)):
  encoder = tf.keras.Sequential(
    layers=[
      tf.keras.layers.Conv2D(16, 3, activation='relu', padding=PADDING, strides=STRIDES, input_shape=inputshape),
      tf.keras.layers.Conv2D(8, 3, activation='relu', padding=PADDING, strides=STRIDES),
      tf.keras.layers.Conv2D(8, 3, activation='relu', padding=PADDING, strides=STRIDES),
    ]
  )
  encoder.build()
  return encoder

def create_dec(inputshape=(2,2,8)):
  decoder = tf.keras.Sequential(
    layers=[
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=STRIDES, activation='relu', padding=PADDING, input_shape=inputshape),
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=STRIDES, activation='relu', padding=PADDING),
      tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=STRIDES, activation='relu', padding=PADDING),
      tf.keras.layers.Conv2D(1, kernel_size=1, padding=PADDING)
    ]
  )
  decoder.build()
  return decoder

############################################
# special autoenc
def create_enc_spec(inputshape=(16,16,1)):
  encoder = tf.keras.Sequential(
    layers=[
      tf.keras.layers.Conv2D(16, 5, activation='relu', padding=PADDING, strides=STRIDES, input_shape=inputshape),
      tf.keras.layers.Conv2D(8, 3, activation='relu', padding=PADDING, strides=STRIDES),
      tf.keras.layers.Conv2D(4, 3, activation='relu', padding=PADDING, strides=STRIDES),
    ]
  )
  encoder.build()
  return encoder

def create_dec_spec(inputshape=(2,2,8)):
  decoder = tf.keras.Sequential(
    layers=[
      tf.keras.layers.Conv2DTranspose(4, kernel_size=3, strides=STRIDES, activation='relu', padding=PADDING, input_shape=inputshape),
      tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=STRIDES, activation='relu', padding=PADDING),
      tf.keras.layers.Conv2DTranspose(16, kernel_size=5, strides=STRIDES, activation='relu', padding=PADDING),
      tf.keras.layers.Conv2D(1, kernel_size=1, padding=PADDING)
    ]
  )
  decoder.build()
  return decoder

#####################################

class ConvAuto(tf.keras.Model):
  def __init__(self, encoder, decoder):
    super().__init__()
    self.encoder = encoder
    self.decoder = decoder
    return

  def call(self, x):
    print(x.shape)
    out = self.encoder(x)
    return self.decoder(out)
  
  def summary(self):
    print(self.encoder.summary())
    print(self.decoder.summary())
    return


class ConvEncoder1(GeneralNet):
  def __init__(self, dataproc):
    super().__init__(dataproc)
    self.model = tf.keras.Sequential(
      layers=[
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2, input_shape=self.inputshape),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)
      ]
    )


  def call(self, x):
    return self.model(x)

  def summary(self):
    return self.model.summary()


class ConvDecoder1(GeneralNet):
  def __init__(self, dataproc):
    super().__init__(dataproc)
    # Does not have an input_shape ==> has not been built!
    self.model = tf.keras.Sequential(
      layers=[
        tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same', input_shape=(8,8,8)),
        tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(1, kernel_size=(3, 3), padding='same')
      ]
    )

  def call(self, x):
    return self.model(x)

  def summary(self):
    return self.model.summary()

