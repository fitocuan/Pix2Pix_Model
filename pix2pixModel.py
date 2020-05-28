import tensorflow as tf
import matplotlib.pyplot as plt 
import numpy as np 
import os
from tensorflow.keras.layers import * 
from tensorflow.keras import * 

PATH = "data_recon"
INPATH = "./Data/STRDATA"
OUTPATH = "./Data/STRDATA_GT"
DPATH = PATH + "./Data/dummy"


n = 10000
train_n = round(n*0.80)

imgurls = [f for f in os.listdir(INPATH)]

randurls = np.copy(imgurls)
np.random.shuffle(randurls)
tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(tr_urls), len(ts_urls))

IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(inimg, tgimg, height, width):
    inimg = tf.image.resize(inimg, [height,width])
    tgimg = tf.image.resize(tgimg, [height,width])
    return inimg, tgimg

def normalize(inimg, tgimg):
    inimg = (inimg / 127.5) - 1
    tgimg = (tgimg / 127.5) - 1
    return inimg, tgimg

def load_image(filename, augment = True):
    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(INPATH + "/" + filename)),tf.float32)[...,:3]
    tgimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(OUTPATH + "/" + filename)),tf.float32)[...,:3]

    inimg, tgimg = resize(inimg,tgimg, IMG_HEIGHT, IMG_WIDTH)
    inimg, tgimg = normalize(inimg, tgimg)

    return inimg, tgimg 

def load_image_dummy(filename):

    inimg = tf.cast(tf.image.decode_jpeg(tf.io.read_file(DPATH + "/" + filename)),tf.float32)[...,:3]
    inimg = tf.image.resize(inimg, [IMG_HEIGHT,IMG_WIDTH])
    inimg = (inimg / 127.5) - 1

    return inimg


def load_train_image(filename):
    return load_image(filename, True)
    
def load_test_image(filename):
    return load_image(filename, False)



train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(load_train_image, num_parallel_calls = 6)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(load_test_image, num_parallel_calls = 6)
test_dataset = test_dataset.batch(1)


def downsample(filters, apply_batchnorm = True):

    result = Sequential()

    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2D(filters, 
        kernel_size = 4, 
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        use_bias = not apply_batchnorm))

    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())

    return result

def upsample(filters, apply_dropout = False):

    result = Sequential()

    initializer = tf.random_normal_initializer(0,0.02)

    result.add(Conv2DTranspose(filters, 
        kernel_size = 4, 
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        use_bias = False))

    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())

    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    downstack = [
        downsample(64, apply_batchnorm = False),
        downsample(128),
        downsample(256),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512)
    ]

    upstack = [
        upsample(512, apply_dropout = True),
        upsample(512, apply_dropout = True),
        upsample(512, apply_dropout = True),
        upsample(512),
        upsample(256),
        upsample(128),
        upsample(64)
    ]

    initializer = tf.random_normal_initializer(0,0.02)

    last = Conv2DTranspose(filters = 3, 
        kernel_size = 4,
        strides = 2, 
        padding = "same", 
        kernel_initializer = initializer,
        activation = "tanh")

    x = inputs
    s = []

    for down in downstack:
        x = down(x)
        s.append(x)

    s = reversed(s[:-1])

    for up, sk in zip(upstack,s):
        x = up(x)
        x = Concatenate()([x, sk])

    x = last(x)

    return Model(inputs = inputs, outputs = x)

def Discriminator():
    ini = Input(shape=[256,256,3], name = "input_img")
    gen = Input(shape=[256,256,3], name = "gener_img")

    con = concatenate([ini,gen])

    initializer = tf.random_normal_initializer(0,0.02)

    down1 = downsample(64, apply_batchnorm = False)(con)
    down2 = downsample(128)(down1)
    down3 = downsample(256)(down2)
    down4 = downsample(512)(down3)
    
    last = Conv2D(filters = 1, 
        kernel_size = 4,
        strides = 1, 
        padding = "same", 
        kernel_initializer = initializer)(down4)

    return tf.keras.Model(inputs = [ini, gen], outputs = last)

loss_object = tf.keras.losses.BinaryCrossentropy(from_logits = True)

def discrimiator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss

LAMBDA = 100

def generator_loss(disc_generated_output, gen_output, target):
    gen_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gen_loss + (LAMBDA*l1_loss)

    return total_gen_loss





def generate_images(model, test_input, tar,filename, save_filename = False, display_imgs = True):
    prediction = model(test_input, training=True)

    if save_filename:
        tf.keras.preprocessing.image.save_img("./output/" + filename + ".jpg", prediction[0,...])

    plt.figure(figsize=(10,10))

    if display_imgs:
        display_list = [test_input[0], tar[0], prediction[0]]
        title = ['Input Image', 'Ground Truth', 'Predicted Image']

        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()


from IPython.display import clear_output

generator = Generator()
discriminator = Discriminator()

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)

checkpoint = tf.train.Checkpoint(
                                step = tf.Variable(1),
                                generator_optimizer = generator_optimizer,
                                discriminator_optimizer = discriminator_optimizer,
                                generator = generator,
                                discriminator = discriminator)

manager = tf.train.CheckpointManager(checkpoint, "./tf_ckpts", max_to_keep = 5)


@tf.function()
def train_step(input_image, target):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as discr_tape:

        output_image = generator(input_image, training = True)

        output_gen_discr = discriminator([output_image, input_image], training = True)

        output_trg_discr = discriminator([target, input_image], training = True)

        discr_loss = discrimiator_loss(output_trg_discr, output_gen_discr)

        gen_loss = generator_loss(output_gen_discr, output_image, target)

        generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
        discriminator_grads = discr_tape.gradient(discr_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(generator_grads, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(discriminator_grads, discriminator.trainable_variables))


def train(dataset, epochs):

    checkpoint.restore(manager.latest_checkpoint)


    if manager.latest_checkpoint:
        print("Restore from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from Scratch")

    for epoch in range(epochs):

        imgi = 0
        for input_image, target in dataset:
            print('epoch ' + str(epoch) + ' - train: ' + str(imgi)+ '/' + str(len(tr_urls)))
            imgi += 1
            train_step(input_image, target)
            clear_output(wait = True)

        
        imgi = 0
        for inp, tar in test_dataset.take(5):
            generate_images(generator, inp, tar, str(imgi) + '_' + str(int(checkpoint.step)),save_filename = True, display_imgs = False)
            imgi += 1

        checkpoint.step.assign_add(1)

        if int(checkpoint.step) % 1 == 0:
            save_path = manager.save()
            print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
            print("Saved checkpoint for step ")



checkpoint.restore(manager.latest_checkpoint)
imgi = 0
for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar, " ",save_filename = False, display_imgs = True)
    imgi += 1

#for item in tf.train.list_variables(tf.train.latest_checkpoint('./tf_ckpts/')):
#    print(item)

#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

#train(train_dataset, 1)
#tf.saved_model.save(generator, PATH + "/models2")


"""

generator = tf.saved_model.load(PATH + "/models2")
inp = load_image_dummy("2.jpg")
inp = tf.expand_dims(inp,0)
generate_images(generator, inp, inp, " ",save_filename = False, display_imgs = True)


generator = tf.saved_model.load(PATH + "/models2")

imgi = 0
for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar, " ",save_filename = False, display_imgs = True)
    imgi += 1
"""
