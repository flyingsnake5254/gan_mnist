import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import os
import imageio
import glob


os.makedirs('generated_images', exist_ok=True)


(x_train, _), (_, _) = mnist.load_data()
x_train = x_train.reshape(-1, 784).astype("float32") / 255 * 2 - 1  # 将图片数据缩放到 -1 到 1 的范围内

noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])


def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    predictions = tf.reshape(predictions, [predictions.shape[0], 28, 28])

    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('generated_images/image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_dim=100, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1024, activation="relu"))
    model.add(layers.Dense(784, activation="tanh"))
    return model


def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(1024, input_dim=784, activation="relu"))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    return model

optimizer = tf.keras.optimizers.Adam(1e-4)


loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)


@tf.function
def train_step(real_images):
    noise = tf.random.normal([real_images.shape[0], 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = loss_fn(tf.ones_like(fake_output), fake_output)
        disc_loss = loss_fn(tf.ones_like(real_output), real_output) + loss_fn(tf.zeros_like(fake_output), fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))


generator = build_generator()
discriminator = build_discriminator()

for epoch in range(200):
    print("Epoch ", epoch)
    for i in range(x_train.shape[0] // 128):
        train_step(x_train[i * 128: (i + 1) * 128])
    generate_and_save_images(generator, epoch, seed)


with imageio.get_writer('generated_images/gan.gif', mode='I') as writer:
    filenames = glob.glob('generated_images/image*.png')
    filenames = sorted(filenames)
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)
