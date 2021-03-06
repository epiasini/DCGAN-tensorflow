from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

import vgg16

from ops import *
from utils import *

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=64, sample_size=64, output_size=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, vgg_reg=0,
                 dataset_name='default', sample_dir='samples',
                 checkpoint_dir='checkpoint', log_dir='logs',
                 is_train=True):
        """

        Args:
            sess: TensorFlow session
            image_size: (optional) size for center-cropping before input image scaling. Use None to just make a maximal square crop [108].
            batch_size: (optional) The size of batch. Should be specified before training [64].
            sample_size: (optional) The size of the sample generated each 100 iterations [64].
            output_size: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
            vgg_reg: (optional) VGG regularisation hyperparameter. [0]
            sample_dir: (optional) Directory for saved samples [./samples]
            log_dir: (optional) Directory for logs [./logs]
            checkpoint_dir: (optional) Directory for checkpoints [./checkpoint]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.is_train = is_train
        self.is_grayscale = (c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = c_dim
        
        self.vgg_reg = vgg_reg
        self.loaded = False

        # batch normalization : deals with poor initialization helps gradient flow
        with tf.variable_scope("Discriminator"):
            self.d_bn1 = batch_norm(name='d_bn1')
            self.d_bn2 = batch_norm(name='d_bn2')

            if not self.y_dim:
                self.d_bn3 = batch_norm(name='d_bn3')

        with tf.variable_scope("Generator"):
            self.g_bn0 = batch_norm(name='g_bn0')
            self.g_bn1 = batch_norm(name='g_bn1')
            self.g_bn2 = batch_norm(name='g_bn2')

            if not self.y_dim:
                self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.sample_dir = sample_dir
        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

        self.images = tf.placeholder(tf.float32, [self.batch_size] + [self.output_size, self.output_size, self.c_dim],
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + [self.output_size, self.output_size, self.c_dim],
                                        name='sample_images')
        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            self.D_logits, self.D = self.discriminator(self.images, self.y, reuse=False)

            self.sampler = self.sampler(self.z, self.y)
            self.D_logits_, self.D_ = self.discriminator(self.G, self.y, reuse=True)
        else:
            with tf.variable_scope("Generator"):
                self.G = self.generator(self.z)
                self.sampler = self.sampler(self.z)

            with tf.variable_scope("Discriminator"):
                self.D, self.D_logits, self.D_activation = self.discriminator(self.images)
                self.D_, self.D_logits_, self.D_activation_ = self.discriminator(self.G, reuse=True)


        self.d_sum = tf.histogram_summary("summaries/d", self.D)
        self.d__sum = tf.histogram_summary("summaries/d_", self.D_)
        self.G_sum = tf.image_summary("summaries/G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        if self.vgg_reg>0:
            # feature matching with VGG
            with tf.variable_scope("VGG"):
                self.vgg = vgg16.Vgg16()
                self.vgg_ = vgg16.Vgg16()
                self.vgg.build(self.images)
                self.vgg_.build(self.G, reuse=True)
                self.V = self.vgg.pool4;
                self.V_ = self.vgg_.pool4;
        
            self.v_activation_real = tf.reduce_mean(self.V, reduction_indices=1)
            self.v_activation_fake = tf.reduce_mean(self.V_, reduction_indices=1)
            self.v_loss = tf.truediv(tf.nn.l2_loss(tf.sub(self.V_, self.V)), tf.nn.l2_loss(self.V))
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_))) + self.vgg_reg * self.v_loss
        elif self.vgg_reg==0:
            # vanilla DCGAN
            # self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
            # vanilla DCGAN with alternative G step
            self.g_loss = -tf.reduce_mean(self.D_logits_)
        else:
            # feature matching with the discriminator
            self.g_loss = tf.nn.l2_loss(tf.sub(self.D_activation, self.D_activation_))

        self.d_loss_real_sum = tf.scalar_summary("summaries/d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("summaries/d_loss_fake", self.d_loss_fake)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.scalar_summary("summaries/g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("summaries/d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator/d_' in var.name]
        self.g_vars = [var for var in t_vars if 'Generator/g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""
        if config.dataset == 'mnist':
            data_X, data_y = self.load_mnist()
        elif config.dataset == 'celebA':
            data = glob("./data/img_align_celeba/*.jpg")
        elif config.dataset == 'cats':
            data = glob("./data/cats_vs_dogs/train/cat.*.jpg")
        #np.random.shuffle(data)
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        tf.initialize_all_variables().run()

        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        #self.merged = tf.merge_all_summaries()
        self.writer = tf.train.SummaryWriter(self.log_dir, self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        if config.dataset == 'mnist':
            sample_images = data_X[0:self.sample_size]
            sample_labels = data_y[0:self.sample_size]
        else:
            sample_files = data[0:self.sample_size]
            sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for sample_file in sample_files]
            if (self.is_grayscale):
                sample_images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                sample_images = np.array(sample).astype(np.float32)
            
        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        for epoch in xrange(config.epoch):
            if config.dataset == 'mnist':
                batch_idxs = min(len(data_X), config.train_size) // config.batch_size
            elif config.dataset == 'celebA':
                data = glob("./data/img_align_celeba/*.jpg")
                batch_idxs = min(len(data), config.train_size) // config.batch_size
            elif config.dataset == 'cats':
                data = glob("./data/cats_vs_dogs/train/cat.*.jpg")
                batch_idxs = min(len(data), config.train_size) // config.batch_size
                
            for idx in xrange(0, batch_idxs):
                if config.dataset == 'mnist':
                    batch_images = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
                else:
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop, resize_w=self.output_size, is_grayscale = self.is_grayscale) for batch_file in batch_files]
                    if (self.is_grayscale):
                        batch_images = np.array(batch).astype(np.float32)[:, :, :, None]
                    else:
                        batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                if config.dataset == 'mnist':
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.z: batch_z, self.y:batch_labels })
                    self.writer.add_summary(summary_str, counter)
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.y:batch_labels})
                    errD_real = self.d_loss_real.eval({self.images: batch_images, self.y:batch_labels})
                    errG = self.g_loss.eval({self.z: batch_z, self.y:batch_labels})
                else:
                    # Update D network
                    _,  summary = self.sess.run([d_optim, self.d_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary, counter)

                    # Update G network
                    _, summary = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary = self.sess.run([g_optim, self.g_sum],
                        feed_dict={ self.images: batch_images, self.z: batch_z })
                    self.writer.add_summary(summary, counter)
                    
                    errD_fake = self.d_loss_fake.eval({self.z: batch_z})
                    errD_real = self.d_loss_real.eval({self.images: batch_images})
                    errG = self.g_loss.eval({self.images: batch_images, self.z: batch_z})

                counter += 1
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                    % (epoch, idx, batch_idxs,
                        time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(counter, 100) == 1:
                    if config.dataset == 'mnist':
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images, self.y:batch_labels}
                        )
                    else:
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={self.z: sample_z, self.images: sample_images}
                        )
                    save_images(samples, [8, 8],
                                '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)

    def sample(self, z):
        """Sample the generator with a specific choice of z.
        
        z should be a n_samples x 100 array
        """

        #self.writer = tf.train.SummaryWriter(self.log_dir, self.sess.graph)
        
        #sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))
        
        if not self.loaded:
            if not self.load(self.checkpoint_dir):
                print("Checkpoint file not found!")
            self.loaded = True

        samples = self.sess.run(self.sampler, feed_dict={self.z: z})

        return samples

#        save_images(samples, [8, 8],
#                    '{}/train_{:02d}_{:04d}.png'.format(self.sample_dir, epoch, idx))
#        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

    def discriminator(self, image, y=None, reuse=False, use_minibatch_discrimination=True):
        if reuse:
            tf.get_variable_scope().reuse_variables()


        if not self.y_dim:
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv', attach_summaries=not reuse))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv', attach_summaries=not reuse), train=True))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv', attach_summaries=not reuse), train=True))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv', attach_summaries=not reuse), train=True))
            h3_reshaped = tf.reshape(h3, [self.batch_size, -1], name='d_h3_reshaped');


            if use_minibatch_discrimination:
                # minibatch discrimination (from "improved GAN" paper)
                n_kernels = 50 # B in the paper
                dim_per_kernel = 50 # C in the paper
                x = linear(h3_reshaped, n_kernels * dim_per_kernel)
                activation = tf.reshape(x, (self.batch_size, n_kernels, dim_per_kernel)) # M in the paper
                
                big = np.zeros((self.batch_size, self.batch_size), dtype='float32')
                big += np.eye(self.batch_size)
                big = tf.expand_dims(big, 1)
            
                abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)
                mask = 1. - big
                masked = tf.exp(-abs_dif) * mask # c_b in the paper

                minibatch_features = tf.reduce_sum(masked, 2)

                print("original features: ", h3_reshaped.get_shape())
                print("minibatch_features: ", minibatch_features.get_shape())            
        
                x = tf.concat(1, [h3_reshaped , minibatch_features])
                print("x: ", x.get_shape())

                h4 = linear(x, 1, 'd_h3_lin')

            else:
                h4 = linear(h3_reshaped, 1, 'd_h3_lin')

            if not reuse:
                variable_summaries(h0, 'd_h0_conv/activation')
                variable_summaries(h1, 'd_h1_conv/activation')
                variable_summaries(h2, 'd_h2_conv/activation')
                variable_summaries(h3, 'd_h3_conv/activation')
                variable_summaries(h4, 'd_h3_lin/activation')
            return tf.nn.sigmoid(h4), h4, h3
        else:
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))
            h0 = conv_cond_concat(h0, yb)

            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv'), train=self.is_train))
            h1 = tf.reshape(h1, [self.batch_size, -1])            
            h1 = tf.concat(1, [h1, y])
            
            h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'), train=self.is_train))
            h2 = tf.concat(1, [h2, y])

            h3 = linear(h2, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h3), h3

    def generator(self, z, y=None):
        if not self.y_dim:
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin', with_w=True)

            self.h0 = tf.reshape(self.z_, [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(self.h0, train=True))
            variable_summaries(h0, 'g_h0_lin/activation')

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                                                     [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1', with_w=True, attach_summaries=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1, train=True))
            variable_summaries(h1, 'g_h1/activation')

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                                                [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2', with_w=True, attach_summaries=True)
            h2 = tf.nn.relu(self.g_bn2(h2, train=True))
            variable_summaries(h2, 'g_h2/activation')

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                                                [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3', with_w=True, attach_summaries=True)
            h3 = tf.nn.relu(self.g_bn3(h3, train=True))
            variable_summaries(h3, 'g_h3/activation')

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                                                [self.batch_size, s, s, self.c_dim], name='g_h4', with_w=True, attach_summaries=True)
            variable_summaries(h4, 'g_h4/activation')

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4) 

            # yb = tf.expand_dims(tf.expand_dims(y, 1),2)
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin'), train=self.is_train))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=self.is_train))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])

            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=self.is_train))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            
            s = self.output_size
            s2, s4, s8, s16 = int(s/2), int(s/4), int(s/8), int(s/16)

            # project `z` and reshape

            h0 = tf.reshape(linear(z, self.gf_dim*8*s16*s16, 'g_h0_lin'),
                            [-1, s16, s16, self.gf_dim * 8])
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, s8, s8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, s4, s4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, s2, s2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, s, s, self.c_dim], name='g_h4')

            return tf.nn.tanh(h4)
        else:
            s = self.output_size
            s2, s4 = int(s/2), int(s/4)

            # yb = tf.reshape(y, [-1, 1, 1, self.y_dim])
            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            z = tf.concat(1, [z, y])

            h0 = tf.nn.relu(self.g_bn0(linear(z, self.gfc_dim, 'g_h0_lin')))
            h0 = tf.concat(1, [h0, y])

            h1 = tf.nn.relu(self.g_bn1(linear(z, self.gf_dim*2*s4*s4, 'g_h1_lin'), train=False))
            h1 = tf.reshape(h1, [self.batch_size, s4, s4, self.gf_dim * 2])
            h1 = conv_cond_concat(h1, yb)

            h2 = tf.nn.relu(self.g_bn2(deconv2d(h1, [self.batch_size, s2, s2, self.gf_dim * 2], name='g_h2'), train=False))
            h2 = conv_cond_concat(h2, yb)

            return tf.nn.sigmoid(deconv2d(h2, [self.batch_size, s, s, self.c_dim], name='g_h3'))

    def load_mnist(self):
        data_dir = os.path.join("./data", self.dataset_name)
        
        fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        trY = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

        fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd,dtype=np.uint8)
        teY = loaded[8:].reshape((10000)).astype(np.float)

        trY = np.asarray(trY)
        teY = np.asarray(teY)
        
        X = np.concatenate((trX, teX), axis=0)
        y = np.concatenate((trY, teY), axis=0)
        
        seed = 547
        np.random.seed(seed)
        np.random.shuffle(X)
        np.random.seed(seed)
        np.random.shuffle(y)
        
        y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
        for i, label in enumerate(y):
            y_vec[i,y[i]] = 1.0
        
        return X/255.,y_vec
            
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
