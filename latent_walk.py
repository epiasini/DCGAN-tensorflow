import argparse

import numpy as np
import tensorflow as tf
import imageio as iio

from model import DCGAN

parser = argparse.ArgumentParser()

parser.add_argument('dataset', default='faces', choices=['faces', 'cats'], help='Picture type')
parser.add_argument('image_size', type=int, default=128, choices=[64, 128], help='Resolution of synthetic pictures (only matters if generating faces)')

def pareto(mu=1., alpha=1.5, size=None):
    #attention, valable seulement pour alpha > 1
    #mu is the mean of the distrib
    #alpha is the stability parameter
    a = (alpha - 1.)*mu/alpha;#    // valeur a partir de laquelle la distribution est non nulle
    xi_=1./np.random.uniform(size=size);
    return a*np.power(xi_, 1./alpha);

def mixture_pareto(mu=1, alpha=1.5, size=None):
    absolute = pareto(mu, alpha, size)
    return absolute*(np.random.randint(2, size=size) - 0.5)*2

def generate_latent_walk(checkpoint_dir, dataset_name, output_size, n_epochs, jump_length, output_file_name):
    n_jumps = n_epochs * 64
    
    boundary = 1.3
    z = np.zeros((n_jumps, 100))
    walk = np.zeros((n_jumps, output_size, output_size, 3))
    for t in range(1,n_jumps):
        jumps = mixture_pareto(mu=0.04, alpha=1.6, size=[1,100])
        # keep walk confined within hypercube with reflective boundaries
        new_z = z[t-1,:] + jumps
        outside_domain_left = new_z<-boundary
        outside_domain_right = new_z>boundary
        new_z_2 = new_z[:]
        new_z_2[outside_domain_left] = - boundary - (new_z[outside_domain_left] + boundary)
        new_z_2[outside_domain_right] = boundary - (new_z[outside_domain_right] - boundary)
        new_z[outside_domain_left] = new_z_2[outside_domain_left]
        new_z[outside_domain_right] = new_z_2[outside_domain_right]
        z[t,:] = new_z

    with tf.Session() as sess:
        dcgan = DCGAN(sess, checkpoint_dir=checkpoint_dir, dataset_name=dataset_name, output_size=output_size, is_train=False)
        for e in range(n_epochs):
            this_start = e*64
            this_end = (e+1)*64
            walk[this_start:this_end,:,:,:] = dcgan.sample(z[this_start:this_end,:])
    save_latent_walk(walk, output_file_name)
    return walk

def save_latent_walk(w, filename='test.gif'):
    iio.mimwrite(filename, w)


if __name__ == '__main__':
    args = parser.parse_args()

    if args.dataset == 'faces':
        if args.image_size==64:
            name = 'faces_batch_d_64'
        elif args.image_size==128:
            name = 'faces_batch_d_gpu'
        generate_latent_walk(checkpoint_dir='checkpoint/{}'.format(name),
                             dataset_name='celebA',
                             output_size=args.image_size,
                             n_epochs=10,
                             jump_length=0.05,
                             output_file_name='test_faces.gif')
    elif args.dataset == 'cats':
        generate_latent_walk(checkpoint_dir='checkpoint/cats_batch_d_gpu',
                             dataset_name='cats',
                             output_size=64,
                             n_epochs=10,
                             jump_length=0.02,
                             output_file_name='test_cats.gif')
    
