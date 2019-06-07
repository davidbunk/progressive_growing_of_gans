# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import numpy as np
import tensorflow as tf

import misc
import tfutil

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]


################ PHASE 1

#----------------------------------------------------------------------------
# Phase 1 Generator loss function used in the paper (WGAN + AC-GAN).

def G_wgan_acgan(G, D, opt, training_set, minibatch_size, lod,
    cond_weight = 1.0): # Weight of the conditioning term.

    # lod = [v for v in tf.global_variables() if 'G/lod' in v.name][0]

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, is_training=True)

    with tf.device('/cpu:0'):
        fake_images_out = tf.image.random_crop(fake_images_out, size=(tf.shape(fake_images_out)[0], tf.shape(fake_images_out)[1], tf.shape(fake_images_out)[2] // 2, tf.shape(fake_images_out)[3] // 2))

    # size = tf.cast(1024 // 2**lod, tf.int32)
    # width = tf.cast(size // 2, tf.int32)
    # maxwidth = tf.cast(size - width, tf.int32)
    #
    # xstart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    # ystart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    #
    # fake_images_out = fake_images_out[:, :, xstart_fake:xstart_fake+width, ystart_fake:ystart_fake+width]
    #
    # xstart_fake = tfutil.autosummary('Crops/G_xstart_fake', xstart_fake)
    # ystart_fake = tfutil.autosummary('Crops/G_ystart_fake', ystart_fake)

    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out# + tf.cast((xstart_fake + ystart_fake) * 0, tf.float32)

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
        loss += label_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Phase 1 Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, lod,
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.
    cond_weight     = 1.0):     # Weight of the conditioning terms.

    #lod = [v for v in tf.global_variables() if 'G/lod' in v.name][0]

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    fake_images_out = G.get_output_for(latents, labels, is_training=True)

    with tf.device('/cpu:0'):
        fake_images_out = tf.image.random_crop(fake_images_out, size=(tf.shape(fake_images_out)[0], tf.shape(fake_images_out)[1], tf.shape(fake_images_out)[2] // 2, tf.shape(fake_images_out)[3] // 2))
        reals = tf.image.random_crop(reals, size=(tf.shape(reals)[0], tf.shape(reals)[1], tf.shape(reals)[2] // 2, tf.shape(reals)[3] // 2))

    # size = tf.cast(1024 // 2**lod, tf.int32)
    # width = tf.cast(size // 2, tf.int32)
    # maxwidth = tf.cast(size - width, tf.int32)
    #
    # xstart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    # ystart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    #
    # xstart_real = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    # ystart_real = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    #
    # fake_images_out = fake_images_out[:, :, xstart_fake:xstart_fake+width, ystart_fake:ystart_fake+width]
    # reals = reals[:,:, xstart_real:xstart_real+width, ystart_real:ystart_real+width]
    #
    # xstart_fake = tfutil.autosummary('Crops/D_xstart_fake', xstart_fake)
    # ystart_fake = tfutil.autosummary('Crops/D_ystart_fake', ystart_fake)
    #
    # xstart_real = tfutil.autosummary('Crops/D_xstart_real', xstart_real)
    # ystart_real = tfutil.autosummary('Crops/D_ystart_real', ystart_real)

    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss/fake_scores', fake_scores_out)

    loss = fake_scores_out - real_scores_out# + tf.cast((xstart_fake + ystart_fake) * 0, tf.float32) + tf.cast((xstart_real + ystart_real) * 0, tf.float32)

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    if D.output_shapes[1][1] > 0:
        with tf.name_scope('LabelPenalty'):
            label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
            label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
            label_penalty_reals = tfutil.autosummary('Loss/label_penalty_reals', label_penalty_reals)
            label_penalty_fakes = tfutil.autosummary('Loss/label_penalty_fakes', label_penalty_fakes)
        loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------


################ PHASE 2

#----------------------------------------------------------------------------
# Phase 2 Generator loss function used in the paper (WGAN + AC-GAN).

def G_2_wgan_acgan(G, D, opt, training_set, minibatch_size, G_old, training_set_old, lod,
                 cond_weight = 1.0): # Weight of the conditioning term.

    # lod = [v for v in tf.global_variables() if 'G/lod' in v.name][0]

    latents_old = tf.random_normal([minibatch_size] + G_old.input_shapes[0][1:])
    labels_old = training_set_old.get_random_labels_tf(minibatch_size)

    mask = G_old.get_output_for(latents_old, labels_old, is_training=False)

    # Check this.
    #mask = (mask + 1) * 127

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    labels = training_set.get_random_labels_tf(minibatch_size)
    fake_images_out = G.get_output_for(latents, labels, mask, is_training=True)

    with tf.device('/cpu:0'):
        fake_images_out = tf.image.random_crop(fake_images_out, size=(tf.shape(fake_images_out)[0], tf.shape(fake_images_out)[1], tf.shape(fake_images_out)[2] // 2, tf.shape(fake_images_out)[3] // 2))

    # size = tf.cast(1024 // 2**lod, tf.int32)
    # width = tf.cast(size // 2, tf.int32)
    # maxwidth = tf.cast(size - width, tf.int32)
    #
    # xstart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    # ystart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    #
    # xstart_fake = tfutil.autosummary('Crops/G2_xstart_fake', xstart_fake)
    # ystart_fake = tfutil.autosummary('Crops/G2_ystart_fake', ystart_fake)

    # fake_images_out = fake_images_out[:, :, xstart_fake:xstart_fake+width, ystart_fake:ystart_fake+width]

    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    loss = -fake_scores_out# + tf.cast((xstart_fake + ystart_fake) * 0, tf.float32)

    # if D.output_shapes[1][1] > 0:
    #     with tf.name_scope('LabelPenalty2'):
    #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #     loss += label_penalty_fakes * cond_weight
    return loss

#----------------------------------------------------------------------------
# Phase 2 Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_2_wgangp_acgan(G, D, opt, training_set, minibatch_size, reals, labels, G_old, training_set_old, lod,
                   wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
                   wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
                   wgan_target     = 1.0,      # Target value for gradient magnitudes.
                   cond_weight     = 1.0):     # Weight of the conditioning terms.

    # lod = [v for v in tf.global_variables() if 'G/lod' in v.name][0]

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    latents_old = tf.random_normal([minibatch_size] + G_old.input_shapes[0][1:])
    labels_old = training_set_old.get_random_labels_tf(minibatch_size)
    mask = G_old.get_output_for(latents_old, labels_old, is_training=False)

    # Check this.
    #mask = (mask + 1) * 127

    fake_images_out = G.get_output_for(latents, labels, mask, is_training=True)

    with tf.device('/cpu:0'):
        fake_images_out = tf.image.random_crop(fake_images_out, size=(tf.shape(fake_images_out)[0], tf.shape(fake_images_out)[1], tf.shape(fake_images_out)[2] // 2, tf.shape(fake_images_out)[3] // 2))
        reals = tf.image.random_crop(reals, size=(tf.shape(reals)[0], tf.shape(reals)[1], tf.shape(reals)[2] // 2, tf.shape(reals)[3] // 2))

    # size = tf.cast(1024 // 2**lod, tf.int32)
    # width = tf.cast(size // 2, tf.int32)
    # maxwidth = tf.cast(size - width, tf.int32)
    #
    # xstart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    # ystart_fake = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    #
    # xstart_real = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    # ystart_real = tf.random.uniform([], minval=0, maxval=maxwidth, dtype=tf.dtypes.int32)
    #
    # fake_images_out = fake_images_out[:, :, xstart_fake:xstart_fake+width, ystart_fake:ystart_fake+width]
    # reals = reals[:,:, xstart_real:xstart_real+width, ystart_real:ystart_real+width]
    #
    # xstart_fake = tfutil.autosummary('Crops/D2_xstart_fake', xstart_fake)
    # ystart_fake = tfutil.autosummary('Crops/D2_ystart_fake', ystart_fake)
    #
    # xstart_real = tfutil.autosummary('Crops/D2_xstart_real', xstart_real)
    # ystart_real = tfutil.autosummary('Crops/D2_ystart_real', ystart_real)

    real_scores_out, real_labels_out = fp32(D.get_output_for(reals, is_training=True))
    fake_scores_out, fake_labels_out = fp32(D.get_output_for(fake_images_out, is_training=True))
    real_scores_out = tfutil.autosummary('Loss2/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss2/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out# + tf.cast((xstart_fake + ystart_fake) * 0, tf.float32) + tf.cast((xstart_real + ystart_real) * 0, tf.float32)

    with tf.name_scope('GradientPenalty2'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss2/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss2/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))

    with tf.name_scope('EpsilonPenalty2'):
        epsilon_penalty = tfutil.autosummary('Loss2/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon

    # if D.output_shapes[1][1] > 0:
    #     with tf.name_scope('LabelPenalty2'):
    #         label_penalty_reals = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=real_labels_out)
    #         label_penalty_fakes = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=fake_labels_out)
    #         label_penalty_reals = tfutil.autosummary('Loss2/label_penalty_reals', label_penalty_reals)
    #         label_penalty_fakes = tfutil.autosummary('Loss2/label_penalty_fakes', label_penalty_fakes)
    #     loss += (label_penalty_reals + label_penalty_fakes) * cond_weight
    return loss

#----------------------------------------------------------------------------
