
from __future__ import division   # accurate division

import time
import tensorflow as tf
import copy
import numpy as np
import os
from ops import *
from utils import *
from utils import pp, generate_data, show_all_variables
 

def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class TableGan(object):
    def __init__(self, sess, input_height=108, input_width=108, crop=True,
                 batch_size=64, sample_num=64, output_height=64, output_width=64,
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, dataset_name='default',
                 checkpoint_dir=None, sample_dir=None,
                 alpha=1.0, beta=1.0, delta_mean=0.0, delta_var=0.0
                 , label_col=-1, attrib_num=0
                 , is_shadow_gan=False
                 , test_id=''
                 ):
        """
        :param sess: TensorFlow session
        :param batch_size:  The size of batch. Should be specified before training.
        :param sample_num:
        :param output_height:
        :param output_width:
        :param y_dim: (optional) Dimension of dim for y. [None]
        :param z_dim: (optional) Dimension of dim for Z. [100]
        :param gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
        :param df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
        :param gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
        :param dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
        :param c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        :param dataset_name: Name of dataset, Required.
        """

        self.test_id = test_id

        self.sess = sess
        self.crop = crop

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.feature_size = 0
        self.attrib_num = 1

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')        
        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')
            
        self.d1_bn1 = batch_norm(name='d1_bn1')
        self.d1_bn2 = batch_norm(name='d1_bn2')       
        if not self.y_dim:
            self.d1_bn3 = batch_norm(name='d1_bn3')
            
        self.d2_bn1 = batch_norm(name='d2_bn1')
        self.d2_bn2 = batch_norm(name='d2_bn2')       
        if not self.y_dim:
            self.d2_bn3 = batch_norm(name='d2_bn3')
            
        self.d3_bn1 = batch_norm(name='d3_bn1')
        self.d3_bn2 = batch_norm(name='d3_bn2')        
        if not self.y_dim:
            self.d3_bn3 = batch_norm(name='d3_bn3')
            
        self.d4_bn1 = batch_norm(name='d4_bn1')
        self.d4_bn2 = batch_norm(name='d4_bn2')       
        if not self.y_dim:
            self.d4_bn3 = batch_norm(name='d4_bn3')
            
        self.d5_bn1 = batch_norm(name='d5_bn1')
        self.d5_bn2 = batch_norm(name='d5_bn2')       
        if not self.y_dim:
            self.d5_bn3 = batch_norm(name='d5_bn3')
            
        # Classifier
        self.c_bn1 = batch_norm(name='c_bn1')
        self.c_bn2 = batch_norm(name='c_bn2')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        self.g_bn3 = batch_norm(name='g_bn3')

        self.alpha = alpha  # Info Loss Weigh
        self.beta = beta  # Class Loss Weigh

        self.delta_mean = delta_mean
        self.delta_var = delta_var

        self.label_col = label_col
        self.attrib_num = attrib_num

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        # mm if self.dataset_name in ["LACity", "Health", "Adult", "Ticket"]:

        self.data_X, self.data_y, self.data_y_normal = self.load_dataset(is_shadow_gan)
        self.c_dim = 1

        self.grayscale = (self.c_dim == 1)
        print("c_dim 1= " + str(self.c_dim))
        ########-build model-########
        self.build_model()

    def build_model(self):

        self.y = tf.placeholder(
            tf.float32, [self.batch_size, self.y_dim], name='y')

        self.y_normal = tf.placeholder(
            tf.int16, [self.batch_size, 1], name='y_normal')            

        # if self.crop:
        #     image_dims = [self.output_height, self.output_width, self.c_dim]
        # else:
        #     image_dims = [self.input_height, self.input_width, self.c_dim]

        data_dims = [self.input_height, self.input_width, self.c_dim]
        self.inputs = tf.placeholder(
            tf.float32, [self.batch_size] + data_dims, name='inputs')

        self.sample_inputs = tf.placeholder(
            tf.float32, [self.sample_num] + data_dims, name='sample_inputs')

        inputs = self.inputs
        inputs1 = inputs
        inputs2 = inputs
        inputs3 = inputs
        inputs4 = inputs       
        inputs5 = inputs   
        
        self.z = tf.placeholder(
            tf.float32, [None, self.z_dim], name='z')

        self.z_sum = histogram_summary("z", self.z)

        if self.y_dim:
            self.G = self.generator(self.z, self.y)
            # real loss 
            # classifier in PATE denoted by d2
            self.D, self.D_logits, self.D_features = self.discriminator(inputs, self.y, reuse=False) 
            # discriminator 1
            self.D1, self.D1_logits, self.D1_features = self.discriminator1(inputs1, self.y, reuse=False)
            # discriminator 2
            self.D2, self.D2_logits, self.D2_features = self.discriminator2(inputs2, self.y, reuse=False)
            # discriminator 3
            self.D3, self.D3_logits, self.D3_features = self.discriminator3(inputs3, self.y, reuse=False)
            # discriminator 4
            self.D4, self.D4_logits, self.D4_features = self.discriminator4(inputs4, self.y, reuse=False)
            # discriminator 5
            self.D5, self.D5_logits, self.D5_features = self.discriminator5(inputs5, self.y, reuse=False)

            
            self.sampler = self.sampler(self.z, self.y)
            self.sampler_disc = self.sampler_discriminator(self.inputs, self.y)
            # fake loss
            self.D_, self.D_logits_, self.D_features_ = self.discriminator(self.G, self.y, reuse=True)
            self.D1_, self.D1_logits_, self.D1_features_ = self.discriminator1(self.G, self.y, reuse=True)
            self.D2_, self.D2_logits_, self.D2_features_ = self.discriminator2(self.G, self.y, reuse=True)
            self.D3_, self.D3_logits_, self.D3_features_ = self.discriminator3(self.G, self.y, reuse=True)
            self.D4_, self.D4_logits_, self.D4_features_ = self.discriminator4(self.G, self.y, reuse=True)
            self.D5_, self.D5_logits_, self.D5_features_ = self.discriminator5(self.G, self.y, reuse=True)
            # Classifier
            if self.label_col > 0:  # We have duplicate attribute in input matrix and the label column should be masked
                inputs_C = masking(inputs, self.label_col, self.attrib_num)
            else:
                inputs_C = inputs

            self.C, self.C_logits, self.C_features = self.classification(inputs_C, self.y, reuse=False)

            if self.label_col > 0:  # We have duplicate attribute in input matrix and the label column should be masked
                self.GC = self.G
            else:
                self.GC = masking(self.G, self.label_col, self.attrib_num)

            self.C_, self.C_logits_, self.C_features = self.classification(self.GC, self.y, reuse=True)

        else:
            self.G = self.generator(self.z)
            self.D, self.D_logits, self.D_features = self.discriminator(inputs)
#            self.D1, self.D1_logits, self.D1_features = self.discriminator1(inputs)
            self.D2, self.D2_logits, self.D2_features = self.discriminator2(inputs)
            self.sampler = self.sampler(self.z)
            self.sampler_disc = self.sampler_discriminator(self.inputs)
            self.D_, self.D_logits_, self.D_features_ = self.discriminator(self.G, reuse=True)
#            self.D1_, self.D1_logits_, self.D1_features_ = self.discriminator1(self.G, reuse=True)
            self.D2_, self.D2_logits_, self.D2_features_ = self.discriminator2(self.G, reuse=True)

        self.d_sum = histogram_summary("d", self.D)
        self.d1_sum = histogram_summary("d1", self.D1)
        self.d2_sum = histogram_summary("d2", self.D2)
        self.d3_sum = histogram_summary("d3", self.D)
        self.d4_sum = histogram_summary("d4", self.D1)
        self.d5_sum = histogram_summary("d5", self.D2)
        
        self.d__sum = histogram_summary("d_", self.D_)
        self.d1__sum = histogram_summary("d1_", self.D1_)
        self.d2__sum = histogram_summary("d2_", self.D2_)
        self.d3__sum = histogram_summary("d3_", self.D_)
        self.d4__sum = histogram_summary("d4_", self.D1_)
        self.d5__sum = histogram_summary("d5_", self.D2_)
        # Classifier
        if self.y_dim:
            self.c_sum = histogram_summary("c", self.C)
            self.c__sum = histogram_summary("c_", self.C_)
        #
        self.G_sum = image_summary("G", self.G)

        def sigmoid_cross_entropy_with_logits(x, y):
            try:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)
            except:
                return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, targets=y)

        y_normal = tf.to_float(self.y_normal)
        
        # reduce_mean 延指定维度降维后的平均值
        
        self.noise = tf.placeholder(
                tf.float32, None, name='noise')
        # score of real (sigmoid_ce) 
        self.d_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d1_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D1_logits, tf.ones_like(self.D1)))
        self.d2_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D2_logits, tf.ones_like(self.D2)))
        self.d3_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D3_logits, tf.ones_like(self.D3)))
        self.d4_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D4_logits, tf.ones_like(self.D4)))
        self.d5_loss_real = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D5_logits, tf.ones_like(self.D5)))        
        # score of fake (sigmoid_ce) self.D_logits_(change on this)
        #self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        # print('label:', tf.zeros_like(self.D_))
        self.d1_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D1_logits_, tf.zeros_like(self.D1_)))
        self.d2_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.zeros_like(self.D2_)))
        self.d3_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D3_logits_, tf.zeros_like(self.D3_)))
        self.d4_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D4_logits_, tf.zeros_like(self.D4_)))
        self.d5_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D5_logits_, tf.zeros_like(self.D5_)))
        
        #score_D_fake=self.D_logits_
        score_D1_fake=self.D1_logits_
        score_D2_fake=self.D2_logits_
        score_D3_fake=self.D3_logits_
        score_D4_fake=self.D4_logits_
        score_D5_fake=self.D5_logits_                       
        
#        values = self.sess.run(score_D_fake.name)
        one = tf.ones_like(score_D1_fake)
        zero = tf.zeros_like(score_D1_fake)
    
        
        score_D1_fake = tf.where(score_D1_fake <0.5, x=zero, y=one)
        score_D2_fake = tf.where(score_D2_fake <0.5, x=zero, y=one)
        score_D3_fake = tf.where(score_D3_fake <0.5, x=zero, y=one)
        score_D4_fake = tf.where(score_D4_fake <0.5, x=zero, y=one)
        score_D5_fake = tf.where(score_D5_fake <0.5, x=zero, y=one)
        
        sum_score_D_fake=tf.add(score_D1_fake,score_D2_fake)
        sum_score_D_fake=tf.add(sum_score_D_fake,score_D3_fake)    
        sum_score_D_fake=tf.add(sum_score_D_fake,score_D4_fake)  
        sum_score_D_fake=tf.add(sum_score_D_fake,score_D5_fake)  
        
        noise = np.random.normal(0,1, [self.batch_size,1])
        noise = noise.astype(np.float32)
        noise = tf.convert_to_tensor(noise)
        #noise = tf.dtypes.cast(noise, tf.float32)
        #sum_score_D_fake=tf.add(score_D_fake,noise)
        sum_score_D_fake=tf.add(sum_score_D_fake,noise)
        
        sum_score_D_fake_label=tf.where(sum_score_D_fake <2.5, x=zero, y=one)
        # print('label:', self.y_dim, sum_score_D_fake_label, self.y, y_normal)
        self.d_loss_fake = tf.reduce_mean(sigmoid_cross_entropy_with_logits(self.D_logits_, sum_score_D_fake_label))
        
        self.d_loss_fake = self.d_loss_fake + self.noise
        self.d1_loss_fake = self.d1_loss_fake + self.noise
        self.d2_loss_fake = self.d2_loss_fake + self.noise
        self.d3_loss_fake = self.d3_loss_fake + self.noise        
        self.d4_loss_fake = self.d4_loss_fake + self.noise
        self.d5_loss_fake = self.d5_loss_fake + self.noise 
       
        #self.d2_loss_fake = self.d2_loss_fake + self.noise
        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d1_loss_real_sum = scalar_summary("d1_loss_real", self.d1_loss_real)
        self.d2_loss_real_sum = scalar_summary("d2_loss_real", self.d2_loss_real)
        self.d3_loss_real_sum = scalar_summary("d3_loss_real", self.d3_loss_real)
        self.d4_loss_real_sum = scalar_summary("d4_loss_real", self.d4_loss_real)
        self.d5_loss_real_sum = scalar_summary("d5_loss_real", self.d5_loss_real)
        
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d1_loss_fake_sum = scalar_summary("d1_loss_fake", self.d1_loss_fake)
        self.d2_loss_fake_sum = scalar_summary("d2_loss_fake", self.d2_loss_fake)
        self.d3_loss_fake_sum = scalar_summary("d3_loss_fake", self.d3_loss_fake)
        self.d4_loss_fake_sum = scalar_summary("d4_loss_fake", self.d4_loss_fake)
        self.d5_loss_fake_sum = scalar_summary("d5_loss_fake", self.d5_loss_fake)
        
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d1_loss = self.d1_loss_real + self.d1_loss_fake
        self.d2_loss = self.d2_loss_real + self.d2_loss_fake
        self.d3_loss = self.d3_loss_real + self.d3_loss_fake
        self.d4_loss = self.d4_loss_real + self.d4_loss_fake
        self.d5_loss = self.d5_loss_real + self.d5_loss_fake

        # Classifier :Loss Funciton
        if self.y_dim:
            self.c_loss = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.C_logits, y_normal))
            self.g_loss_c = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.C_logits_, y_normal))

        # Original Loss Function PATE change on the self.D_logits_
        self.g_loss_o = tf.reduce_mean(
            sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        # PATE : the loss of generator should be calculated by d2 and true label
#         self.g_loss_o = tf.reduce_mean(
#            sigmoid_cross_entropy_with_logits(self.D2_logits_, tf.ones_like(self.D2_)))
        # Loss function for Information Loss
        self.D_features_mean = tf.reduce_mean(self.D_features, axis=0, keep_dims=True)
        self.D_features_mean_ = tf.reduce_mean(self.D_features_, axis=0, keep_dims=True)

        self.D_features_var = tf.reduce_mean(tf.square(self.D_features - self.D_features_mean), axis=0, keep_dims=True)

        self.D_features_var_ = tf.reduce_mean(tf.square(self.D_features_ - self.D_features_mean_), axis=0,
                                              keep_dims=True)

        dim = self.D_features_mean.get_shape()[-1]

        self.feature_size = dim

        print("Feature Size = %s" % (self.D_features_mean.get_shape()[-1]))

        # Previous Global Mean for real Data
        self.prev_gmean = tf.placeholder(tf.float32, [1, dim], name='prev_gmean')

        # Previous Global Mean  for fake Data
        self.prev_gmean_ = tf.placeholder(tf.float32, [1, dim], name='prev_gmean_')

        # Previous Global Variance for real Data
        self.prev_gvar = tf.placeholder(tf.float32, [1, dim], name='prev_gvar')

        # Previous Global Variance for fake Data
        self.prev_gvar_ = tf.placeholder(tf.float32, [1, dim], name='prev_gvar_')

        # Moving Average Contributions
        mac = 0.99

        self.gmean = mac * self.prev_gmean + (1 - mac) * self.D_features_mean

        self.gmean_ = mac * self.prev_gmean_ + (1 - mac) * self.D_features_mean_

        self.gvar = mac * self.prev_gvar + (1 - mac) * self.D_features_var

        self.gvar_ = mac * self.prev_gvar_ + (1 - mac) * self.D_features_var_

        self.info_loss = tf.add(tf.maximum(x=0.0, y=tf.reduce_sum(tf.abs(self.gmean - self.gmean_) - self.delta_mean))
                                , tf.maximum(x=0.0, y=tf.reduce_sum(tf.abs(self.gvar - self.gvar_) - self.delta_var)))

        ## Note: not sure if this can go or what it was used for.
        # Prefix Origin
        # self.g_loss =  self.g_loss_o

        # OI Prefix in test_IDs
        self.g_loss = self.alpha * (self.g_loss_o) + self.beta * self.info_loss

        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)

        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        self.d1_loss_sum = scalar_summary("d1_loss", self.d1_loss)
        self.d2_loss_sum = scalar_summary("d2_loss", self.d2_loss)
        self.d3_loss_sum = scalar_summary("d3_loss", self.d3_loss)
        self.d4_loss_sum = scalar_summary("d4_loss", self.d4_loss)
        self.d5_loss_sum = scalar_summary("d5_loss", self.d5_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.d1_vars = [var for var in t_vars if 'd1_' in var.name]
        self.d2_vars = [var for var in t_vars if 'd2_' in var.name]
        self.d3_vars = [var for var in t_vars if 'd3_' in var.name]
        self.d4_vars = [var for var in t_vars if 'd4_' in var.name]
        self.d5_vars = [var for var in t_vars if 'd5_' in var.name]        
        
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # Classifier: COI Prefix in test_IDs
        if self.y_dim:
            self.g_loss = self.alpha * (0.5 * self.g_loss_c + self.g_loss_o) + self.beta * self.info_loss
            self.c_loss_sum = scalar_summary("c_loss", self.c_loss)
            self.c_vars = [var for var in t_vars if 'c_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        print("Start Training...\n") 

        if not os.path.exists('./samples/{}'.format(config.dataset)):    
            os.mkdir('./samples/{}'.format(config.dataset))
        
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)
        d1_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d1_loss, var_list=self.d1_vars)
        d2_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d2_loss, var_list=self.d2_vars)
        d3_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d3_loss, var_list=self.d3_vars)
        d4_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d4_loss, var_list=self.d4_vars)
        d5_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.d5_loss, var_list=self.d5_vars)

        # Classifier
        if self.y_dim:
            c_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                .minimize(self.c_loss, var_list=self.c_vars)

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = merge_summary([self.z_sum, self.d__sum,
                                    self.G_sum, self.g_loss_sum])
   
        self.d_sum = merge_summary(
            [self.z_sum, self.d_sum, self.d_loss_sum])
        self.d1_sum = merge_summary(
            [self.z_sum, self.d1_sum, self.d1_loss_sum])
        self.d2_sum = merge_summary(
            [self.z_sum, self.d2_sum, self.d2_loss_sum])   
        self.d3_sum = merge_summary(
            [self.z_sum, self.d3_sum, self.d3_loss_sum])
        self.d4_sum = merge_summary(
            [self.z_sum, self.d4_sum, self.d4_loss_sum])
        self.d5_sum = merge_summary(
            [self.z_sum, self.d5_sum, self.d5_loss_sum])

        # Classifier
        if self.y_dim:
            self.c_sum = merge_summary([self.z_sum, self.c_sum, self.c_loss_sum])

        self.writer = SummaryWriter("./logs", self.sess.graph)

        sample_z = np.random.uniform(-1, 1, size=(self.sample_num, self.z_dim))

        sample = self.data_X[0:self.sample_num]

        if self.y_dim:
            sample_labels = self.data_y[0:self.sample_num]
            sample_labels_normal = self.data_y_normal[0:self.sample_num]

        if (self.grayscale):
            sample_inputs = np.array(sample).astype(
                np.float32)[:, :, :, None]
        else:
            sample_inputs = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        feature_size = self.feature_size

        gmean = np.zeros((1, feature_size), dtype=np.float32)
        gmean_ = np.zeros((1, feature_size), dtype=np.float32)
        gvar = np.zeros((1, feature_size), dtype=np.float32)
        gvar_ = np.zeros((1, feature_size), dtype=np.float32)
        d_loss_list, g_loss_list = [], []
        d1_loss_list, d2_loss_list = [], []
        for epoch in xrange(config.epoch):

            batch_idxs = min(len(self.data_X),
                             config.train_size) // config.batch_size  # train_size= np.inf

            seed = np.random.randint(100000000)
            np.random.seed(seed)
            np.random.shuffle(self.data_X)

            if self.y_dim:
                np.random.seed(seed)
                np.random.shuffle(self.data_y)

                np.random.seed(seed)
                np.random.shuffle(self.data_y_normal)

            for idx in xrange(0, batch_idxs - 1):

                batch = self.data_X[idx * config.batch_size:(idx + 1) * config.batch_size]

                if self.y_dim:
                    batch_labels = self.data_y[
                                   idx * config.batch_size: (idx + 1) * config.batch_size]

                    batch_labels_normal = self.data_y_normal[
                                          idx * config.batch_size: (idx + 1) * config.batch_size]

                if self.grayscale:
                    batch_images = np.array(batch).astype(
                        np.float32)[:, :, :, None]
                else:
                    batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                    .astype(np.float32)
                    
                ########-add noise on loss-#######
                if config.noise_style == "loss":
                    loss_noise = np.random.normal(0,config.sigma)
                    loss_noise = np.clip(loss_noise,-config.noise_bound,config.noise_bound)
                else:
                ########-add noise on parameters-#######
                    loss_noise = 0
                    sum_values = 0
                    t_vars = tf.trainable_variables()
                    for var in t_vars:
                        if 'd_h' in var.name and 'conv' in var.name:
                            values = self.sess.run(var.name)
                            for index, element in np.ndenumerate(values):
                                # print('value:', element)
                                sum_values = sum_values+element
                    norm_values = np.sqrt(sum_values)
                    for var in t_vars:
                        if 'd_h' in var.name and 'conv' in var.name:
                            values = self.sess.run(var.name)
                            para_noise = np.random.normal(0,config.sigma, values.shape)
                            para_noise = np.clip(para_noise,-config.noise_bound,config.noise_bound)
                            if norm_values > config.para_bound:
                                values = config.para_bound*values/norm_values + para_noise
                            else:
                                values = values + para_noise
                            self.sess.run(tf.assign(var, values))
                            #print("Var name:", var.name)
                
                # Update D network
                if self.y_dim:
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal,
                                                       self.noise: loss_noise,
                                                   })
                    self.writer.add_summary(summary_str, counter)
                     # Update D1 network
                    _, summary_str = self.sess.run([d1_optim, self.d1_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal,
                                                       self.noise: loss_noise,
                                                   })
    
                    self.writer.add_summary(summary_str, counter) 
                     # Update D2 network
                    _, summary_str = self.sess.run([d2_optim, self.d2_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       #sum_score_D_fake_label,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal,
                                                       self.noise: loss_noise,
                                                   })
    
                    self.writer.add_summary(summary_str, counter) 
                     # Update D3 network
                    _, summary_str = self.sess.run([d3_optim, self.d3_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       #sum_score_D_fake_label,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal,
                                                       self.noise: loss_noise,
                                                   })
    
                    self.writer.add_summary(summary_str, counter) 
                     # Update D4 network
                    _, summary_str = self.sess.run([d4_optim, self.d4_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       #sum_score_D_fake_label,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal,
                                                       self.noise: loss_noise,
                                                   })
    
                    self.writer.add_summary(summary_str, counter) 
                     # Update D5 network
                    _, summary_str = self.sess.run([d5_optim, self.d5_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       #sum_score_D_fake_label,
                                                       self.y: batch_labels,
                                                       self.y_normal: batch_labels_normal,
                                                       self.noise: loss_noise,
                                                   })
    
                    self.writer.add_summary(summary_str, counter) 
                    # Classifier  Update C network
                    if self.y_dim:
                        _, summary_str = self.sess.run([c_optim, self.c_sum],
                                                       feed_dict={
                                                           self.inputs: batch_images,
                                                           self.z: batch_z,
                                                           self.y: batch_labels,
                                                           self.y_normal: batch_labels_normal
                                                       })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={
                                          self.z: batch_z,
                                          #sum_score_D_fake_label,
                                          self.y: batch_labels,
                                          self.inputs: batch_images,
                                          self.y_normal: batch_labels_normal,
                                          self.prev_gmean: gmean,
                                          self.prev_gmean_: gmean_,
                                          self.prev_gvar: gvar,
                                          self.prev_gvar_: gvar_,
                                      })
                    # print('\ngener_model:', self.sess.run)
                    # x = tf.constant(gmean)
                    # print('g_mean:{}'.format(x))

                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={self.z: batch_z,
                                                 #sum_score_D_fake_label,
                                                 self.y: batch_labels,
                                                 self.inputs: batch_images,
                                                 self.y_normal: batch_labels_normal,
                                                 self.prev_gmean: gmean,
                                                 self.prev_gmean_: gmean_,
                                                 self.prev_gvar: gvar,
                                                 self.prev_gvar_: gvar_,
                                                 })
                    self.writer.add_summary(summary_str, counter)

                    # Classifier
                    errC = self.c_loss.eval({
                        self.inputs: batch_images,
                        self.z: batch_z,
                        self.y: batch_labels,
                        self.y_normal: batch_labels_normal,
                    })

                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.y: batch_labels,
                        self.y_normal: batch_labels_normal,
                        self.inputs: batch_images,
                        self.prev_gmean: gmean,
                        self.prev_gmean_: gmean_,
                        self.prev_gvar: gvar,
                        self.prev_gvar_: gvar_,
                        # self.noise: loss_noise
                    })

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.y: batch_labels,
                        self.noise: loss_noise
                    })

                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.y: batch_labels,
                        self.noise: loss_noise
                    })
                    
                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.noise: loss_noise,
                                                   })
                    self.writer.add_summary(summary_str, counter)
                    
                    # Update D1 network
#                    _, summary_str = self.sess.run([d1_optim, self.d1_sum],
#                                                   feed_dict={
#                                                       self.inputs: batch_images,
#                                                       self.z: batch_z,
#                                                       self.noise: loss_noise,
#                                                   })
#                    self.writer.add_summary(summary_str, counter)
                    
                    # Update D2 network
                    _, summary_str = self.sess.run([d2_optim, self.d2_sum],
                                                   feed_dict={
                                                       self.inputs: batch_images,
                                                       self.z: batch_z,
                                                       self.noise: loss_noise,
                                                   })
                    self.writer.add_summary(summary_str, counter)

                    # Classifier  Update C network
                    if self.y_dim:
                        _, summary_str = self.sess.run([c_optim, self.c_sum],
                                                       feed_dict={
                                                           self.inputs: batch_images,
                                                           self.z: batch_z,
                                                           self.y: batch_labels
                                                       })
                        self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={
                                          self.z: batch_z,
                                          self.inputs: batch_images,
                                          self.prev_gmean: gmean,
                                          self.prev_gmean_: gmean_,
                                          self.prev_gvar: gvar,
                                          self.prev_gvar_: gvar_
                                      })

                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str, gmean, gmean_, gvar, gvar_ = \
                        self.sess.run([g_optim, self.g_sum, self.gmean, self.gmean_, self.gvar, self.gvar_],
                                      feed_dict={self.z: batch_z,
                                                 self.inputs: batch_images,
                                                 self.prev_gmean: gmean,
                                                 self.prev_gmean_: gmean_,
                                                 self.prev_gvar: gvar,
                                                 self.prev_gvar_: gvar_
                                                 })

                    errG = self.g_loss.eval({
                        self.z: batch_z,
                        self.inputs: batch_images,
                        self.prev_gmean: gmean,
                        self.prev_gmean_: gmean_,
                        self.prev_gvar: gvar,
                        self.prev_gvar_: gvar_
                    })
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({
                        self.z: batch_z,
                        self.noise: loss_noise
                    })
                    errD_real = self.d_loss_real.eval({
                        self.inputs: batch_images,
                        self.noise: loss_noise
                    })

                counter += 1
                if self.y_dim:
                    print("Dataset: [%s] -> [%s] -> Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, "
                          "c_loss: %.8f" % (config.dataset, config.test_id, epoch, idx, batch_idxs,
                                            time.time() - start_time, errD_fake + errD_real, errG, errC))
                else:
                    print("Dataset: [%s] -> [%s] -> Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, "
                          % (config.dataset, config.test_id, epoch, idx, batch_idxs,
                             time.time() - start_time, errD_fake + errD_real, errG))



                if np.mod(counter, 100) == 1:

                    # Classifier
                    if self.y_dim:
                        samples, d_loss, c_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.c_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.y: sample_labels,
                                self.y_normal: sample_labels_normal,
                                self.prev_gmean: gmean,
                                self.prev_gmean_: gmean_,
                                self.prev_gvar: gvar,
                                self.prev_gvar_: gvar_,
                                self.noise: loss_noise

                            }
                        )
                        print("[Sample] d_loss: %.8f, g_loss: %.8f, c_loss: %.8f" % (d_loss, g_loss, c_loss))

                    else:
                        # Without Classifier
                        samples, d_loss, g_loss = self.sess.run(
                            [self.sampler, self.d_loss, self.g_loss],
                            feed_dict={
                                self.z: sample_z,
                                self.inputs: sample_inputs,
                                self.prev_gmean: gmean,
                                self.prev_gmean_: gmean_,
                                self.prev_gvar: gvar,
                                self.prev_gvar_: gvar_,
                                self.noise: loss_noise
                            }
                        )

                        print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

                if np.mod(counter, 3*np.floor(config.epoch/config.num_saver)) == 2:
                    self.save(config.checkpoint_dir, counter)
                    d_loss_list.append(errD_fake + errD_real)
                    g_loss_list.append(errG)

        with open('./samples/Adult/'+'log_{}_epochs{}.txt'.format(config.noise_style, config.epoch),'w',encoding='utf-8') as f:
            f.write('g_loss:\n')
            f.write(str(g_loss_list))
            f.write('\nd_loss:\n')
            f.write(str(d_loss_list))

    def discriminator(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']))
                # rt = tf.random_normal(self.c_dim.get_shape().as_list(), mean=0.0, stddev= 10)
                h0 = lrelu(
                    conv2d(x,self.c_dim + self.y_dim, name='d_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd_h3_lin')

                print("D Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D

    def discriminator1(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator1") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d1_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d1_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d1_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d1_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd1_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']))
                # rt = tf.random_normal(self.c_dim.get_shape().as_list(), mean=0.0, stddev= 10)
                h0 = lrelu(
                    conv2d(x,self.c_dim + self.y_dim, name='d1_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d1_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd1_h3_lin')

                print("D1 Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D
    def discriminator2(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator2") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d2_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d2_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d2_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d2_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd2_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']))
                # rt = tf.random_normal(self.c_dim.get_shape().as_list(), mean=0.0, stddev= 10)
                h0 = lrelu(
                    conv2d(x,self.c_dim + self.y_dim, name='d2_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d2_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd2_h3_lin')

                print("D2 Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D

    def discriminator3(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator3") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d3_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d3_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d3_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d3_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd3_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']))
                # rt = tf.random_normal(self.c_dim.get_shape().as_list(), mean=0.0, stddev= 10)
                h0 = lrelu(
                    conv2d(x,self.c_dim + self.y_dim, name='d3_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d3_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd3_h3_lin')

                print("D3 Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D
            
    def discriminator4(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator4") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d4_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d4_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d4_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d4_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd4_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']))
                # rt = tf.random_normal(self.c_dim.get_shape().as_list(), mean=0.0, stddev= 10)
                h0 = lrelu(
                    conv2d(x,self.c_dim + self.y_dim, name='d4_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d4_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd4_h3_lin')

                print("D4 Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D

    def discriminator5(self, image, y=None, reuse=False):
        with tf.variable_scope("discriminator5") as scope:
            if reuse:
                scope.reuse_variables()
            print(not self.y_dim)
            if not self.y_dim:
                h0 = lrelu(conv2d(image, self.df_dim, name='d5_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d5_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d5_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d5_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd5_h3_lin')

                return tf.nn.sigmoid(h4), h4, h3_f
            else:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                x = conv_cond_concat(image, yb)

                # tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']))
                # rt = tf.random_normal(self.c_dim.get_shape().as_list(), mean=0.0, stddev= 10)
                h0 = lrelu(
                    conv2d(x,self.c_dim + self.y_dim, name='d5_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d5_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                # print( "D Shape h1: " + str(h1.get_shape())) 

                # h2 = lrelu(self.d_bn2(linear(h1, self.dfc_dim, 'd_h2_lin'))) #new D remove

                # h2 = concat([h2, y], 1) #new D remove

                h3 = linear(h1, 1, 'd5_h3_lin')

                print("D5 Shape h3: " + str(h3.get_shape()))

                # return tf.nn.sigmoid(h3), h3, h2
                return tf.nn.sigmoid(h3), h3, h1  # new D

    def sampler_discriminator(self, input, y=None):
        with tf.variable_scope("discriminator") as scope:

            scope.reuse_variables()

            if not self.y_dim:
                h0 = lrelu(conv2d(input, self.df_dim, name='d_h0_conv'))
                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
                h2 = lrelu(self.d_bn2(
                    conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
                h3 = lrelu(self.d_bn3(
                    conv2d(h2, self.df_dim * 8, name='d_h3_conv')))

                h3_f = tf.reshape(h3, [self.batch_size, -1])
                # h4 = linear(tf.reshape(
                #     h3, [self.batch_size, -1]), 1, 'd_h3_lin')

                h4 = linear(h3_f, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h4)
            else:

                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])

                x = conv_cond_concat(input, yb)

                h0 = lrelu(
                    conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv'))

                h0 = conv_cond_concat(h0, yb)

                h1 = lrelu(self.d_bn1(
                    conv2d(h0, self.df_dim + self.y_dim, name='d_h1_conv')))

                h1 = tf.reshape(h1, [self.batch_size, -1])

                h1 = concat([h1, y], 1)

                h3 = linear(h1, 1, 'd_h3_lin')

                return tf.nn.sigmoid(h3)

    # Classifier
    def classification(self, image, y, reuse=False):

        with tf.variable_scope("classification") as scope:
            if reuse:
                scope.reuse_variables()
            assert (y is not None)

            yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
            x = conv_cond_concat(image, yb)

            h0 = lrelu(
                conv2d(x, self.c_dim + self.y_dim, name='c_h0_conv'))

            h0 = conv_cond_concat(h0, yb)

            # Classifier c_bn1()
            h1 = lrelu(self.c_bn1(
                conv2d(h0, self.df_dim + self.y_dim, name='c_h1_conv')))

            h1 = tf.reshape(h1, [self.batch_size, -1])  # h1 is 2-d
            h1 = concat([h1, y], 1)

            h3 = linear(h1, 1, 'c_h3_lin')

            return tf.nn.sigmoid(h3), h3, h1

    def generator(self, z, y=None):
        # Add
        with tf.variable_scope("generator") as scope:

            s_h, s_w = self.output_height, self.output_width

            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # input_height >= 16
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            if self.y_dim:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

            # input_height >= 16 , gf_dim = 64
            # self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 8 * s_h16 * s_w16, 'g_h0_lin', with_w=True)

            # input_height < 16
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 4 * s_h8 * s_w8, 'g_h0_lin', with_w=True)

            print(" G Shape z : " + str(self.z_.get_shape()))

            # input_height >= 16
            # self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])

            # input_height < 16
            self.h0 = tf.reshape(self.z_, [-1, s_h8, s_w8, self.gf_dim * 4])

            h0 = tf.nn.relu(self.g_bn0(self.h0))
            if self.y_dim:
                h0 = conv_cond_concat(h0, yb)

            # input_height < 16
            h2, self.h2_w, self.h2_b = deconv2d(
                h0, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)

            h2 = tf.nn.relu(self.g_bn2(h2))
            if self.y_dim:
                h2 = conv_cond_concat(h2, yb)

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)

            h3 = tf.nn.relu(self.g_bn3(h3))
            if self.y_dim:
                h3 = conv_cond_concat(h3, yb)

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:

            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width

            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)

            # input_height >= 16
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            # project `z` and reshape
            if self.y_dim:
                yb = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
                z = concat([z, y], 1)

            # input_height < 16
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim * 4 * s_h8 * s_w8, 'g_h0_lin',
                                                   with_w=True)  # 4*64=256

            # input_height >= 16
            # self.h0 = tf.reshape(self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])

            # input_height < 16
            self.h0 = tf.reshape(self.z_, [-1, s_h8, s_w8, self.gf_dim * 4])

            h0 = tf.nn.relu(self.g_bn0(self.h0))
            if self.y_dim:
                h0 = conv_cond_concat(h0, yb)

            # input_height >= 16
            # self.h1, self.h1_w, self.h1_b = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim * 4], name='g_h1',
            #                                         with_w=True) #2*2*256

            # h1 = tf.nn.relu(self.g_bn1(self.h1))
            # h1 = conv_cond_concat(h1, yb)

            # h2, self.h2_w, self.h2_b = deconv2d(
            #     h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True) # 4*4*128

            # input_height < 16
            h2, self.h2_w, self.h2_b = deconv2d(
                h0, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_h2', with_w=True)  # 2*2*128

            h2 = tf.nn.relu(self.g_bn2(h2))
            if self.y_dim:
                h2 = conv_cond_concat(h2, yb)

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_h3', with_w=True)  # 4*4*64 , 8*8*64

            h3 = tf.nn.relu(self.g_bn3(h3))
            if self.y_dim:
                h3 = conv_cond_concat(h3, yb)

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def load_dataset(self, load_fake_data=False):

        return self.load_tabular_data(self.dataset_name, self.input_height, self.y_dim, self.test_id, load_fake_data)

    def load_tabular_data(self, dataset_name, dim, classes=2, test_id='', load_fake_data=False):

        # self.train_data_path = f"./data/{dataset_name}/{dataset_name}"
        self.train_data_path = f'data/{dataset_name}/{dataset_name}'
        self.train_label_path = f'data/{dataset_name}/{dataset_name}_labels'

        if os.path.exists(self.train_data_path + ".csv"):

            X = pd.read_csv(self.train_data_path + ".csv", sep=',')
            print("Loading CSV input file : %s" % (self.train_data_path + ".csv"))

            self.attrib_num = X.shape[1]

            if self.y_dim:
                y = np.genfromtxt(open(self.train_label_path + ".csv", 'r'), delimiter=',')

                print("Loading CSV input file : %s" % (self.train_label_path + ".csv"))

                self.zero_one_ratio = 1.0 - (np.sum(y) / len(y))

        elif os.path.exists(self.train_data_path + ".pickle"):
            with open(self.train_data_path + '.pickle', 'rb') as handle:
                X = pickle.load(handle)
                #X = pk.load(handle)
            with open(self.train_label_path + '.pickle', 'rb') as handle:
                y = pickle.load(handle)
                #y = pk.load(handle)
            print("Loading pickle file ....")
        else:
            print("Error Loading Dataset !!")
            exit(1)

        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

        # Normalizing Initial Data
        X = pd.DataFrame(min_max_scaler.fit_transform(X))
        # X is [rows * config.attrib_num] 15000 * 23
        # print('dim:', dim)
        padded_ar = padding_duplicating(X, dim * dim)
        # print('padded:', padded_ar)
        X = reshape(padded_ar, dim)

        print("Final Real Data shape = " + str(X.shape))  # 15000 * 8 * 8

        if self.y_dim:
            y = y.reshape(y.shape[0], -1).astype(np.int16)
            y_onehot = np.zeros((len(y), classes), dtype=np.float)
            for i, lbl in enumerate(y):
                y_onehot[i, y[i]] = 1.0
            return X, y_onehot, y

        return X, None, None

    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(
            self.dataset_name, self.batch_size,
            self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "tableGAN_model"
        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                if f.startswith(f'{self.test_id}'):
                    file_idx = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_idx)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {f} is not an integer. Skipping')
            checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num + 1}'
            print(checkpoint_dir)
        else:
            checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        
        # print('highest num', highest_num)

        print(" [Saving checkpoints in " + checkpoint_dir + " ...")
        self.saver = tf.train.Saver(max_to_keep=50)
        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints from " + checkpoint_dir + " ...")

        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                print(f)
                if f.startswith(f'{self.model_dir}') and f.replace(self.model_dir, '') != '':
                    print(f)
                    file_name = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_name)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {file_name} is not an integer. Skipping')
            if highest_num == 0:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}'
            else:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num}'
        print(f'checkpoint dir: {checkpoint_dir}')
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter

            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
        
    def load_g(self, checkpoint_dir, num_epoch):
        import re
        print(" [*] Reading checkpoints from " + checkpoint_dir + " ...")

        if os.path.exists(f'{checkpoint_dir}/{self.model_dir}'):
            highest_num = 0
            for f in os.listdir(f'{checkpoint_dir}'):
                print(f)
                if f.startswith(f'{self.model_dir}') and f.replace(self.model_dir, '') != '':
                    print(f)
                    file_name = os.path.splitext(f)[0][-1]
                    try:
                        file_num = int(file_name)
                        if file_num > highest_num:
                            highest_num = file_num
                    except ValueError:
                        print(f'The file name {file_name} is not an integer. Skipping')
            if highest_num == 0:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}'
            else:
                checkpoint_dir = f'{checkpoint_dir}/{self.model_dir}_{highest_num}'
        print(f'checkpoint dir: {checkpoint_dir}')
        checkpoint_dir = os.path.join(checkpoint_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt:
            ckpt_name = 'tableGAN_model-{}'.format(num_epoch)

            self.saver.restore(self.sess, os.path.join(
                checkpoint_dir, ckpt_name))
            
            counter = int(
                next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))

            print(" [*] Success to read {}".format(ckpt_name))

            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
