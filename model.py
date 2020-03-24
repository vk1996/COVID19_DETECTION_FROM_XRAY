'''
Copyright (C) 2020 Sam Sepiol

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation,Add,GlobalAveragePooling2D,Multiply
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
import utils




def get_shape(z):
    if hasattr(z,'_keras_shape'):
        #print(z._keras_shape[-1])
        return z._keras_shape[-1]
    elif hasattr(z,'shape'):
        #if hasattr(x.shape,'value'):
        #print(z.shape[-1].value)
        return z.shape[-1].value

def se(x_):
    y=GlobalAveragePooling2D()(x_)
    y=Reshape(target_shape=(1,1,int(get_shape(y))))(y)
    y=Conv2D(int(get_shape(y)*0.25),(1,1),padding='same')(y)
    y=swish(y)
    y=Conv2D(int(get_shape(x_)),(1,1),padding='same')(y)
    y=Activation('sigmoid')(y)
    y=Multiply()([x_,y])
    y=Conv2D(int(get_shape(y)),(1,1),padding='same')(y)
    y=swish(y)
    return y


    
def swish(x_):
    #return Lambda(swish_fn, output_shape=x_._keras_shape[1:])(x_)
    #return Multiply()([x_,Activation('tanh')(Activation('softplus')(x_))])
    #return Multiply()([x_,Activation('sigmoid')(x_)])
    return Activation('relu')(x_)

def build_model(image_size,
                redux=1.0,
                l2_regularization=0.0,
                show_flops=False,
                first_stride=2):

    l2_reg = l2_regularization
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]
    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))
    
    conv1 = Conv2D(int(16*redux), (3,3), strides=(first_stride, first_stride), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x)
    bn1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    #conv1 = ELU(name='elu1')(conv1)
    act1=se(swish(bn1))
    #add1=Add()([conv1,act1])
    #bn1_a = BatchNormalization(axis=3, momentum=0.99, name='bn1_a')(add1)
    #add_act1=swish(bn1_a)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(act1)
    '''
    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(add_act1)
    bn2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    #conv2 = ELU(name='elu2')(conv2)
    act2=Activation('relu')(bn2)
    add2=Add()([conv2,act2])
    bn2_a = BatchNormalization(axis=3, momentum=0.99, name='bn2_a')(add2)
    add_act2=Activation('relu')(bn2_a)
    #pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(add_act2)
    '''
    conv3 = Conv2D(int(16*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool1)
    #conv3=SeparableConv2D(filters=int(64*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool1)
    bn3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    #conv3 = ELU(name='elu3')(conv3)
    act3=swish(bn3)
    add3=Add()([pool1,act3])
    bn3_a = BatchNormalization(axis=3, momentum=0.99, name='bn3_a')(add3)
    add_act3=se(swish(bn3_a))
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(add_act3)
    
    conv4 = Conv2D(int(32*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    #conv4=SeparableConv2D(filters=int(64*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool3)
    bn4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    #conv4 = ELU(name='elu4')(conv4)
    act4=swish(bn4)
    #add4=Add()([pool3,act4])
    #bn4_a = BatchNormalization(axis=3, momentum=0.99, name='bn4_a')(add4)
    #add_act4=swish(bn4_a)
    act4=se(act4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(act4)

    conv5 = Conv2D(int(32*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    #conv5=SeparableConv2D(filters=int(48*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool4)
    bn5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    #conv5 = ELU(name='elu5')(conv5)
    act5=swish(bn5)
    add5=Add()([pool4,act5])
    bn5_a = BatchNormalization(axis=3, momentum=0.99, name='bn5_a')(add5)
    add_act5=swish(bn5_a)
    add_act5=se(add_act5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(add_act5)

    conv6 = Conv2D(int(64*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    #conv6=SeparableConv2D(filters=int(48*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool5)
    bn6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    #conv6 = ELU(name='elu6')(conv6)
    act6=swish(bn6)
    #add6=Add()([pool5,act6])
    #bn6_a = BatchNormalization(axis=3, momentum=0.99, name='bn6_a')(add6)
    #add_act6=swish(bn6_a)
    act6=se(act6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(act6)

    conv7 = Conv2D(int(64*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    #conv7=SeparableConv2D(filters=int(32*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool6)
    bn7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    #conv7 = ELU(name='elu7')(conv7)
    act7=swish(bn7)
    add7=Add()([pool6,act7])
    bn7_a = BatchNormalization(axis=3, momentum=0.99, name='bn7_a')(add7)
    add_act7=swish(bn7_a)
    add_act7=se(add_act7)
    
    '''
    conv8 = Conv2D(int(64*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8')(add_act7)
    #conv7=SeparableConv2D(filters=int(32*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool6)
    bn8 = BatchNormalization(axis=3, momentum=0.99, name='bn8')(conv8)
    #conv7 = ELU(name='elu7')(conv7)
    act8=swish(bn8)
    add8=Add()([conv7,act8])
    bn8_a = BatchNormalization(axis=3, momentum=0.99, name='bn8_a')(add8)
    add_act8=swish(bn8_a)
    add_act8=se(add_act8)
    
    conv9 = Conv2D(int(128*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9')(add_act8)
    #conv7=SeparableConv2D(filters=int(32*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool6)
    bn9 = BatchNormalization(axis=3, momentum=0.99, name='bn9')(conv9)
    act9=swish(bn9)
    add_act9=se(act9)
    '''
    
    
    model = Model(inputs=x, outputs=add_act7)

    if show_flops:
        utils.show_flops(model,image_size)

    
    return model
