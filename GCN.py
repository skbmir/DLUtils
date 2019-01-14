#!/usr/bin/env python3

import keras
from keras.models import Model
from keras.layers import Conv2D, Conv2DTranspose, ReLU, add, BatchNormalization
from keras.layers import Dropout, MaxPool2D, UpSampling2D, Input

def ResidualGlobalConvolution(in_filters, out_filters, k, padding='same'):
    def block(x):
        x1 = Conv2D(filters=in_filters, kernel_size=(1, k), padding=padding)(x)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x1 = Conv2D(filters=in_filters, kernel_size=(k, 1), padding=padding)(x1)
        x1 = BatchNormalization()(x1)
        x1 = ReLU()(x1)
        x2 = Conv2D(filters=in_filters, kernel_size=(k, 1), padding=padding)(x)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        x2 = Conv2D(filters=in_filters, kernel_size=(1, k), padding=padding)(x2)
        x2 = BatchNormalization()(x2)
        x2 = ReLU()(x2)
        x3 = add([x1, x2])
        x3 = Conv2D(filters=out_filters, kernel_size=(1,1), padding=padding)(x3)
        x3 = BatchNormalization()(x3)
        x = add([x, x3])
        return x
    return block
        
def GlobalConvolution(filters, k, padding='same'):
    def block(x):
        x1 = Conv2D(filters=filters, kernel_size=(k, 1), padding=padding)(x)
        x1 = Conv2D(filters=filters, kernel_size=(1, k), padding=padding)(x1)
        x2 = Conv2D(filters=filters, kernel_size=(1, k), padding=padding)(x)
        x2 = Conv2D(filters=filters, kernel_size=(k, 1), padding=padding)(x2)
        x = add([x1, x2])
        return x
    return block

def BoundaryRefinement(filters, kernel_size=(3,3), padding='same'):
    def block(x):
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x)
        x1 = ReLU()(x1)
        x1 = Conv2D(filters=filters, kernel_size=kernel_size, padding=padding)(x1)
        x = add([x, x1])
        return x
    return block

def DeconvBlock(filters, kernel_size, strides, padding='same'):
    def block(x):
        x = Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        x = ReLU()(x)
        return x
    return block

def GCN(input_shape=(512, 512, 3),
        encoder_filters=(64, 256, 512, 1024, 2048),
        input_conv_kernel_size=(3,3),
        binary_segmentation=False,
        num_classes=20, 
        gc_br_filters=-1, 
        decoder_type='upsampling',
        padding='same',
        k=15):

    if binary_segmentation:
        num_classes = 1

    if gc_br_filters == -1:
        gc_br_filters = num_classes + 1

    in_layer = Input(shape=input_shape)
    
    conv1 = Conv2D(filters=encoder_filters[0], 
                   kernel_size=input_conv_kernel_size,
                   padding=padding)(in_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = ReLU()(conv1)
    conv1_pool = MaxPool2D(pool_size=(2,2), padding=padding)(conv1)

    res2 = ResidualGlobalConvolution(in_filters=encoder_filters[0], 
                                     out_filters=encoder_filters[1],
                                     k=k,
                                     padding=padding)(conv1_pool)
    res2_pool = MaxPool2D(pool_size=(2,2), padding=padding)(res2)
    
    res3 = ResidualGlobalConvolution(in_filters=encoder_filters[1],
                                     out_filters=encoder_filters[2],
                                     k=k,
                                     padding=padding)(res2_pool)
    res3_pool = MaxPool2D(pool_size=(2,2), padding=padding)(res3)

    res4 = ResidualGlobalConvolution(in_filters=encoder_filters[2],
                                     out_filters=encoder_filters[3],
                                     k=k,
                                     padding=padding)(res3_pool)
    res4_pool = MaxPool2D(pool_size=(2,2), padding=padding)(res4)

    res5 = ResidualGlobalConvolution(in_filters=encoder_filters[3],
                                     out_filters=encoder_filters[4],
                                     k=k,
                                     padding=padding)(res4_pool)
    res5_pool = MaxPool2D(pool_size=(2,2), padding=padding)(res5)

    gcn_res2 = GlobalConvolution(filters=gc_br_filters, k=k, padding=padding)(res2_pool)
    gcn_res3 = GlobalConvolution(filters=gc_br_filters, k=k, padding=padding)(res3_pool)
    gcn_res4 = GlobalConvolution(filters=gc_br_filters, k=k, padding=padding)(res4_pool)
    gcn_res5 = GlobalConvolution(filters=gc_br_filters, k=k, padding=padding)(res5_pool)

    br_res2 = BoundaryRefinement(filters=gc_br_filters, padding=padding)(gcn_res2)
    br_res3 = BoundaryRefinement(filters=gc_br_filters, padding=padding)(gcn_res3)
    br_res4 = BoundaryRefinement(filters=gc_br_filters, padding=padding)(gcn_res4)
    br_res5 = BoundaryRefinement(filters=gc_br_filters, padding=padding)(gcn_res5)

    if decoder_type == 'transpose':
        deconv = Conv2DTranspose(filters=gc_br_filters, 
                                 kernel_size=(2,2), 
                                 strides=(2,2), 
                                 padding=padding)
        deconv_out = Conv2DTranspose(filters=num_classes, kernel_size=(2,2), strides=(2,2), padding=padding)
    else:
        deconv = deconv_out = UpSampling2D(size=(2,2))

    deconv_res5 = deconv(br_res5)
    add_res54 = add([deconv_res5, br_res4])
    add_res54_br = BoundaryRefinement(filters=gc_br_filters, padding=padding)(add_res54)

    deconv_res4 = deconv(add_res54_br)
    add_res43 = add([deconv_res4, br_res3])
    add_res43_br = BoundaryRefinement(filters=gc_br_filters, padding=padding)(add_res43)

    deconv_res3 = deconv(add_res43_br)
    add_res32 = add([deconv_res3, br_res2])
    add_res32_br = BoundaryRefinement(filters=gc_br_filters, padding=padding)(add_res32)

    deconv_res2 = deconv(add_res32_br)
    deconv_res2_br = BoundaryRefinement(filters=gc_br_filters, padding=padding)(deconv_res2)

    deconv_out = deconv_out(deconv_res2_br)
    br_out = BoundaryRefinement(filters=gc_br_filters, padding=padding)(deconv_out)

    out = None

    if binary_segmentation:
        out = Conv2D(filters=1, kernel_size=(3,3), padding=padding, activation='sigmoid')(br_out)
    else:
        out = Conv2D(filters=num_classes, kernel_size=(3,3), padding=padding, acrivation='softmax')(out)

    model = Model(inputs=in_layer, outputs=out)

    return model
