# ResNet_source_code
source code notes for ResNet

reference:

- https://zhuanlan.zhihu.com/p/39316040
- https://www.jianshu.com/p/e502e4b43e6d

## Table of Contents

* [resnet_v2](#resnet_v2)
* [bottleneck](#bottleneck)
* [resent_v2_block](#resnet_v2_block)
* [resnet_v2_50](#resnet_v2_50)

## resnet_v2

### 函数介绍

可以认为是resnet构造器。resnet_v2_*()函数利用该构造器构造不同深度的resnet。

- 在 imagenet 上训练分类网络通常使用[224, 224] 的输入，经过$2^5=32$的下采样（即$output_stride=32$）后，得到$ 7*7 $的特征图；

- 如果是预测密集像素点的分类（如语意分割），那么建议使用$ (32 * n + 1) $的输入尺寸（如$ 321 * 321$），得到$ [(height - 1) / output_stride + 1, (width - 1) / output_stride + 1]$ 的输出。

  - 得到的特征图严格对齐输入的图片，如输入$ 225 * 225$ 得到$ 8 * 8 $的特征图。

  - resnet 需要使用全卷积模式，并且不使用全局池化，建议使用$ output_stride=16 $来增加输出特征图密度而不过分增加计算量和内存使用

### 参数

```python
def resnet_v2(inputs,
              blocks,
              num_classes=None,
              is_training=True,
              global_pool=True,
              output_stride=None,
              include_root_block=True,
              spatial_squeeze=True,
              reuse=None,
              scope=None):
```

#### Args:

- inputs: A tensor of size [batch, height_in, width_in, channels].

- blocks: A list of length equal to the number of ResNet blocks. Each element is a resnet_utils.Block object describing the units in the block. units组成block，blocks组成一个rennet

- num_classes: Number of predicted classes for classification tasks. If 0 or None, we return the features before the logit layer. 即输出的类别数

- is_training: whether batch_norm layers are in training mode. 可以区分【训练模式】or【测试模式】

- global_pool: If True, we perform global average pooling before computing the logits. Set to True for image classification, False for dense prediction. 我应该只用到image classification.

  - 以1000分类为例，Logits就是最后一层输出后，进入到Softmax函数之前得到的1000维向量。而Softmax仅仅是对Logits做了一个归一化。 Ref: https://blog.csdn.net/a2806005024/article/details/84113190

- output_stride: If None, then the output will be computed at the nominal network stride. If output_stride is not None, it specifies the requested ratio of input to output spatial resolution. 输出的压缩比例

- include_root_block: If True, include the initial convolution followed by max-pooling, if False excludes it. If excluded, `inputs` should be the results of an activation-less convolution. 是否包含根block（？）

- spatial_squeeze: if True, logits is of shape [B, C], if false logits is of shape [B, 1, 1, C], where B is batch_size and C is number of classes. 针对logits，也就是进入softmax之前的那个向量。正常情况下，logit.shape=[B, C]。To use this parameter, the input images must be smaller than 300x300 pixels 【指出input images要小于300x300像素】, in which case the output logit layer does not contain spatial information and can be removed.

- reuse: whether or not the network and its variables should be reused. To be able to reuse `scope` must be given.

- scope: Optional variable_scope.

#### Returns:

- net: A rank-4 tensor of size [batch, height_out, width_out, channels_out]. 
  - If global_pool is False, then height_out and width_out are reduced by a factor of output_stride compared to the respective height_in and width_in, else both height_out and width_out equal one. 我使用global_pool=True，意味着$ height_out=width_out=1$ （因为是分类网络，最后输出应当只有channels_out个类别，其他无。
  - If num_classes is 0 or None, then net is the output of the last ResNet block, potentially after global average pooling. 
  - If num_classes is a non-zero integer, net contains the pre-softmax activations.
- end_points: A dictionary from components of the network to the corresponding activation.

#### Raises:

- ValueError: If the target output_stride is not valid.

### 网络结构


```python
with tf.variable_scope(scope, 'resnet_v2', [inputs], reuse=reuse) as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, bottleneck,
                             resnet_utils.stack_blocks_dense],
                            outputs_collections=end_points_collection):
            with slim.arg_scope([slim.batch_norm], is_training=is_training):
                net = inputs
                if include_root_block:
                    if output_stride is not None:
                        if output_stride % 4 != 0:
                            raise ValueError(
                                'The output_stride needs to be a multiple of 4.')
                        output_stride /= 4
                    # We do not include batch normalization or activation functions in
                    # conv1 because the first ResNet unit will perform these. Cf.
                    # Appendix of [2].
                    with slim.arg_scope([slim.conv2d],
                                        activation_fn=None, normalizer_fn=None):
                        net = resnet_utils.conv2d_same(
                            net, 64, 7, stride=2, scope='conv1')
                    net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1')
                net = resnet_utils.stack_blocks_dense(
                    net, blocks, output_stride)
                # This is needed because the pre-activation variant does not have batch
                # normalization or activation functions in the residual unit output. See
                # Appendix of [2].
                net = slim.batch_norm(
                    net, activation_fn=tf.nn.relu, scope='postnorm')
                # Convert end_points_collection into a dictionary of end_points.
                end_points = slim.utils.convert_collection_to_dict(
                    end_points_collection)

                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(
                        net, [1, 2], name='pool5', keep_dims=True)
                    end_points['global_pool'] = net
                if num_classes is not None:
                    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                                      normalizer_fn=None, scope='logits')
                    end_points[sc.name + '/logits'] = net
                    if spatial_squeeze:
                        net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')
                        end_points[sc.name + '/spatial_squeeze'] = net
                    end_points['predictions'] = slim.softmax(
                        net, scope='predictions')
                return net, end_points
```

## bottleneck

### 函数介绍

- 它构造残差网络的基本单元，返回的是：output = shortcut + residual；

- 这个基本单元的特点是：参数少，训练时间短

### 参数

```python
def bottleneck(inputs, depth, depth_bottleneck, stride, rate=1,
               outputs_collections=None, scope=None):
```

#### Args:

- inputs: A tensor of size [batch, height, width, channels].
- depth: The depth of the ResNet unit output. 残差单元的输出层（最后一层）的通道数；
- depth_bottleneck: The depth of the bottleneck layers. 残差单元的前面2层的通道数。
- stride: The ResNet unit's stride. Determines the amount of downsampling of the units output compared to its input.
- rate: An integer, rate for atrous convolution.
- outputs_collections: Collection to add the ResNet unit output. 
  - Collection: `tensorflow`的`collection`提供一个全局的存储机制，不会受到`变量名`生存空间的影响。一处保存，到处可取。
- scope: Optional variable_scope.

#### Returns:

​      The ResNet unit's output.

### 源码

```python
with tf.variable_scope(scope, 'bottleneck_v2', [inputs]) as sc:
        depth_in = slim.utils.last_dimension(inputs.get_shape(), min_rank=4)
        preact = slim.batch_norm(
            inputs, activation_fn=tf.nn.relu, scope='preact')
        if depth == depth_in:
            shortcut = resnet_utils.subsample(inputs, stride, 'shortcut')
        else:
            shortcut = slim.conv2d(preact, depth, [1, 1], stride=stride,
                                   normalizer_fn=None, activation_fn=None,
                                   scope='shortcut')

        residual = slim.conv2d(preact, depth_bottleneck, [1, 1], stride=1,
                               scope='conv1')
        residual = resnet_utils.conv2d_same(residual, depth_bottleneck, 3, stride,
                                            rate=rate, scope='conv2')
        residual = slim.conv2d(residual, depth, [1, 1], stride=1,
                               normalizer_fn=None, activation_fn=None,
                               scope='conv3')

        output = shortcut + residual

        return slim.utils.collect_named_outputs(outputs_collections,
                                                sc.name,
                                                output)
```



## resnet_v2_block

## resnet_v2_50