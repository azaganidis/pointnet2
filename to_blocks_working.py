import tensorflow as tf
###Tensor definition
def init_block_model(points_per_block, block_size=10.0, stride=1.0, num_in=5):
    print points_per_block
    data = tf.placeholder(tf.float32,shape=(None,num_in))
    limit_min=tf.reduce_min(data,axis=0)
    limit_max=tf.reduce_max(data,axis=0)
    limit =tf.subtract(limit_max,limit_min)

    num_block= tf.ceil(tf.divide(tf.subtract(limit, block_size), stride)) + 1
    blocks = tf.multiply(num_block[0],num_block[1])
    c_index=tf.constant(0.)

    data_out=tf.fill((1,points_per_block,num_in+3),0.)
    def body (c_index, data_out ): 
        xind = tf.floordiv(c_index, num_block[1])
        yind = tf.mod(c_index, num_block[1])
        xbeg=limit_min[0]+xind*stride
        ybeg=limit_min[1]+yind*stride

        # Only for new versions of tensorflow
        #xcond = tf.logical_and(tf.less_equal(tf.gather(data,0, axis=1), xbeg+block_size), tf.greater_equal(tf.gather(data,0, axis=1), xbeg)) 
        #ycond = tf.logical_and(tf.less_equal(tf.gather(data,1, axis=1), ybeg+block_size), tf.greater_equal(tf.gather(data,1, axis=1), ybeg)) 

        # Only for old versions of tensorflow
        gx = tf.transpose(tf.gather(tf.transpose(data),0))
        gy = tf.transpose(tf.gather(tf.transpose(data),1))
        label = tf.transpose(tf.gather(tf.transpose(data),4))
        xcond = tf.logical_and(tf.less_equal(gx, xbeg+block_size), tf.greater_equal(gx, xbeg)) 
        ycond = tf.logical_and(tf.less_equal(gy, ybeg+block_size), tf.greater_equal(gy, ybeg)) 
        label_condition = tf.greater_equal(label,0)

        combined_condition = tf.logical_and(xcond,ycond)
        combined_condition = tf.logical_and(label_condition, combined_condition)
        data_h = tf.boolean_mask(data, combined_condition) 

        #The number of points in this block
        num_count = tf.count_nonzero(combined_condition, dtype=tf.int32)

        #for safe execution, as cond evaluates all parameters first
        num_safe = tf.cond(num_count<1,lambda: tf.constant(1),lambda: num_count)
        points_per_block_diff_safe = tf.cond(points_per_block-num_count<1,lambda: tf.constant(1),lambda: points_per_block-num_count)
        data_safe = tf.cond(num_count<1,lambda: tf.constant([[0. for _ in range(num_in)]]),lambda: data_h) 

        upsample_indices   = tf.random_uniform([points_per_block_diff_safe ], maxval=num_safe,dtype=tf.int32)
        downsample_indices = tf.random_uniform([points_per_block],      maxval=num_safe,dtype=tf.int32)

        upsample_f   = lambda: tf.concat([data_h, tf.gather(data_safe, upsample_indices)], axis=0)
        downsample_f = lambda: tf.gather(data_safe, downsample_indices)

        #downsample if the number of points is higher than the required. 
        sampled   = tf.cond(num_count>=points_per_block, downsample_f, upsample_f)
        xyz_ = tf.divide(sampled[:,:3], limit_max[:3])
        XY = tf.subtract(sampled[:,:2], tf.reduce_min(sampled[:,:2])+block_size/2)
        sampled=tf.expand_dims(tf.concat([XY,sampled[:,2:4], xyz_, tf.expand_dims(sampled[:,4],1)], axis=1) ,0)

        enough_ = lambda: tf.concat([sampled, data_out ], axis=0)
        not_enough_ = lambda: data_out

        result =tf.cond(num_count<500, not_enough_, enough_) 
        return tf.add(c_index,1) , result

    end_cond=lambda c_index, data_out:c_index<blocks

    c_, data_out = tf.while_loop(end_cond, body, [c_index, data_out], shape_invariants=[c_index.get_shape(), tf.TensorShape([None,points_per_block,num_in+3])])
    return data_out[:-1,:], data


