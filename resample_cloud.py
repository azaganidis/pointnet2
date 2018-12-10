import tensorflow as tf
def init_resample2(data,num_out,num_in, scale=(130.0, 1)):
    data=tf.boolean_mask(data, data[:,-1]>=0)
    numPoints=tf.shape(data)[0]
    def upS():
        upsample_I = tf.random_uniform([num_out-numPoints],maxval=numPoints,dtype=tf.int32)
        return tf.concat([data,tf.gather(data,upsample_I)],axis=0)
    def downS():
        downsmpl_I = tf.random_uniform([num_out],maxval=numPoints, dtype=tf.int32)
        return tf.gather(data,downsmpl_I)
    with tf.control_dependencies([data]):
        sampled=tf.cond(numPoints<num_out,lambda: upS(), lambda: downS())
    #extr_val=tf.reduce_max(tf.abs(sampled[:,:3]))
    #xyz_=tf.divide(sampled[:,:3],extr_val)
    xyz_=tf.divide(sampled[:,:3],scale[0])
    channels=tf.divide(sampled[:,3:],scale[1])
    sampled=tf.concat([xyz_,channels], axis=1)
    return sampled

