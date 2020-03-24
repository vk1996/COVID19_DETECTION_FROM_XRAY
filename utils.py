import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def show_flops(model,size):
   
    run_meta = tf.RunMetadata()
    curr_sess= tf.keras.backend.get_session()
    with curr_sess as sess:
        tf.keras.backend.set_session(sess)
        print(1,size[0],size[1],size[2])
        net = model(tf.keras.backend.placeholder(shape=(1,size[0],size[1],size[2])))

        opts = tf.profiler.ProfileOptionBuilder.float_operation()    
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()    
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        print("num flops: {:,} --- num parmas: {:,}".format(flops.total_float_ops, params.total_parameters))
