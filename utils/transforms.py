import jax 
import jax.numpy as jnp
import tensorflow as tf

def low_pass_filter(img : jax.Array, method : str, **method_kwargs) -> jax.Array: 
    img = tf.cast(img, tf.float32)
    
    if method == 'downsample':
        low_pass_img =  low_pass_filter_downsample(img, **method_kwargs)
    else:
        raise ValueError('Incorrect method provided')
    return low_pass_img 

def low_pass_filter_downsample(image, scale_factor : str =0.5, interp_method : str ='bilinear'):

    # Get the image size
    C, H, W = image.shape

    image_tf = tf.transpose(image, perm=[1, 2, 0])

    # Downsample Upsample
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    downsampled_image = tf.image.resize(image_tf, (new_H, new_W), method=interp_method, antialias=False)
    upsampled_image = tf.image.resize(downsampled_image, (H, W), method=interp_method, antialias=False)
    
    upsampled_image = tf.transpose(upsampled_image, perm=[2, 0, 1])
    
    return upsampled_image