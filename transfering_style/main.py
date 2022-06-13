import scipy
import argparse
import numpy as np
import progressbar
from scipy import io
from PIL import Image
import tensorflow as tf
from colorama import Fore, Style



def load_vgg_model(path, image_height, image_width, color_chanels):
    vgg = scipy.io.loadmat(VGG_MODEL)
    vgg_layers = vgg['layers']

    # Show vgg_layers:
    # _ = [print(f'{n}||{i[0][0][0][0]}') for n, i in enumerate(vgg_layers[0])]
    # List of convs layers.
    # conv_dict = {i[0][0][0][0]: n for n, i in enumerate(vgg_layers[0]) if i[0][0][0][0].startswith('conv')}
    # print(conv_dict)
    def _weights(layer, expected_layer_name):
        """
        Return the weights and bias from the VGG model for
        a given layer.
        """
        W = vgg_layers[0][layer][0][0][2][0][0]
        b = vgg_layers[0][layer][0][0][2][0][1]
        layer_name = vgg_layers[0][layer][0][0][0][0]
        assert layer_name == expected_layer_name
        return W, b

    def _relu(conv2d_layer):
        """
        Return the RELU function wrapped over a Tensorflow
        layer.
        Conv2d layer input.
        """
        return tf.nn.relu(conv2d_layer)

    def _conv2d(prev_layer, layer, layer_name):
        """
        Return the Conv2D layer using the weights, biases from
        the model at 'layer.'
        """
        W, b = _weights(layer, layer_name)
        W = tf.constant(W)
        b = tf.constant(np.reshape(b, (b.size)))
        return tf.nn.conv2d(prev_layer,
                            filters=W,
                            strides=[1, 1, 1, 1],
                            padding='SAME') + b

    def _conv2d_relu(prev_layer, layer, layer_name):
        """
        Return the Conv2D + RELU layer using the weights, biases
        from the VGG model at "layer".
        """
        return _relu(_conv2d(prev_layer, layer, layer_name))

    def _avgpool(prev_layer):
        """
        Return the AveragePooling layer.
        """
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')

    # Constructs the graph model.
    graph = {}
    graph['input'] = tf.Variable(np.zeros((1,
                                           image_height,
                                           image_width,
                                           color_chanels)), dtype=np.float32)
    graph['conv1_1'] = _conv2d_relu(graph['input'], 0, 'conv1_1')
    graph['conv1_2'] = _conv2d_relu(graph['conv1_1'], 2, 'conv1_2')
    graph['avgpool1'] = _avgpool(graph['conv1_2'])
    graph['conv2_1'] = _conv2d_relu(graph['avgpool1'], 5, 'conv2_1')
    graph['conv2_2'] = _conv2d_relu(graph['conv2_1'], 7, 'conv2_2')
    graph['avgpool2'] = _avgpool(graph['conv2_2'])
    graph['conv3_1'] = _conv2d_relu(graph['avgpool2'], 10, 'conv3_1')
    graph['conv3_2'] = _conv2d_relu(graph['conv3_1'], 12, 'conv3_2')
    graph['conv3_3'] = _conv2d_relu(graph['conv3_2'], 14, 'conv3_3')
    graph['conv3_4'] = _conv2d_relu(graph['conv3_3'], 16, 'conv3_4')
    graph['avgpool3'] = _avgpool(graph['conv3_4'])
    graph['conv4_1'] = _conv2d_relu(graph['avgpool3'], 19, 'conv4_1')
    graph['conv4_2'] = _conv2d_relu(graph['conv4_1'], 21, 'conv4_2')
    graph['conv4_3'] = _conv2d_relu(graph['conv4_2'], 23, 'conv4_3')
    graph['conv4_4'] = _conv2d_relu(graph['conv4_3'], 25, 'conv4_4')
    graph['avgpool4'] = _avgpool(graph['conv4_4'])
    graph['conv5_1'] = _conv2d_relu(graph['avgpool4'], 28, 'conv5_1')
    graph['conv5_2'] = _conv2d_relu(graph['conv5_1'], 30, 'conv5_2')
    graph['conv5_3'] = _conv2d_relu(graph['conv5_2'], 32, 'conv5_3')
    graph['conv5_4'] = _conv2d_relu(graph['conv5_3'], 34, 'conv5_4')
    graph['avgpool5'] = _avgpool(graph['conv5_4'])


    return graph

def resize_image(style, target):
    # Get shape of target and make the style image the same.
    target_shape = target.shape
    print(f'{Fore.LIGHTBLACK_EX}Target image shape: {target_shape}')
    style = np.asarray(style.resize((target_shape[1], target_shape[0]), Image.ANTIALIAS))
    print(f'Style image reshaped: {style.shape}')
    return style, target

def process_image(image):
    image = np.reshape(image, ((1,) + image.shape))
    image = image - MEAN_VALUES
    return image

def content_loss_func(sess, mdl):
    """
    Content loss function as defined in the paper.
    """
    def _content_loss(p, x):
        # N is the number of filters (at layer 1).
        N = p.shape[3]
        # M is the height times the wight of the feature map (at layer 1).
        M = p.shape[1] * p.shape[2]
        return (1 / (4 * N * M)) * tf.reduce_sum(tf.pow(x - p, 2))

    return _content_loss(sess.run(mdl['conv4_2']), mdl['conv4_2'])

def style_loss_func(sess, mdl):
    """
    Style loss function as defined in the paper.
    """
    # Which layers we use.
    STYLE_LAYERS = [
        ('conv1_1', 0.5),
        ('conv2_1', 1.0),
        ('conv3_1', 1.5),
        ('conv4_1', 3.0),
        ('conv5_1', 4.0),
    ]
    def _gram_matrix(F, N, M):
        """
        The gram matrix G.
        """
        Ft = tf.reshape(F, (M, N))
        return tf.matmul(tf.transpose(Ft), Ft)
    def _style_loss(a, x):
        """
        The style loss calculation.
        """
        # N is the number of the filters (at layer 1).
        N = a.shape[3]
        M = a.shape[1] * a.shape[2]
        A = _gram_matrix(a, N, M)
        G = _gram_matrix(x, N, M)
        result = (1 / (4 * N**2 * M**2)) * tf.reduce_sum(tf.pow(G - A, 2))
        return result
    E = [_style_loss(sess.run(mdl[layer_name]), mdl[layer_name]) for layer_name, _ in STYLE_LAYERS]
    W = [w for _, w in STYLE_LAYERS]
    loss = sum([W[l] * E[l] for l in range(len(STYLE_LAYERS))])
    return loss

def generate_noise_image(cnt_image, noise_ratio):
    """
    Returns a noise image intermixed with the content image
    at a cartain ratio.
    """
    noise_image = np.random.uniform(-20, 20,
                                   (1,
                                   cnt_image[0].shape[0],
                                   cnt_image[0].shape[1],
                                   cnt_image[0].shape[2])).astype('float32')
    inp_image = noise_image * noise_ratio + cnt_image * (1 - noise_ratio)
    return inp_image

def save_image(path, image):
    image = image + MEAN_VALUES
    image = image[0]
    image = np.clip(image, 0, 255).astype('uint8')
    im = Image.fromarray(image)
    im.save(path)



def arg_parser(parser_obj):
    parser_obj.add_argument('--style', default='starry_night.jpg', help='Choose style.')
    parser_obj.add_argument('--content', default='marilyn_monroe.jpg', help='Choose content image.')
    parser_obj.add_argument('--noise', type=float, default='0.6', help='How much noise is in the image.')
    parser_obj.add_argument('--beta', type=float, default='0.6', help='How much emphasis on content loss.')
    parser_obj.add_argument('--alpha', type=float, default='100', help='How much emphasis on style loss.')
    parser_obj.add_argument('--itter', type=int, default='100', help='How many itterations.')
    return parser_obj


if __name__ == "__main__":
    pars_obj = argparse.ArgumentParser()
    parser = arg_parser(pars_obj)
    args = parser.parse_args()

    BETA = args.beta
    ALPHA = args.alpha
    NOISE_RATIO = args.noise
    VGG_MODEL = './data/imagenet/imagenet-vgg-verydeep-19.mat'
    STYLE_IMAGE = f'data/styles/{args.style}'
    CONTENT_IMAGE = f'data/{args.content}'

    content_image = np.asarray(Image.open(CONTENT_IMAGE))
    style_image = Image.open(STYLE_IMAGE)
    # The mean used when the VGG was trained.
    # It is subtracted from the input to tge VGG model.
    MEAN_VALUES = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((1, 1, 1, 3))
    # Resize Style image.
    style_image, content_image = resize_image(style_image, content_image)

    # Process images.
    content_image = process_image(content_image).astype('float32')
    style_image = process_image(style_image).astype('float32')
    print(content_image.dtype)

    input_image = generate_noise_image(content_image, NOISE_RATIO)



    model = load_vgg_model(VGG_MODEL, style_image[0].shape[0], style_image[0].shape[1], style_image[0].shape[2])

    sess = tf.compat.v1.InteractiveSession()
    init = tf.compat.v1.global_variables_initializer()
    sess.run(init)
    sess.run(model['input'].assign(content_image))
    content_loss = content_loss_func(sess, model)
    sess.run(model['input'].assign(style_image))
    style_loss = style_loss_func(sess, model)
    total_loss = BETA * content_loss + ALPHA * style_loss
    optimizer = tf.compat.v1.train.AdamOptimizer(2.0)
    train_step = optimizer.minimize(total_loss)
    sess.run(tf.compat.v1.global_variables_initializer())
    sess.run(model['input'].assign(input_image))
    
    bar = progressbar.ProgressBar(max_value=args.itter)
    print(f'{Fore.BLUE}')
    for it in range(args.itter):
        sess.run(train_step)
        bar.update(it)
        if (it + 1) % args.itter == 0:   
            mixed_image = sess.run(model['input'])
            sess.run(tf.reduce_sum(mixed_image))
            save_image(f'{it+1}_{args.content}', mixed_image)
            bar.update(args.itter)
        
