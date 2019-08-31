import tensorflow as tf
import argparse
from PIL import Image
import imageio
import numpy as np
import os
import tensorflow.lite as lite


# generate random data points for feeding to network
def load_data_pair(size, in_channels, out_channels, scale):
    image = np.random.rand(size, size, in_channels) * scale
    label = np.random.rand(size, size, out_channels) * scale

    return image, label


# general model for the image-to-image transformation
def model(x, is_training, keep_prob):
    with tf.variable_scope('quantized_model'):
        x = tf.layers.conv2d(x, filters=16, kernel_size=3, strides=2, padding='same')
        x = tf.nn.relu(x)
        x = tf.layers.batch_normalization(x, training=is_training, fused=False)
        x = tf.nn.dropout(x, keep_prob=keep_prob)
        sp_size = tf.shape(x)[1:3]
        x = tf.image.resize_bilinear(x, size= 2 * sp_size)
        x = tf.layers.conv2d(x, filters=3, kernel_size=1, strides=1, padding='same')

        return x


def train(args):
    # define the placeholders of the graph
    image_tf = tf.placeholder(tf.float32, [1, args.spatial_size, args.spatial_size, args.input_channels], name='input')
    label_tf = tf.placeholder(tf.float32, [1, args.spatial_size, args.spatial_size, args.output_channels], name='output')
    keep_prob_tf = tf.placeholder_with_default(input=1., shape=[], name="keep_prob")

    # build the model and insert into the graph fake nodes(min/max) for further quantization
    with tf.variable_scope('quantize'):
        output= model(x=image_tf, is_training=True, keep_prob=keep_prob_tf)
    tf.contrib.quantize.create_training_graph(quant_delay=0)

    # definition of the loss, the optimizer
    loss = tf.losses.mean_squared_error(labels=image_tf, predictions=output)
    saver = tf.train.Saver(max_to_keep=1000)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer().minimize(loss)

    # run training for the several iterations and save dummy checkpoint
    with tf.Session() as sess:

        tf.global_variables_initializer().run()

        for iter in range(args.iterations):
            image, label = load_data_pair(args.spatial_size,args.input_channels, args.output_channels, args.scale)
            training_loss, _ = sess.run([loss, optimizer], feed_dict={image_tf:image[None, ...], label_tf:label[None, ...], keep_prob_tf:0.7})

            print('loss - ', training_loss)

        saver.save(sess=sess, save_path=os.path.join(args.chkp, 'qt_test'))


def export(args):
    graph = tf.Graph()
    with graph.as_default():
        # define the input placeholder and the model (add useless op after the main model,
        # so tflite will not ignore fake min/max nodes of the last layer)
        input = tf.placeholder(tf.float32, [1, args.spatial_size, args.spatial_size, args.input_channels], name='input')
        with tf.variable_scope('quantize'):
            output = model(x=input, is_training=False, keep_prob=1.)
            output = tf.maximum(output, -1e27)

        # define eval graph, by quantizing the weights of the model with learned min/max values for each layer
        g = tf.get_default_graph()
        tf.contrib.quantize.create_eval_graph(input_graph=g)
        saver = tf.train.Saver()

        graph.finalize()

        with open('eval.pb', 'w') as f:
            f.write(str(g.as_graph_def()))

    with tf.Session(graph=graph) as session:
        checkpoint = tf.train.latest_checkpoint(args.chkp)
        saver.restore(session, checkpoint)
        # fix the input, output, choose types of the weights and activations for the tflite model
        converter = lite.TFLiteConverter.from_session(session, [input], [output])
        converter.inference_type = tf.uint8
        converter.inference_input_type = tf.uint8
        input_arrays = converter.get_input_arrays()
        converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}

        flatbuffer = converter.convert()

        with open('test.tflite', 'wb') as outfile:
            outfile.write(flatbuffer)
    print('Model successfully converted and saved in the project directory')


def test(args):
    # read or create test image, convert to uint8 for the inference with quantized activation
    if args.test_image is not None:
        test_image = imageio.imread(args.test_image)
        test_image = np.array(Image.fromarray(test_image).resize([args.spatial_size, args.spatial_size]), dtype=np.uint8)
    else:
        test_image = np.array(np.random.rand(args.spatial_size, args.spatial_size, args.input_channels) * args.scale, dtype=np.uint8)

    # define the tflite interpreter and infer on the test image
    interpreter = lite.Interpreter('test.tflite')

    input_info = interpreter.get_input_details()[0]
    output_info = interpreter.get_output_details()[0]

    interpreter.resize_tensor_input(input_info['index'], (1, args.spatial_size, args.spatial_size, args.input_channels))
    interpreter.allocate_tensors()

    interpreter.set_tensor(input_info['index'], test_image[None, ...])
    interpreter.invoke()
    result = interpreter.get_tensor(output_info['index'])
    imageio.imwrite('./tflite_res.png', result[0])
    print('Result saved in the training directory')


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iterations', default=10, type=int, help='training epochs')
    parser.add_argument('--input_channels', default=3, type=int, help='number of channels of the input image')
    parser.add_argument('--output_channels', default=1, type=int, help='number of channels of the output')
    parser.add_argument('--spatial_size', default=288, type=int, help='image size')
    parser.add_argument('--scale', default=255, type=float, help='scale transform of the input data')
    parser.add_argument('--mode', default='train', type=str, help='run mode, train/export/test')
    parser.add_argument('--chkp', default='chkp', type=str, help='path to checkpoint directory')
    parser.add_argument('--dataset_path', type=str, help='path to dataset')
    parser.add_argument('--test_image', type=str, help='path to testing image for the "test" mode')

    args = parser.parse_args()

    if args.mode == 'train':
        if not os.path.exists(args.chkp):
            os.makedirs(args.chkp)
        train(args)
    if args.mode == 'export':
        export(args)
    if args.mode == 'test':
        test(args)


if __name__ == '__main__':
    run()