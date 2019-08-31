import tensorflow as tf
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--post_train_quantize', default=False, type=bool, help='do post training quantization')
parser.add_argument('--model_path', type=str, help='path to the trained model')
parser.add_argument('--model_type', default='pb', type=str, help='type of the saved model, pb/chkp/keras')
parser.add_argument('--input', type=str, help='name of the input, needed only in case of model_type == pb')
parser.add_argument('--output', type=str, help='name of the output, needed only in case of the model_type == pb')


args = parser.parse_args()
tf.enable_eager_execution()

# create object of the TFLiteConverter type, depending on saved model
if args.model_type == 'pb':
    input = [args.input]
    output = [args.output]
    converter = tf.lite.TFLiteConverter.from_frozen_graph(args.model_path, input, output)
elif args.model_type == 'chkp':
    converter = tf.contrib.lite.TFLiteConverter.from_saved_model(args.model_path)
elif args.model_type == 'keras':
    converter = tf.lite.TFLiteConverter.from_keras_model_file(args.model_path)

# do quantization of the weights
if args.post_train_quantize == True:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

# convert the model
tflite_model = converter.convert()
tflite_model_quant_file = "./test.tflite"
tflite_model_quant_file.write_bytes(tflite_model)