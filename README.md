# Guide on tflite quantization and converting

Repository contains examples on how to do quantization aware training, and converting to tflite freezed graphdef, keras model, or checkpoint file.

### Requirements

tensorflow 1.4 or higher
python 3
```
pip3 install -r requirements.txt
```

## Running quantization aware training
Script contains training/exporting/testing of the dummy model for the image-to-image transformation

```
python3 quantization_aware_training.py
```
for the export add ```--mode export```, and for the testing ```--mode test```

### Converting to tflite trained model
there is a 3 different options for obtaining tflite model - from graphdef, from the keras model and from the checkpoint. Also you can choose to do post training quantization for reducing the model size, which will convert weights to uint8.
```
python3 convert_tf_model.py --model_type "type of the trained model" --model_path "path to the trained model"
```

## Useful resources
[Official documentation of the Tensorflow Lite](https://www.tensorflow.org/lite/guide/get_started) </br>
[Blog post with explanation of examples](https://medium.com/@lusinlu/mobile-inference-b943dc99e29b)
