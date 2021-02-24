import cv2
import numpy as np
import pathlib
from PIL import Image
import tensorflow as tf
import time

from PIL import Image

def load_labels(filename):
  with open(filename, 'r') as f:
    return [line.strip() for line in f.readlines()]

def classifier(img):

    model_path = 'data/model.tflite'
    label_path = 'data/parking_labels.txt'
    #image_path = '/Volumes/Macintosh SSD/CapstoneProject/image_classification/data/testspace'
    image_path = '/Volumes/Macintosh SSD/CapstoneProject/image_classification/data/testspace_3'
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    # print(input_details)
    # print(output_details)

    # check the type of the input tensor
    floating_model = input_details[0]['dtype'] == np.float32
    start_time = time.time()

        

            # NxHxWxC, H:1, W:2 
            # open the image and resize to the input standar
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    #img = Image.open(file).resize((224,224))
    new_image = Image.fromarray(img).resize((224,224))
            # add dimensions
    input_data = np.expand_dims(new_image, axis=0)

            # Input standard deviation = Input mean = 127.5 (default)
    if floating_model:
        input_data = (np.float32(input_data) - 127.5) / 127.5


    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_path)

    top_result = []
    #frame = cv2.imread(str(file), cv2.IMREAD_COLOR)
    for i in top_k:
        if floating_model:
            #print('{:08.6f}: {}'.format(float(results[i]), labels[i]))
            if labels[i] == "Empty":
                return True
            else:
                #print('Occupied')
                return False
            #top_result.append((i, float(results[i])))
        else:
            print('{:08.6f}: {}'.format(float(results[i] / 255.0), labels[i]))
            top_result.append((i, float(results[i] / 255.0)))

        #print("========================")
    #         display_results(top_result, frame, labels, str(file))
    # stop_time = time.time()
    # print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
    # key = cv2.waitKey(0)
    # if key == 27:  
    #     cv2.destroyAllWindows()
    return True

