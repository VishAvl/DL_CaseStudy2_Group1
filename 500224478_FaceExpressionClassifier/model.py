import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
from keras.layers import Input, Dropout, Dense
from keras.models import Model
from keras_vggface.vggface import VGGFace

def get_model(image_shape, num_classes, model_weights, unfreeze_layers=-3, drop_rate=0.5):
    
    input_layer = Input(shape=image_shape)
    vgg_base_model = VGGFace(include_top = False, input_shape = image_shape, pooling='avg')
    
    # Freeze all the layers till unfreeze layers
    for layer in vgg_base_model.layers[:unfreeze_layers]:
        layer.trainable = False
    
    for layer in vgg_base_model.layers[unfreeze_layers:]:
        layer.trainable = True
    
    x = vgg_base_model(input_layer)
    
    x = Dropout(drop_rate)(x)
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[input_layer], outputs=[output], name="Expression_Classifier")
    model.load_weights(model_weights)
    return model


if __name__ == "__main__":
	model_path = "vgg_face_weights2.h5"
	model = get_model(
			image_shape = (224, 224, 3),
			num_classes = 6,
			model_weights = model_path
		)

	print(model.summary())