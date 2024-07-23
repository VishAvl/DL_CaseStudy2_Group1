import tensorflow as tf

def get_model(model_path):
    return tf.keras.models.load_model(model_path)

if __name__ == "__main__":
    model_path = 'model.keras'
    model = get_model(model_path)
    model.summary()