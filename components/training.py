import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from kerastuner import RandomSearch
import os



#Make CNN
def build_cnn_model(hp):
    model = tf.keras.Sequential()
    
    print("here1")

    
    isFirstConvLayer = True

    # Tuning the number of sets of Conv Layers + Max Pooling
    for i in range(hp.Int('num_sets_conv_maxpool', 1, 3)):
        for j in range(hp.Int(f'num_conv_layers_set_{i+1}', 1, 3)):
            print("here2")
            # Add Conv2D layer
            if isFirstConvLayer:
                print("here3")
                # Specify input_shape only for the first Conv2D layer
                model.add(layers.Conv2D(
                    filters=hp.Int(f'conv_{i+1}_{j+1}_filters', min_value=32, max_value=128, step=32),
                    kernel_size=hp.Choice(f'conv_{i+1}_{j+1}_kernel', values=[3, 5]),
                    activation='relu',
                    padding=hp.Choice(f'conv_{i+1}_{j+1}_padding', values=['valid', 'same']),
                    input_shape=(256, 256, 3)
                ))
                isFirstConvLayer = False
                print("here4")
            else:
                # Omit input_shape for subsequent Conv2D layers
                model.add(layers.Conv2D(
                    filters=hp.Int(f'conv_{i+1}_{j+1}_filters', min_value=32, max_value=128, step=32),
                    kernel_size=hp.Choice(f'conv_{i+1}_{j+1}_kernel', values=[3, 5]),
                    activation='relu',
                    padding=hp.Choice(f'conv_{i+1}_{j+1}_padding', values=['valid', 'same'])
                ))
        
        # Max Pooling layer after each set of Conv Layers
        model.add(layers.MaxPooling2D(
            pool_size=hp.Int(f'maxpool_{i+1}_size', 2, 4, step=1)
        ))
    
    model.add(tf.keras.layers.Flatten())
    
    model.add(tf.keras.layers.Dense(units=hp.Int('dense_units', min_value=64, max_value=512, step=64),
                                    activation='relu'))
    model.add(tf.keras.layers.Dense(36, activation='softmax'))  # Assuming 10 classes for the output layer

    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    
    return model


#Make tuner
def run_tuner(train_generator, val_generator):
    '''c = {
        "max_trials": 100,
        "executions_per_trial": 3,
        "epochs": 100,
        "patience": 5
    }'''
    
    c = {
        "max_trials": 5,
        "executions_per_trial": 1,
        "epochs": 10,
        "patience": 3
    }

    tuner = RandomSearch(build_cnn_model,
                         objective='val_accuracy',
                         max_trials=c["max_trials"],  # Adjust as necessary
                         executions_per_trial=c["executions_per_trial"],  # Adjust as necessary for reliability
                         directory='../models',
                         project_name='cnn_tuning')

    # Use the 'fit' method of the tuner, passing in the generators
    tuner.search(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        epochs=c["epochs"],
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=c["patience"])]
    )

    return tuner

def get_top_models(train_generator, val_generator, top_n, models_dir='../models'):
    
    best_model, tuner = run_tuner(train_generator, val_generator)

    
    # Ensure the directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Get the top N performing models
    top_models = tuner.get_best_models(num_models=top_n)
    
    for i, model in enumerate(top_models):
        # Define model save path
        model_save_path = os.path.join(models_dir, f'model_{i+1}')
        os.makedirs(model_save_path, exist_ok=True)
        
        # Save model architecture and weights
        model_json = model.to_json()
        with open(os.path.join(model_save_path, 'model_architecture.json'), 'w') as json_file:
            json_file.write(model_json)
        
        # Save weights
        model.save_weights(os.path.join(model_save_path, 'model_weights.h5'))
        
        print(f'Model {i+1} saved to {model_save_path}')
