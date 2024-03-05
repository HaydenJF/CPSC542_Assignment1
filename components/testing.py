import os
import json
from keras.models import model_from_json, load_model

def evaluate_top_models(test_generator, top_n, models_dir='../models'):
    results = []
    
    for i in range(1, top_n + 1):
        model_dir = os.path.join(models_dir, f'model_{i}')
        model_architecture_path = os.path.join(model_dir, 'model_architecture.json')
        model_weights_path = os.path.join(model_dir, 'model_weights.h5')
        
        # Load model architecture
        with open(model_architecture_path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json)
        
        # Load model weights
        model.load_weights(model_weights_path)
        
        # Compile model
        model.compile(optimizer='adam',  # Assuming Adam optimizer was used; adjust as necessary
                      loss='categorical_crossentropy',  # Assuming categorical crossentropy; adjust for your use case
                      metrics=['accuracy'])
        
        # Evaluate model on test data
        print(f"Evaluating Model {i}...")
        test_loss, test_accuracy = model.evaluate(test_generator)
        print(f"Model {i}: Loss = {test_loss}, Accuracy = {test_accuracy}")
        
        results.append({
            'model': f'Model {i}',
            'loss': test_loss,
            'accuracy': test_accuracy
        })
    
    return results