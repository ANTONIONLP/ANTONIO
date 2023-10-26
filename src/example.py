from data import load_data, load_embeddings, load_pca, prepare_data_for_training
from perturbations import create_perturbations
from hyperrectangles import load_hyperrectangles
from train import train_base, train_adversarial
from property_parser import parse_properties
from results import calculate_accuracy, calculate_perturbations_accuracy, calculate_marabou_results, calculate_number_of_sentences_inside_the_verified_hyperrectangles, calculate_cosine_perturbations_filtering
from tensorflow import keras
import os
import nltk
nltk.download('punkt')


def get_model(n_components):
    inputs = keras.Input(shape=(n_components,), name="embeddings")
    x = keras.layers.Dense(128, activation="relu", name="dense_1")(inputs)
    outputs = keras.layers.Dense(2, activation="linear", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    print(model.summary())
    return model


if __name__ == '__main__':
    # Variables
    path = 'datasets'
    dataset_names = ['ruarobot']
    encoding_models = {'all-MiniLM-L6-v2': 'sbert22M'}
    og_perturbation_name = 'original'
    perturbation_names = ['character']
    hyperrectangles_names = {'character': ['character']}

    n_components = 30
    batch_size = 64
    seed = 42
    epochs = 30
    pgd_steps = 5

    load_saved_embeddings = False
    load_saved_align_mat = False
    load_saved_pca = False
    load_saved_perturbations = False
    load_saved_hyperrectangles = False
    from_logits = True

    # Derived variables
    dataset_name = dataset_names[0]
    encoding_model = list(encoding_models.keys())[0]
    encoding_model_name = encoding_models[encoding_model]
    perturbation_name = perturbation_names[0]
    hyperrectangles_name = list(hyperrectangles_names.keys())[0]

    # Load the data and embed them
    data_o = load_data(dataset_name, path=path)
    X_train_pos_embedded_o, X_train_neg_embedded_o, X_test_pos_embedded_o, X_test_neg_embedded_o, y_train_pos_o, y_train_neg_o, y_test_pos_o, y_test_neg_o = load_embeddings(dataset_name, encoding_model, encoding_model_name, og_perturbation_name, load_saved_embeddings, load_saved_align_mat, data_o, path)

    # Create pthe erturbations and embed them
    data_p = create_perturbations(dataset_name, perturbation_name, data_o, path)
    X_train_pos_embedded_p, X_train_neg_embedded_p, X_test_pos_embedded_p, X_test_neg_embedded_p, y_train_pos_p, y_train_neg_p, y_test_pos_p, y_test_neg_p = load_embeddings(dataset_name, encoding_model, encoding_model_name, perturbation_name, load_saved_perturbations, load_saved_align_mat, data=data_p, path=path)
    
    # Prepare the data for training
    X_train_pos, X_train_neg, X_test_pos, X_test_neg = load_pca(dataset_name, encoding_model_name, load_saved_pca, X_train_pos_embedded_o, X_train_neg_embedded_o, X_test_pos_embedded_o, X_test_neg_embedded_o, n_components, path=path)
    train_dataset, test_dataset = prepare_data_for_training(X_train_pos, X_train_neg, X_test_pos, X_test_neg, y_train_pos_o, y_train_neg_o, y_test_pos_o, y_test_neg_o, batch_size)

    # Create the hyper-rectangles
    hyperrectangles = load_hyperrectangles(dataset_name, encoding_model_name, hyperrectangles_name, load_saved_hyperrectangles, path=path)

    # Train and save the base and adversarial models
    model_path = f'{path}/{dataset_name}/models/tf/{encoding_model_name}'
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    n_samples = int(len(X_train_pos))

    model = get_model(n_components)
    model = train_base(model, train_dataset, test_dataset, epochs, seed=seed, from_logits=from_logits)
    model.save(f'{model_path}/base_{seed}')

    model = get_model(n_components)
    model = train_adversarial(model, train_dataset, test_dataset, hyperrectangles, epochs, batch_size, n_samples, pgd_steps, seed=seed, from_logits=from_logits)
    model.save(f'{model_path}/{perturbation_name}_{seed}')

    # Parse properties to VNNlib and Marabou formats
    parse_properties(dataset_names, encoding_models, hyperrectangles_names, target='vnnlib', path=path)
    parse_properties(dataset_names, encoding_models, hyperrectangles_names, target='marabou', path=path)

    # Results
    calculate_accuracy(dataset_names, encoding_models, batch_size, path=path)
    calculate_perturbations_accuracy(dataset_names, encoding_models, perturbation_names, batch_size, path=path)
    calculate_cosine_perturbations_filtering(dataset_names, encoding_models, perturbation_names, path=path)
