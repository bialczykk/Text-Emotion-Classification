
import time
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import tensorflow as tf
import cohere
from cohere import ClassifyExample
import gensim.downloader as api
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('loss')

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')

    # Show the plot
    plt.show()

    # Print final loss and accuracy statistics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    print(f"\nFinal Training Loss: {final_train_loss:.4f}")
    print(f"Final Validation Loss: {final_val_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")


def evaluate_model(model, test_dataset):
    # Evaluate the model on the test set
    test_loss, test_accuracy = model.evaluate(test_dataset)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
    return test_loss, test_accuracy


def evaluate_and_log_model(model, model_description, test_dataset, train_time, n_epochs, model_registry=None):
    test_loss, test_accuracy = model.evaluate(test_dataset)

    new_row = pd.DataFrame(columns=['model', 'test_loss', 'test_accuracy', 'training_time', 'n_epochs', 'avg_epoch_time'],
                           data=[[model_description, test_loss, test_accuracy, train_time, n_epochs, train_time/n_epochs]])

    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    if model_registry is None:
        print('*** INITIALISING RESULTS TABLE ***')
        model_registry = pd.DataFrame(columns=['model', 'test_loss', 'test_accuracy', 'training_time', 'n_epochs', 'avg_epoch_time'])

    model_registry = pd.concat([model_registry, new_row]).reset_index(drop=True)

    model_registry.to_csv('model_results_table.csv')

    return model_registry


def create_rnn_model(vectorize_layer, embedding_dim, rnn_units, num_rnn_layers=1, num_classes=6, seed=42, dropout_rate=0.3, l2_lambda=0.001):
    """
    Create an RNN model with a configurable number of RNN layers, using an external TextVectorization layer,
    with regularization techniques to prevent overfitting.

    Parameters:
    - vectorize_layer (tf.keras.layers.Layer): Pre-configured TextVectorization layer.
    - embedding_dim (int): Dimension of the embedding layer.
    - rnn_units (int): Number of units in each RNN layer.
    - num_rnn_layers (int): Number of RNN layers in the model.
    - num_classes (int): Number of classes to predict.
    - seed (int): Random seed for reproducibility.
    - dropout_rate (float): Dropout rate for Dropout layers.
    - l2_lambda (float): L2 regularization parameter.

    Returns:
    - A compiled TensorFlow model.
    """
    # Input layer for raw text inputs
    inputs = tf.keras.Input(name='text', shape=(1,), dtype=tf.string)

    # Apply the external vectorization layer to inputs
    x = vectorize_layer(inputs)

    # Embedding layer with GlorotUniform initializer
    x = tf.keras.layers.Embedding(
        input_dim=vectorize_layer.vocabulary_size(),
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        embeddings_regularizer=tf.keras.regularizers.l2(l2_lambda),
        mask_zero=True
    )(x)

    # Add RNN layers with Dropout and L2 regularization
    for i in range(num_rnn_layers):
        return_sequences = (i < num_rnn_layers - 1)  # only return sequences for all but the last RNN layer
        x = tf.keras.layers.SimpleRNN(
            rnn_units,
            return_sequences=return_sequences,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
            recurrent_regularizer=tf.keras.regularizers.l2(l2_lambda)
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, seed=seed)(x)  # Add Dropout layer

    # Output layer with L2 regularization
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def create_lstm_model(vectorize_layer, embedding_dim, lstm_units, num_lstm_layers=1, num_classes=6, seed=42, dropout_rate=0.5, l2_lambda=0.01):
    """
    Create an LSTM model with a configurable number of LSTM layers, using an external TextVectorization layer,
    with regularization techniques to prevent overfitting.

    Parameters:
    - vectorize_layer (tf.keras.layers.Layer): Pre-configured TextVectorization layer.
    - embedding_dim (int): Dimension of the embedding layer.
    - lstm_units (int): Number of units in each LSTM layer.
    - num_lstm_layers (int): Number of LSTM layers in the model.
    - num_classes (int): Number of classes to predict.
    - seed (int): Random seed for reproducibility.
    - dropout_rate (float): Dropout rate for Dropout layers.
    - l2_lambda (float): L2 regularization parameter.

    Returns:
    - A compiled TensorFlow model.
    """
    # Input layer for raw text inputs
    inputs = tf.keras.Input(name='text', shape=(1,), dtype=tf.string)

    # Apply the external vectorization layer to inputs
    x = vectorize_layer(inputs)

    # Embedding layer with GlorotUniform initializer and L2 regularization
    x = tf.keras.layers.Embedding(
        input_dim=vectorize_layer.vocabulary_size(),
        output_dim=embedding_dim,
        embeddings_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        embeddings_regularizer=tf.keras.regularizers.l2(l2_lambda),
        mask_zero=True
    )(x)

    # Add LSTM layers with Dropout and L2 regularization
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)  # only return sequences for all but the last LSTM layer
        x = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
            recurrent_regularizer=tf.keras.regularizers.l2(l2_lambda)
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, seed=seed)(x)  # Add Dropout layer

    # Output layer with L2 regularization
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def create_lstm_model_with_embeddings(vectorize_layer, embedding_dim, lstm_units, embeddings_path, num_lstm_layers=1, num_classes=6, seed=42, dropout_rate=0.5, l2_lambda=0.01):
    """
    Create an LSTM model with a configurable number of LSTM layers, using an external TextVectorization layer
    and pre-trained embeddings from a Gensim model, with a fixed random seed for reproducibility.

    Parameters:
    - vectorize_layer (tf.keras.layers.Layer): Pre-configured TextVectorization layer.
    - embedding_dim (int): Dimension of the embedding layer.
    - lstm_units (int): Number of units in each LSTM layer.
    - num_lstm_layers (int): Number of LSTM layers in the model.
    - num_classes (int): Number of classes to predict.
    - embeddings_path (str): Name of the pre-trained embeddings model from Gensim.
    - seed (int): Random seed for reproducibility.
    - dropout_rate (float): Dropout rate for Dropout layers.
    - l2_lambda (float): L2 regularization parameter.

    Returns:
    - A compiled TensorFlow model.
    """

    # Input layer for raw text inputs
    inputs = tf.keras.Input(name='text', shape=(1,), dtype=tf.string)

    # Apply the external vectorization layer to inputs
    x = vectorize_layer(inputs)

    # Load the pre-trained embeddings from Gensim using downloader
    word_vectors = api.load(embeddings_path)
    vocab_size = vectorize_layer.vocabulary_size()

    # Create an embedding matrix
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    word_index = {word: idx for idx, word in enumerate(vectorize_layer.get_vocabulary())}

    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        try:
            embedding_vector = word_vectors[word]
            embedding_matrix[i] = embedding_vector
        except KeyError:
            # Words not found in embedding index will be all zeros.
            pass

    # Embedding layer with pre-trained embeddings and L2 regularization
    x = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix],
        trainable=False,
        mask_zero=True
    )(x)

    # Add LSTM layers with Dropout and L2 regularization
    for i in range(num_lstm_layers):
        return_sequences = (i < num_lstm_layers - 1)  # only return sequences for all but the last LSTM layer
        x = tf.keras.layers.LSTM(
            lstm_units,
            return_sequences=return_sequences,
            kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
            recurrent_initializer=tf.keras.initializers.Orthogonal(seed=seed),
            kernel_regularizer=tf.keras.regularizers.l2(l2_lambda),
            recurrent_regularizer=tf.keras.regularizers.l2(l2_lambda)
        )(x)
        x = tf.keras.layers.Dropout(dropout_rate, seed=seed)(x)  # Add Dropout layer

    # Output layer with L2 regularization
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed),
        kernel_regularizer=tf.keras.regularizers.l2(l2_lambda)
    )(x)

    # Create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model


def classify_emotions(api_key, train_data, test_data, batch_size=50, num_examples_per_class=100, display_samples=False, model='large'):
    

    # Define the emotion labels
    emotion_labels = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}
    label_to_text = {v: k for k, v in emotion_labels.items()}  # Reverse mapping for predictions

    # Select random examples from the train subset for each emotion
    examples = []
    for label_id, label_name in emotion_labels.items():
        # Filter the dataset for the current label
        label_data = train_data.filter(lambda x: x['label'] == label_id)
        # Randomly select examples
        examples_raw = random.choices(label_data, k=num_examples_per_class)
        # Add the examples to the examples list
        for e in examples_raw:
            examples.append(ClassifyExample(text=e['text'], label=label_name))

    # Instantiate the cohere client
    co = cohere.Client(api_key)

    # Prepare the inputs and true labels
    inputs = [sample['text'] for sample in test_data]
    true_labels = [sample['label'] for sample in test_data]

    # Start the timer
    start_time = time.time()

    # Perform the classification in batches to handle large datasets
    predicted_labels = []
    all_classifications = []

    print(f'Generating {len(test_data)} predictions based on {num_examples_per_class * len(emotion_labels)} training examples (few-show learning).')
    print(f'\n\nModel: {model}\n\n')

    for i in range(0, len(inputs), batch_size):
        print(f'Batch {int(round((i+1) / batch_size, 0) +1)} / {int(round((len(inputs) + 1) / batch_size, 0))}')
        batch_inputs = inputs[i:i + batch_size]
        response = co.classify(
            model=model,
            inputs=batch_inputs,
            examples=examples
        )
        # Map predictions back to numeric labels
        batch_predictions = [label_to_text[cls.prediction] for cls in response.classifications]
        predicted_labels.extend(batch_predictions)
        all_classifications.extend(response.classifications)

    # End the timer
    end_time = time.time()
    execution_time = end_time - start_time

    # Calculate accuracy, precision, and recall
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')

    # Print performance metrics
    print('\n\nPerformance Metrics: ')
    print('-' * 20)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print('\n\n')

    if display_samples is False:
        # Print distribution of predicted labels
        pred_label_distribution = pd.Series(predicted_labels).value_counts().sort_index()
        pred_label_distribution.index = [emotion_labels[idx] for idx in pred_label_distribution.index]

        # Compute the number of correct predictions
        correct_predictions = pd.Series(0, index=emotion_labels.values())
        for true, pred in zip(true_labels, predicted_labels):
            if true == pred:
                correct_predictions[emotion_labels[true]] += 1

        # Compute confusion matrix and normalize it row-wise
        cm = confusion_matrix(true_labels, predicted_labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Set up the figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [2, 1]})

        # Plot normalized confusion matrix
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=emotion_labels.values(), yticklabels=emotion_labels.values(), ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('Normalized Confusion Matrix')

        # Plot the stacked bar chart
        bar_width = 0.5
        total_bars = ax2.barh(pred_label_distribution.index[::-1], pred_label_distribution.values[::-1], color='blue', edgecolor='black', height=bar_width, label='Total Predictions')
        correct_bars = ax2.barh(correct_predictions.index[::-1], correct_predictions.values[::-1], color='lightblue', edgecolor='black', height=bar_width, label='Correct Predictions')

        # Add the number inside each bar
        for index, value in enumerate(pred_label_distribution.values[::-1]):
            ax2.text(value, index, str(value), color='black', va='center')
        for index, value in enumerate(correct_predictions.values[::-1]):
            ax2.text(value, index, str(value), color='white', va='center')

        ax2.set_ylabel('Emotion')
        ax2.set_xlabel('Frequency')
        ax2.set_title('Distribution of Predicted Labels')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    else:
        # Print each input sample with its assigned label and confidence
        for idx, classification in enumerate(all_classifications):
            print(f"Input: {inputs[idx]}")
            print(f"Predicted Label: {classification.prediction}")
            print(f"Confidence: {classification.confidence}\n")

    # Print the time of execution
    print(f"\nTime of execution: {execution_time} seconds")