import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

# Returns the norm of a vector
def norm(v):
  sum = float(0)
  for i in range(len(v)):
    sum += v[i]**2
  return sum**(0.5)

# Loading data from the Excel sheet
def loadData(data_size, training, min_range, max_range, motors, replay_size):
    print("Loading Data...")
    sheets = pd.read_excel('walkingDataClean.xlsx', sheet_name=None)
    sheets_names = list(sheets.keys())
    data = np.zeros((data_size, replay_size, motors))
    labels = np.zeros(data_size)
    rows = sheets[sheets_names[0]].shape[0]

    # Creating the data
    for data_slice in range(data_size):
        # False data
        if random.randint(0, 1) == 1:
            labels[data_slice] = False  # This isn't a step

            for itr in range(replay_size):
                values = np.random.uniform(min_range, max_range, motors)
                values = np.where(values < 0, 360 + values, values) # convert negative angles to positive
                values = np.where(values == 0, 0.001, values)
                data[data_slice, itr] = values

        # True data
        else:
            labels[data_slice] = True
            name = sheets_names[random.randint(0, len(sheets_names) - 1)]
            sheet = sheets[name]
            start = random.randint(0, (rows - replay_size - 1))

            for itr in range(replay_size):
                values = sheet.iloc[(start + itr), 1:].values
                values = np.where(values < 0, 360 + values, values) # convert negative angles to positive
                values = np.where(values == 0, 0.001, values)
                data[data_slice, itr] = values


    # Normalizing the data
    for piece in range(data_size):
        tmp = np.reshape(data[piece], (motors * replay_size))
        tmp_norm = norm(tmp)
        data[piece] = data[piece] / tmp_norm

    # Splitting the data by training amount
    split_index = int(training * data_size)
    train_data = data[:split_index]
    train_labels = labels[:split_index]
    test_data = data[split_index:]
    test_labels = labels[split_index:]

    print("Data Loaded!")

    return [train_data, train_labels, test_data, test_labels]


# This plots the loss and accuracy
def plotHist(loss_hist, accuracy_hist):
    # Plot the training loss values
    plt.plot(loss_hist)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    # Plot the training accuracy values
    plt.plot(accuracy_hist)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


# Model definition
def createModel(motors, replay_size):
    # Input layer with 12x4 input shape
    input_layer = tf.keras.layers.Input(shape=(replay_size, motors))

    # Hidden layers with relu activations
    hidden_layer1 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')(input_layer)
    hidden_layer2 = tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_normal')(hidden_layer1)
    hidden_layer3 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_normal')(hidden_layer2)
    flat_layer = tf.keras.layers.Flatten()(hidden_layer3)

    # Output layer with 1 sigmoid activation output
    output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(flat_layer)

    return tf.keras.models.Model(inputs=input_layer, outputs=output_layer)


# The program that trains and evaluates the model
def trainingLoop():
    data_size = 32000
    custom_Rate = False
    learning_rate = 0.001
    training_per = 0.7
    motors = 12
    replay_size = 4
    min_range = -35
    max_range = 78
    # Load your data into numpy arrays
    train_data, train_labels, test_data, test_labels = loadData(data_size, training_per, max_range, min_range, motors,
                                                                replay_size)

    # Define the model
    model = createModel(motors, replay_size)
    if custom_Rate:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam()

    loss = tf.keras.losses.BinaryCrossentropy()

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Define the number of epochs, batch size, and clip value
    num_epochs = 5
    batch_size = 32
    clip_val = None # Applying no clipping to the gradients

    # For evaluations
    loss_hist = []
    accuracy_hist = []

    # Train the model
    for epochs in range(num_epochs):
        for i in range(len(train_data)):
            with tf.GradientTape() as tape:
                logits = model(tf.constant(train_data[i][tf.newaxis], dtype=tf.float32))
                loss_val = tf.keras.losses.BinaryCrossentropy()(tf.expand_dims(train_labels[i], 0), logits)
            grads = tape.gradient(loss_val, model.trainable_variables)
            if clip_val is not None:
                clipped_grads, _ = tf.clip_by_global_norm(grads, clip_val)
                optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
            else:
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_loss, train_acc = model.evaluate(train_data, train_labels)
        loss_hist.append(train_loss)
        accuracy_hist.append(train_acc)

    # Evaluate the model on the test data
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print("Test accuracy:", test_acc)

    model.save("critic(A "+str(test_acc)+", E "+str(num_epochs)+", B "+str(batch_size)+", LR "+str(learning_rate)+", DS "+str(data_size)+", CV "+str(clip_val)+").h5")

    # Plotting graphs
    plotHist(loss_hist, accuracy_hist)


trainingLoop()
