import os
import pandas as pd
import numpy as np
import json
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, MaxPooling1D, Activation, 
    Flatten, Dropout, GlobalAveragePooling1D, 
    Reshape, Multiply, Attention, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from keras.models import load_model                                  

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
import pdb
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


cur_dir = os.getcwd()

data_dir = r'C:\Cloud\MachineLearningCVE'
figure_dir = data_dir + r'\\Images\\'

BATCH_SIZE = 1024
EPOCHS = 50

processed_x_name = 'X_processed.npy'
pca_x_name = 'X_pca.npy'
processed_y = 'y_processed.csv'
n_components = 32
nrows = 200000000000





def laod_data():

    files = os.listdir(data_dir)
    files = [file for file in files if file.endswith('.csv')]
    li = []
    for filename in files:
        #file_path = os.join(data_dir, filename)
        file_path = data_dir + r'\\' + filename
        df = pd.read_csv(file_path, encoding='cp1252', index_col=None, header=0, low_memory=False, nrows=nrows)
        li.append(df)
        print("Read Completed for ", filename)
       
    
    df = pd.concat(li, axis=0, ignore_index=True)
    print('Dataset shape', df.shape)

    return df



   

    return df

def save_label_encode_mappings(label_encoder):
    
    print('Saving endoing mappings')
    processed_data_dir = data_dir + r'\\Processed_Data\\'
    file_path = processed_data_dir + 'label_encoder_mapping.json'
    label_mappings = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    
    label_mappings = {key: int(value) for key, value in label_mappings.items()}

    with open(file_path, 'w') as json_file:
        json.dump(label_mappings, json_file)

def clean_data_files(df):
    print('Cleaning started')
    
    processed_data_dir = data_dir + r'\\Processed_Data\\'
    os.makedirs(processed_data_dir, exist_ok=True)

    df = df.rename(columns={' Label': 'Label'})

    # Column tranformation
    print('*****************************Column Trabsformation*****************************')
    data_clean = df.dropna().reset_index()
    print('dropping duplicates')
    data_clean.drop_duplicates(keep='first', inplace = True)
    print('datset shape: ' , data_clean.shape)
    print('Label count values')
    print(data_clean['Label'].value_counts())

    print('encoding labels')
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(data_clean['Label'])
    save_label_encode_mappings(label_encoder)
    data_clean['Label'] = y

    print('dropping infinity values')
    data_clean = data_clean.to_numpy(dtype='float32')
    data_clean = data_clean[~np.isinf(data_clean).any(axis=1)]
    print('dataset shape: ', data_clean.shape)

    y = data_clean[:,79]
    X = data_clean[:,:79]
    print('X: {}, y {}'.format(X.shape, y.shape))
    del data_clean
    
    # Row Transformation
    print('*****************************Row Trabsformation*****************************')

    sclaer = MinMaxScaler()
    X = sclaer.fit_transform(X)
    np.save(processed_data_dir + processed_x_name, X)
    print('X Shape preprocessed', X.shape)
    

    pca = PCA(n_components=n_components)
    X = pca.fit_transform(X)
    print('X Shape PCA ', X.shape)

    y = pd.DataFrame(y)
    np.save(processed_data_dir + pca_x_name, X)
    y.to_csv(processed_data_dir + processed_y)

    return X, y

def load_processed_data():

    print('Loading Processed data')
    processed_data_dir = data_dir + r'\\Processed_Data\\'
    files = os.listdir(processed_data_dir)
    print('Loading Processed data from ', processed_data_dir)
    
    for file in files:
        file_path = processed_data_dir + file
        print('Loadinf file: ', file)
       
        if 'pca' in file:
            X_pca = np.load(file_path)
        elif 'X_processed' in file:
            X_processed = np.load(file_path)
        elif 'mapping' in file:
            with open(file_path, 'r') as json_file:
                mappings = json_file.read()
                mappings = json.loads(mappings)
                mappings = {value: key for key, value in mappings.items()}
        else:
            y = pd.read_csv(file_path, encoding='cp1252', index_col=None)
            y = y['0'].values
    print('Loading processed data complete')
    return (X_pca, X_processed, y, mappings)

def conver_files_to_csv():

    X_pca, X_processed, label, mappings =  load_processed_data()
    x = X_pca[:, 0]
    y = X_pca[:, 1]
    z = X_pca[:, 2]

    data = {'c1':x, 'c2':y, 'c3':z, 'label':label['0'].values}

    data = pd.DataFrame(data)
    data.to_csv(figure_dir +'c1c2c3.csv')

def autoencoder_model():

   # Autoencoder with Attention Mechanism
    input_dim_autoencoder = 79
    encoding_dim_autoencoder = 32  # Adjust this based on your preference
    input_layer_autoencoder = Input(shape=(input_dim_autoencoder,))

    encoded_autoencoder = Dense(96, activation='relu')(input_layer_autoencoder)
    encoded_autoencoder = BatchNormalization()(encoded_autoencoder)
    encoded_autoencoder = Dense(encoding_dim_autoencoder, activation='relu', name='encoding_dimentions')(encoded_autoencoder)

    attention_weights = Dense(encoding_dim_autoencoder, activation='softmax')(encoded_autoencoder)
    attended_encoding = Multiply()([encoded_autoencoder, attention_weights])

    decoded_autoencoder = Dense(input_dim_autoencoder, activation='tanh')(attended_encoding)

    autoencoder = Model(inputs=input_layer_autoencoder, outputs=decoded_autoencoder)      

    learning_rate_autoencoder = 0.001
    opt_autoencoder = Adam(learning_rate=learning_rate_autoencoder)

    autoencoder.compile(optimizer=opt_autoencoder, loss='mean_absolute_error')


    return autoencoder

def train_auto_encoder(X_processed):

    
    model_path = data_dir + r'\\models\\'
    model_name_with_path = model_path + 'autoencoder.h5'
    os.makedirs(model_path, exist_ok=True)


    autoencoder = autoencoder_model()
    early_stopping_autoencoder = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

   
    autoencoder.fit(
        X_processed, X_processed,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        shuffle=True, validation_split=0.2,
        callbacks=[early_stopping_autoencoder]
    )
    print(model_name_with_path)
    #pdb.set_trace()

    autoencoder.save(model_name_with_path)
    print('Training Completed')

def get_autoencoder_dimentions(X):

    model_path = data_dir + '\\models\\autoencoder.h5'
    print('Loading trained model:  ', model_path)
    autoencoder_embeddings_filename = 'autoencoder.npy'
    autoencoder_dir = data_dir + r'\\Processed_Data\\'

    filename = autoencoder_dir + autoencoder_embeddings_filename
    print('Converting X to autoencoder embeddings')
    model = load_model(model_path)
    print(model.summary())
    encoder = Model(inputs=model.input, outputs=model.get_layer('encoding_dimentions').output)


    X_encoded = encoder.predict(X)
    np.save(filename, X_encoded)
    print(X_encoded.shape)
    print('Autoencoder embeddings done')
    print('embeddings saved at: ', filename)
    return X_encoded


def plot_3d_scatter(X, labels, title, mapping, axis_labels, fig_name):
     
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
   
    print('Plotting {} in {} directory '.format(fig_name, figure_dir))
    colors = [
        '#1f78b4', '#33a02c', '#e31a1c', '#ff7f00', '#6a3d9a',
        '#a6cee3', '#b2df8a', '#fb9a99', '#fdbf6f', '#cab2d6',
        '#b5b5b5', '#ffed6f', '#c2eabd', '#fccde5', '#d9d9d9'
    ]
    cmap = ListedColormap(colors)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot with color-coded labels
    scatter = ax.scatter(x, y, z, c=labels, cmap='viridis')

    # Customize the plot
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_zlabel(axis_labels[2])
    ax.set_title(title)

    ax.margins(0, 0, 0)

    # Add a colorbar
    colorbar = fig.colorbar(scatter, ticks=sorted(set(labels)))
    colorbar.set_label('Labels')

    # Show the plot
    os.makedirs(figure_dir, exist_ok=True)

    plt.savefig(figure_dir+fig_name, dpi=600)

def autoencoder_embeddings_plot():
    
    _, X_processed, y, mappings = load_processed_data()
    X_embedded = get_autoencoder_dimentions(X_processed[:100])
    
    title = 'Autoencoder Embedding Representations'
    axis_labels = ['Embedding 0', 'embedding 1', 'embedding 2']
    fig_name = 'autencoder_embedding.png'
    plot_3d_scatter(
        X=X_embedded, 
        labels=y[:100], 
        title=title,
        mapping= mappings, 
        axis_labels=axis_labels, 
        fig_name=fig_name
    )

def pca_components_plot():
    X_pca, _, y, mappings = load_processed_data()
    title = 'PCA components plot'
    axis_labels = ['PC1', 'PC2', 'PC3']
    figure_name = 'pca_components.png'
    plot_3d_scatter(
        X = X_pca[:300],
        labels=y[:300],
        mapping=mappings,
        title=title,
        axis_labels=axis_labels,
        fig_name=figure_name
    )
    

pca_x, x, _,_ = load_processed_data()

get_autoencoder_dimentions(x)