

# Primero instalar openCV package para importar cv2

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt


# Para descargar los datasets
import download

from random import shuffle

from keras.applications import VGG16

from keras import backend as K
from keras.models import Model, Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense, Activation
import sys


import h5py


def print_progress(count, max_count):
    # Percentage completion.
    pct_complete = count / max_count

    # Status-message. Note the \r which means the line should
    # overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()
    



# Directorio donde vamos a poner todos los videos
in_dir = "data"

# Tamanyo de cada imagen    
img_size = 224

img_size_touple = (img_size, img_size)


# Donde se van a almacenar todas las imagene
#images = []

# Numero de canales
num_channels = 3

# Tamanyo imagen cuando se aplana en vector 1 dimension
img_size_flat = img_size * img_size * num_channels

# Numero de clases
num_classes = 2

# Numero de videos para entreno
_num_files_train = 1

# Numero de frames por video
_images_per_file = 20

# Numero de imagenes total en el training-set
_num_images_train = _num_files_train * _images_per_file

# Extension de video
video_exts = ".avi"

# Url de descarga directa
url_hockey = "http://visilab.etsii.uclm.es/personas/oscar/FightDetection/HockeyFights.zip"

url_movies = "http://visilab.etsii.uclm.es/personas/oscar/FightDetection/Peliculas.rar"

in_dir = "data"


# Funcion para descargar los datos
def download_data(in_dir, url):
    
    # Si la carpeta no existe la creamos
    if not os.path.exists(in_dir):
        os.makedirs(in_dir)
    
    # Para descargar del link directo y extraer los archivos
    download.maybe_download_and_extract(url,in_dir)
    
    
#def label_vid(vid_name):
#    
#    word_label = 
#    
download_data(in_dir,url_hockey)
    
    
    
def get_frames(current_dir, file_name):
    
    in_file = os.path.join(current_dir, file_name)
    
    images = []
    
    vidcap = cv2.VideoCapture(in_file)
    
    success,image = vidcap.read()
        
    count = 0

    while count<_images_per_file:
                
        RGB_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
        res = cv2.resize(RGB_img, dsize=(img_size, img_size),
                                 interpolation=cv2.INTER_CUBIC)
    
        # Convertir imagen en un vector y añadirlo
        #images.append(res.flatten())
        
        images.append(res)
    
        success,image = vidcap.read()
    
        count += 1
        
    resul = np.array(images)
    
    # Mirar esto alomejor no va despues
    
    resul = (resul / 255.).astype(np.float16)
        
    return resul

    


image_model = VGG16(include_top=True, weights='imagenet')


image_model.summary()


# We will use the output of the layer prior to the final
# classification-layer which is named fc2. This is a fully-connected (or dense) layer.
transfer_layer = image_model.get_layer('fc2')

image_model_transfer = Model(inputs=image_model.input,
                             outputs=transfer_layer.output)

transfer_values_size = K.int_shape(transfer_layer.output)[1]


print("La entrada de la red dimensiones:",K.int_shape(image_model.input)[1:3])

print("La salida de la red dimensiones: ", transfer_values_size)


def get_transfer_values(current_dir, file_name):
    
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    image_batch = np.zeros(shape=shape, dtype=np.float16)
    
    image_batch = get_frames(current_dir, file_name)
    
    # Arreglar esto para obtener los valores de los filtros despues de pooling
    
    
    # Pre-allocate output-array for transfer-values.
    # Note that we use 16-bit floating-points to save memory.
    shape = (_images_per_file, transfer_values_size)
    transfer_values = np.zeros(shape=shape, dtype=np.float16)

    transfer_values = \
            image_model_transfer.predict(image_batch)
            
    return transfer_values

in_dir_prueba = 'data'

def proces_transfer(vid_names, in_dir, labels):
    
    
    count = 0
    
    tam = len(vid_names)
    
    # Pre-allocate input-batch-array for images.
    shape = (_images_per_file,) + img_size_touple + (3,)
    
    while count<tam:
        
        video_name = vid_names[count]
        
        image_batch = np.zeros(shape=shape, dtype=np.float16)
    
        image_batch = get_frames(in_dir, video_name)
        
         # Note that we use 16-bit floating-points to save memory.
        shape = (_images_per_file, transfer_values_size)
        transfer_values = np.zeros(shape=shape, dtype=np.float16)
        
        transfer_values = \
            image_model_transfer.predict(image_batch)
         
        labels1 = labels[count]
        
        aux = np.ones([20,2])
        
        labelss = labels1*aux
        
        yield transfer_values, labelss
        
        count+=1
    

def make_files(n_files):
    
    gen = proces_transfer(names_training, in_dir_prueba, labels_training)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('prueba.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            
            
            numer += 1
            
def make_files_validation(n_files):
    
    gen = proces_transfer(names_validation, in_dir_prueba, labels_validation)

    numer = 1

    # Read the first chunk to get the column dtypes
    chunk = next(gen)

    row_count = chunk[0].shape[0]
    row_count2 = chunk[1].shape[0]
    
    with h5py.File('pruebavalidation.h5', 'w') as f:
    
        # Initialize a resizable dataset to hold the output
        maxshape = (None,) + chunk[0].shape[1:]
        maxshape2 = (None,) + chunk[1].shape[1:]
    
    
        dset = f.create_dataset('data', shape=chunk[0].shape, maxshape=maxshape,
                                chunks=chunk[0].shape, dtype=chunk[0].dtype)
    
        dset2 = f.create_dataset('labels', shape=chunk[1].shape, maxshape=maxshape2,
                                 chunks=chunk[1].shape, dtype=chunk[1].dtype)
    
         # Write the first chunk of rows
        dset[:] = chunk[0]
        dset2[:] = chunk[1]

        for chunk in gen:
            
            if numer == n_files:
            
                break

            # Resize the dataset to accommodate the next chunk of rows
            dset.resize(row_count + chunk[0].shape[0], axis=0)
            dset2.resize(row_count2 + chunk[1].shape[0], axis=0)

            # Write the next chunk
            dset[row_count:] = chunk[0]
            dset2[row_count:] = chunk[1]

            # Increment the row count
            row_count += chunk[0].shape[0]
            row_count2 += chunk[1].shape[0]
            
            print_progress(numer, n_files)
        
            
            
            numer += 1
  

#valores = get_transfer_values(in_dir, file_name)

#transfer_values = cache.cache(cache_path='datos_cache.pkl',
#                        fn=get_transfer_values,
#                        current_dir=in_dir,
#                        file_name="no381_xvid.avi")


def label_video_names(in_dir):
    
    names = []
    labels = []
      
    for current_dir, dir_names,file_names in os.walk(in_dir):
        
        for file_name in file_names:
            
            if file_name[0:2] == 'fi':
                labels.append([1,0])
                names.append(file_name)
            elif file_name[0:2] == 'no':
                labels.append([0,1])
                names.append(file_name)
                     
            
    c = list(zip(names,labels))
    
    shuffle(c)
    
    names, labels = zip(*c)
            
    return names, labels




names, labels = label_video_names(in_dir_prueba)


training_set = int(len(names)*0.8)
validation_set = int(len(names)*0.2)

names_training = names[0:training_set]
names_validation = names[training_set:]

labels_training = labels[0:training_set]
labels_validation = labels[training_set:]


make_files(training_set)
make_files_validation(validation_set)




def process_alldata_training():
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File('prueba.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target

def process_alldata_validation():
    
    joint_transfer=[]
    frames_num=20
    count = 0
    
    with h5py.File('pruebavalidation.h5', 'r') as f:
            
        X_batch = f['data'][:]
        y_batch = f['labels'][:]

    for i in range(int(len(X_batch)/frames_num)):
        inc = count+frames_num
        joint_transfer.append([X_batch[count:inc],y_batch[count]])
        count =inc
        
    data =[]
    target=[]
    
    for i in joint_transfer:
        data.append(i[0])
        target.append(np.array(i[1]))
        
    return data, target
    
    


chunk_size = 4096
n_chunks = 20
rnn_size = 512

model = Sequential()
model.add(LSTM(rnn_size, input_shape=(n_chunks, chunk_size)))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('sigmoid'))
model.add(Dense(2))
model.add(Activation('softmax'))
model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])


epoch = 20
batchS = 100

data, target = process_alldata_training()

data_val, target_val = process_alldata_validation()
    
#model.fit(np.array(data), np.array(labels[0:8]), epochs=epoch, batch_size=batchS, verbose=2)
    
model.fit(np.array(data), np.array(target), epochs=epoch, batch_size=batchS, verbose=2)

#model.fit(data, target, epochs=epoch, batch_size=batchS, verbose=2)


result = model.evaluate(np.array(data_val), np.array(target_val))

for name, value in zip(model.metrics_names, result):
    print(name, value)


    
    
    
    




