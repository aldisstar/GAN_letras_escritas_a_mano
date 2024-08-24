#%%
# Librerias
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Input, Conv2D, Dropout, Flatten, Dense, BatchNormalization, Reshape, UpSampling2D, Activation
from tensorflow.keras.optimizers import Adam
from keras.layers import LeakyReLU
from keras.models import Model
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython import display
from skimage import io
from skimage.transform import resize
import os
import glob
from PIL import Image

np.random.seed(0)
#%%

# Cargamos las imágenes de nuestro dataset ya clasificado
base_dir = 'C:/Users/Aldis/Documents/Master Data Science/GitHub/GAN_letras_escritas_a_mano/GAN_data'
image_coll = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

# Datos de entrenamiento
X_train = []

for collection in image_coll:
    print('Cargando letra: ' + collection)
    # Ruta de las imágenes
    image_path = os.path.join(base_dir, collection, '*.png')
    # Obtener la lista de archivos de imágenes
    images = glob.glob(image_path)
    
    for image_file in images:
        with Image.open(image_file) as img:
            img = img.convert('L')  
            img = np.array(img)  
            img = resize(img, (28, 28), anti_aliasing=True)  
            X_train.append(img)

X_train = np.array(X_train) 
print(X_train.shape)
# %%

# Analizo imagenes
index = np.random.choice(X_train.shape[0], 16)
samples = X_train[index, :, :]
plt.figure(figsize=(7, 7))
for i in range(samples.shape[0]):
    plt.subplot(4, 4, i + 1)
    img = samples[i, :, :]  
    plt.imshow(img, interpolation='nearest', cmap='gray_r')
    plt.axis('off')
plt.tight_layout()
plt.show()
# %%

# Normalizar las imagenes
print(np.min(X_train), np.max(X_train))
# Llevar datos al rango [-1.0,1.0]
X_train = (X_train.astype('float32') - 0.5) / 0.5
print(np.min(X_train), np.max(X_train))
print('Shape de X_train:', X_train.shape)
print('Cantidad de muestras:', X_train.shape[0])
# %%

# Red neuronal para el generador GAN
shp = X_train.shape[1:]
inidim = 100
dropout_rate = 0.3
opt = Adam(learning_rate=2e-4,beta_1=0.5)

# Generador
g_input = Input(shape=[inidim])
H = Dense(7*7*128, kernel_initializer='glorot_normal')(g_input)
H = LeakyReLU(0.2)(H)
H = Reshape( [7, 7, 128] )(H)
H = UpSampling2D(size=(2, 2))(H)
H = Conv2D(64, (5, 5), padding='same', kernel_initializer='glorot_uniform')(H)
H = LeakyReLU(0.2)(H)
H = UpSampling2D(size=(2, 2))(H)
H = Conv2D(1, (5, 5), padding='same', kernel_initializer='glorot_uniform')(H)
g_V = Activation('tanh')(H)
generator = Model(g_input,g_V)
generator.compile(loss='binary_crossentropy', optimizer=opt)
generator.summary()
# %%

# Modelo para el discriminador (entra = shp, salen 2 nodos = generado, dataset)
shp = (28, 28, 1) #Agrego 4ta dimención de canal grises
dopt = Adam(learning_rate=2e-4, beta_1=0.5)
d_input = Input(shape=shp)
H = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(d_input)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(H)
H = LeakyReLU(0.2)(H)
H = Dropout(dropout_rate)(H)
H = Flatten()(H)
d_V = Dense(2, activation='sigmoid')(H)

discriminator = Model(d_input, d_V)
discriminator.compile(loss='binary_crossentropy', optimizer=dopt)
discriminator.summary()
# %%

# Juntar ambas partes para generar el GAN
# Congelo los pesos del discriminador para que no se entrenen en principio
discriminator.trainable = False
# Capa input
gan_input = Input(shape=[inidim])
# Agreguemos el generador a continuación de la capa input. Guardemos en la variable 'x'
x = generator(gan_input)
# Agreguemos el discriminador a continuación del generador
gan_V = discriminator(x)

GAN = Model(gan_input, gan_V)

# Compilar el modelo con función de pérdida binary_crossentropy y el optimizador definido
GAN.compile(loss='binary_crossentropy', optimizer=opt)
GAN.summary()
# %%

#  Entrenar el GAN
def plot_loss(losses):
        display.clear_output(wait=True)
        display.display(plt.gcf())
        plt.figure(figsize=(10,8))
        plt.plot(losses["d"], label='Pérdida del discriminador')
        plt.plot(losses["g"], label='Pérdida del generador')
        plt.legend()
        plt.show()
        
def plot_gen(n_ex=16,dim=(4,4), figsize=(7,7) ):
    noise = np.random.normal(0,1,size=[n_ex,inidim])
    generated_images = generator.predict(noise)

    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0],dim[1],i+1)
        img = generated_images[i,:,:,0]
        plt.imshow(img,interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
# %%

# Pre-entreno al discriminador para obtener mejores resultados cuando se entrene junto con el generador
# Número de iteraciones
ntrain = 5000
# Tomo datos al azar para entrenar (agregar 4ta dim de canal gris en dataset original)
trainidx = np.random.choice(X_train.shape[0], ntrain)
XT = X_train[trainidx,:, :]
if XT.ndim == 3:
    XT = np.expand_dims(XT, axis=-1)

#Genero ruido
noise_gen = np.random.normal(0,1,size=[XT.shape[0],inidim])
#Genero imágenes con el generador a partir del ruido
generated_images = generator.predict(noise_gen)
 
#Junto las imágenes generadas con las originales del dataset
X = np.concatenate((XT, generated_images))
n = XT.shape[0]
y = np.zeros([2*n,2])
#Trato de engañar al discriminador diciendo que todas las imágenes son reales
y[:n,1] = 1
y[n:,0] = 1
#Permito cambiar los pesos del discriminador
discriminator.trainable = True
#Pre-entreno
discriminator.fit(X,y, epochs=1, batch_size=32)
# %%

#Predigo
y_hat = discriminator.predict(X)

#Analizo los resultados del discriminador pre-entrenado
y_hat_idx = np.argmax(y_hat,axis=1)
y_idx = np.argmax(y,axis=1)
diff = y_idx-y_hat_idx
n_tot = y.shape[0]
n_rig = (diff==0).sum()
acc = n_rig*100.0/n_tot
print("Precisión: %0.02f imágenes (%d de %d) correctas"%(acc, n_rig, n_tot))
# %%

# Función para ayudar a entrenar el GAN
# Defino un vector de pérdidas del generador y discriminador
losses = {"d":[], "g":[]}
if X_train.ndim == 3:
    X_train = np.expand_dims(X_train, axis=-1)
    
def train_for_n(nb_epoch=5000, plt_frq=25,BATCH_SIZE=32):

    for e in tqdm(range(nb_epoch)):
        
        # Creo imágenes a partir del ruido con el generador
        image_batch = X_train[np.random.choice(X_train.shape[0],size=BATCH_SIZE),:,:,:]
        noise_gen = np.random.normal(0,1,size=[BATCH_SIZE,inidim])
        generated_images = generator.predict(noise_gen)
        
        # Entreno al discriminador con las imágenes generadas
        # Junto imágenes generadas con imágenes del dataset original
        X = np.concatenate((image_batch, generated_images))
        y = np.zeros([2*BATCH_SIZE,2])
        # Trato de engañar al discriminador
        y[0:BATCH_SIZE,1] = 1
        y[BATCH_SIZE:,0] = 1

        # Permito entrenar al discriminador
        discriminator.trainable = True
        # Entreno
        d_loss  = discriminator.train_on_batch(X,y)
        # Guardo las pérdidas del discriminador
        losses["d"].append(d_loss)

        # Congelo los pesos del discriminador para que no se modifiquen en esta parte del entrenamiento del GAN
        discriminator.trainable = False
        for i in range(1):
            noise_tr = np.random.normal(0,1,size=[BATCH_SIZE,inidim])
            y2 = np.zeros([BATCH_SIZE,2])
            y2[:,1] = 1
            # Entrenamiento
            g_loss = GAN.train_on_batch(noise_tr, y2 )

        # Guardo las pérdidas del generador
        losses["g"].append(g_loss)

        # Actualizo las gráficas cada plt_frq iteraciones
        if e%plt_frq==0:
            plot_loss(losses)
            plot_gen()
# %%

# Llamo a la función para entrenar
train_for_n(nb_epoch=1500, plt_frq=25,BATCH_SIZE=128)
# %%

# Crear imagenes nuevas
plot_gen() 
# %%
