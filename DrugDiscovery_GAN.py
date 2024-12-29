import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rdkit import Chem
from rdkit.Chem import Draw
#Sample dataset from ChatGPT! -- to only test
smiles_data = ['CCO', 'CCCO', 'CCCCO', 'CCN', 'CCCl', 'C1CCCCC1', 'C1CCOCC1']

def smiles_molecule(smile):
    mol = Chem.MolFromSmiles(smile)
    return mol

molecule = [smiles_molecule(smile) for smile in smiles_data]

Draw.MolsToGridImage(molecule, molsPerRow=3)

#GAN Input
def encode(smile_list):
    le = LabelEncoder()
    le.fit([char for smile in smile_list for char in smile])
    encoded = [le.transform(list(smiles)) for smiles in smile_list]
    return encoded

from tensorflow.keras.preprocessing.sequence import pad_sequences
encoded  = encode(smiles_data)

length = 10
padding = pad_sequences(encoded, maxlen = length, padding="post", truncating="post" )

print(padding)

#Architecture
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.models import Sequential
def generator():
    model = Sequential()
    model.add(Dense(128, input_dim=100, activation="relu"))
    model.add(Dense(256, activation="relu"))
    model.add(Dense(length*len(set([char for smile in smiles_data for char in smile])), activation="sigmoid"))
    model.add(Reshape((length, len(set([char for smile in smiles_data for char in smile])))))
    return model

def discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(length, len(set([char for smile in smiles_data for char in smile])))))
    model.add(Dense(512, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

gen = generator()
disc = discriminator()

disc.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

#Training
def gan_model(gen, disc):
    disc.trainable = False
    model = Sequential()
    model.add(gen)
    model.add(disc)
    model.compile(loss="binary_crossentropy", metrics=["accuracy"])
    return model

gan =gan_model(gen, disc)
disc = discriminator()
gen = generator()
disc.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])
def train(gen, disc, epochs=1000, batch_size=64):
    for epoch in range(epochs):
        noise = np.random.normal(0,1,(batch_size, 100))
        generated = gen.predict(noise)

        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size,1))

        idx = np.random.randint(0, padding.shape[0], batch_size)
        real_smiles = padding[idx]

        d_loss_real = disc.train_on_batch(real_smiles, real)
        d_loss_fake = disc.train_on_batch(generated, fake)
        d_loss = 0.5*np.add(d_loss_real,d_loss_fake)

        noise = np.random.normal(0,1,(batch_size,100))
        gan_loss = gan.train_on_batch(noise, real)
    
        if epoch%100==0:
            print(f"{epoch}[D loss: {d_loss[0]} & D Accuracy: {100* d_loss[1]}][G loss: {gan_loss}]")

            train(gen,disc, gan)
         
