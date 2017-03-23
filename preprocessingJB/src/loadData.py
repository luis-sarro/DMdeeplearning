'''
@author: Jesús Barroso Seano
'''

from keras.utils import np_utils
import numpy as np
import os
from random import shuffle
from PIL import Image

def resizeCrop(imageIn, imgSize):
    l = min(imageIn.size)
    p1 = imageIn.crop((0, 0, l, l))
    p2 = imageIn.crop((imageIn.width - l, imageIn.height - l, 
                       imageIn.width, imageIn.height))
    return (p1.resize((imgSize, imgSize), Image.ANTIALIAS),
            p2.resize((imgSize, imgSize), Image.ANTIALIAS))

def resizeIm(imageIn, imgSize):
    """ resizeIm
    Esta función escala la imagen y la convierte en una imagen 
    cuadrada. Para mantener la relación de aspecto del objeto 
    de interés, se genera una imagen cuadrada replicando los
    bordes extremos 
    """

    cols, rows = imageIn.size
    newSize = max((rows, cols))
        
    # Creación de la nueva imagen con la dimensión mayor
    newIm = Image.new("RGB", (newSize, newSize))
    
    if rows < cols:
        
        # Calcular las filas a insertar
        n = cols - rows
        top = round(n / 2)
        bottom = round(n / 2)
        
        if n%2 is not 0:
            top += 1
            
        # Obtener fila superior y fila inferior que serán replicadas
        topRow = imageIn.crop((0, 0, cols, 1))
        bottomRow = imageIn.crop((0, rows - 1, cols, rows))
        
        # Pegar filas superiores
        for i in range(top):
            newIm.paste(topRow, (0, i))
            
        # Pegar filas inferiores
        for i in range(bottom):
            newIm.paste(bottomRow, (0, top + rows -1 + i))
    
        # Pegar imagen original 
        newIm.paste(imageIn, (0, top))
        
    else:
        # Calcular las columnas a insertar
        n = rows - cols
        left = round(n / 2)
        right = round(n / 2)
        
        if n%2 is not 0:
            left += 1
            
        # Obtener columnas derecha e izquierda que serán replicadas
        leftCol = imageIn.crop((0, 0, 1, rows))
        rightCol = imageIn.crop((cols - 1, 0, cols, rows))
        
        # Pegar filas a la izquierda
        for i in range(left):
            newIm.paste(leftCol, (i, 0))
            
        # Pegar filas a la derecha
        for i in range(right):
            newIm.paste(rightCol, (left + cols - 1 + i, 0))
    
        # Pegar imagen original
        newIm.paste(imageIn, (left, 0))
    
    # Devolver imagen reescalada
    return (newIm.resize((imgSize, imgSize), Image.ANTIALIAS),)


def loadPolenData(pathDataTrain, pathOut, resizeType, imgSize, numTest):
    """ Cargar datos de polen
    Esta función lee la base de datos de las imágenes de polen, las reescala
    a imágenes cuadradas replicando bordes, guarda las nuevas imágenes 
    generadas en un directorio y devuelve los datos siguiendo el formato
    de Keras.
    """
    
    listing = os.listdir(pathDataTrain)
    
    # Crear nuevo directorio en caso de que no exista
    if os.path.exists(pathOut) is False:
        os.mkdir(pathOut)
        
    #Ordenar por nombre
    listing.sort()
    
    # Listas auxiliares
    tempIms = []
    tempTestIms = []
    tempLabels = []
    tempTestLabels = []
    
    for idx, iclass in enumerate(listing):
        
        # Iterar sobre las imágenes de cada clase
        files = os.listdir(os.path.join(pathDataTrain, iclass))
        classIms = []

        
        for file in files:
            # Abrir imagen
            im = Image.open(os.path.join(pathDataTrain, iclass, file))
            
            # Reescalar
            if resizeType is "FILLBORDERS":
                img = resizeIm(im, imgSize)
            elif resizeType is "CROP":
                img = resizeCrop(im, imgSize)
            elif resizeType is "RESIZE":
                img = (im.resize((imgSize, imgSize), Image.ANTIALIAS),)
            else:
                img = (im,)
            
            # Creación de directorio si no existiese
            if os.path.exists(os.path.join(pathOut, iclass)) is False:
                os.mkdir(os.path.join(pathOut, iclass))
            
            for nim, image in enumerate(img):
                # Guardar imagen generada          
                image.save(os.path.join(pathOut, iclass, str(nim) + file), "JPEG")
                
                #Almacenar imagen en la lista auxiliar
                classIms.append(np.array(image).astype("float32") / 255.0)
            
        # Combinar las imánges de la clase de manera aleatoria
        shuffle(classIms)
            
        classLabels = [idx] * len(classIms)
        
        # Reservar las imágenes de test
        tempTestIms += classIms[-numTest:]
        tempIms += classIms[:-numTest]
        tempTestLabels += classLabels[-numTest:]
        tempLabels += classLabels[:-numTest]
            
    # Generar salida en formato Keras
    trainX = np.array(tempIms)
    trainY = np_utils.to_categorical(np.array(tempLabels), len(listing))
    testX = np.array(tempTestIms)
    testY = np_utils.to_categorical(np.array(tempTestLabels), len(listing))
    
    return (trainX, trainY), (testX, testY)