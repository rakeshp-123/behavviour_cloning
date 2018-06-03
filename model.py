import csv
import numpy as np
import cv2
import ntpath

# Modified nvidia model with prepocessing and dropout to tackle overfitting
def mymodel(nb_classes_val, keep_prob):
    model = Sequential()
    # Preprocessing: Normalization and cropping
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    # model acchitecture
    model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(keep_prob))
    model.add(Flatten()) 
    model.add(Dense(100))
    #model.add(Dropout(keep_prob))
    model.add(Dense(50)) 
    model.add(Dense(10))
    #model.add(Dropout(keep_prob))
    model.add(Dense(nb_classes_val))
    return model

def loadimages():
    
    lines = []
    with open('driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    features = []
    labels = []

    for line in lines:
    # loop for center,left and right camera images
        for i in range(3):                         
            filename = ntpath.basename(line[i])
            current_p = 'IMG/' + filename
            #print(current_p)
            im = cv2.imread(current_p)
            # convert from BGR to RGB
            im_rgb = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
            features.append(im_rgb)
            angle = float(line[3])
            # Add and subtract extra angle(i.e 0.2 in this case) for left and
            # right images angle value respectively
            if i == 1:
                angle += 0.2 # left image angle
            elif i == 2:
                angle -= 0.2 # right image angle
            else:
                angle = angle # center image angle
            labels.append(angle)
            #height, width, channels = im.shape
    return(features, labels)	
  
def generator(samples, batch_size = 32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        samples = sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            augmented_images = []
            augmented_measurement = []
    
            for image, measurement in batch_samples:
                augmented_images.append(image)
                augmented_measurement.append(measurement)
                # augment image
                augmented_images.append(cv2.flip(image,1)) 
                augmented_measurement.append(measurement * -1.0)
	
            X_train_t = np.array(augmented_images)
            y_train_t = np.array(augmented_measurement)
            #print(y_train_t.shape)
            #print(X_train_t.shape)
            yield sklearn.utils.shuffle(X_train_t, y_train_t)

            #X_train = np.sum(X_train/3, axis=3, keepdims=True)


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split

X_train_total, y_train_total = loadimages()
X_train, X_valid, y_train, y_valid = train_test_split(X_train_total, y_train_total, test_size = 0.25, random_state = 0)

train_samples = list(zip(X_train, y_train))
validation_samples = list(zip(X_valid, y_valid))

print('total train samples: {}'.format(len(train_samples)))
print('total validation samples: {}'.format(len(validation_samples)))

train_generator = generator(train_samples, 128)
validation_generator = generator(validation_samples, 128)

nb_classes = 1
keep_prob = 0.5
model = mymodel(nb_classes, keep_prob)
model.compile(optimizer='adam', loss='mse')
#model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch=7)
final_data = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), 
                    nb_epoch=7)
model.save('model.h5')
#plt.plot(final_data.history['loss'])
#plt.plot(final_data.history['val_loss'])
#plt.ylabel('loss')
#plt.xlabel('epoch')
#plt.title('loss graph')
#plt.legend(['training data', 'validation data'], loc='upper right')
#plt.show()
