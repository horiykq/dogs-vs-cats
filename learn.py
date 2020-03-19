import h5py
import matplotlib.pyplot as plt
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator

DATASET_DIR = './dataset/'
TRAIN_DIR = DATASET_DIR + 'train/'
TEST_DIR = DATASET_DIR + 'test/'

def construct_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32,(3,3),activation="relu",input_shape=(150,150,3)))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128,(3,3),activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(512,activation="relu"))
    model.add(layers.Dense(1,activation="sigmoid"))

    model.summary()

    model.compile(loss="binary_crossentropy",
             optimizer=optimizers.RMSprop(lr=1e-4),
             metrics=["acc"])
    preprocess_imgs(model)


def preprocess_imgs(model):
    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )
    validation_generator = validation_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=(150,150),
        batch_size=20,
        class_mode="binary"
    )
    print(train_generator.class_indices, validation_generator.class_indices)
    learn(model, train_generator, validation_generator)


def learn(model, train_generator, validation_generator):
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=1250,
                                  epochs=10,
                                  validation_data=validation_generator,
                                  validation_steps=1250)
    model.save('./dogorcat.h5')
    show_result(history)


def show_result(history):
    acc = history.history["acc"]
    val_acc = history.history["val_acc"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs = range(1,len(acc) + 1)

    plt.plot(epochs, acc,"bo",label="Training Acc")
    plt.plot(epochs, val_acc,"b",label="Validation Acc")
    plt.legend()

    plt.figure()

    plt.plot(epochs,loss,"bo",label="Training Loss")
    plt.plot(epochs,val_loss,"b",label="Validation Loss")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    construct_model()
