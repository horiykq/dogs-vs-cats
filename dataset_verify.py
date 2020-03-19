import glob
import shutil


def main():
    data_dir = './dataset/'
    train_dogs = glob.glob(data_dir + 'train/dogs/dog.*.jpg')
    train_cats = glob.glob(data_dir + 'train/cats/cat.*.jpg')
    test_imgs = glob.glob(data_dir + 'test/*.jpg')
    print('train: {}dogs, {}cats    test: {}imgs'.format(len(train_dogs),len(train_cats),len(test_imgs)))


if __name__ == "__main__":
    main()
