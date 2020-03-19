import glob
import shutil


def main():
    data_dir = '../CNNelements/dogsvscats/'
    train_imgs(data_dir)

def train_imgs(data_dir):
    base_dir = './dataset/'
    target_dogs = glob.glob(data_dir + 'train_imgs/' + 'dog' + '.*.jpg')
    target_cats = glob.glob(data_dir + 'train_imgs/' + 'cat' + '.*.jpg')
    for i in target_dogs:
        shutil.copy(i, base_dir + 'train/dogs/')
    for i in target_cats:
        shutil.copy(i, base_dir + 'train/cats/')
    test_imgs(base_dir, data_dir)


def test_imgs(base_dir, data_dir):
    targets = glob.glob(data_dir + 'test_imgs/' + '*.jpg')
    for i in targets:
        shutil.copy(i, base_dir + 'test/')
    print('completed.')



if __name__ == "__main__":
    main()
