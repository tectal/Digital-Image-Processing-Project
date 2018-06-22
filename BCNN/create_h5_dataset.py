from tflearn.data_utils import build_hdf5_image_dataset
import h5py

trainset = "train"
testset = "test"
build_hdf5_image_dataset(testset, image_shape=(224, 224),
                         mode='folder', output_path='new_test.h5',
                         categorical_labels=True, normalize=False)

build_hdf5_image_dataset(testset, image_shape=(224, 224),
                         mode='folder', output_path='new_val.h5',
                         categorical_labels=True, normalize=False)

print('Done creating new_test.h5')
build_hdf5_image_dataset(trainset, image_shape=(224, 224),
                         mode='folder', output_path='new_train.h5',
                         categorical_labels=True, normalize=False)
print ('Done creating new_train.h5')

