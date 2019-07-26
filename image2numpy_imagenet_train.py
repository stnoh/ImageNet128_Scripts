# http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287

from argparse import ArgumentParser
from utils import *
import os
import imageio
import numpy as np

# Number of classes to be subsampled
num_classes = 1000
# Make sure height and width match the size of the input images
height = width = 128
# Number of pickle files to be created for the training set. Preferably, each file should not be too small and can fit into your memory
n = 100

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images")
    parser.add_argument('-o', '--out_dir', help="Output directory for pickle files")
    args = parser.parse_args()

    return args.in_dir, args.out_dir


# Strong assumption about in_dir and out_dir (They must contain proper data)
def process_folder(in_dir, out_dir):
    label_dict = get_label_dict()
    folders = get_ordered_folders()
    folders = folders[0::1000//num_classes]
    print('Chosen classes: ')
    print([label_dict[folder] for folder in folders])

    print("Processing folder %s" % in_dir)
    x = np.zeros([1281167, height*width*3], dtype=np.uint8)
    row = 0
    labels_list_train = []
    num_images = 0
    for folder in folders:
        label = label_dict[folder]
        print("Processing images from folder %s as class %d" % (folder, label))
        # Get images from this folder
        images = []
        for image_name in os.listdir(os.path.join(in_dir, folder)):
            try:
                img = imageio.imread(os.path.join(in_dir, folder, image_name),pilmode='RGB')
                r = img[:, :, 0].flatten()
                g = img[:, :, 1].flatten()
                b = img[:, :, 2].flatten()
                num_images+=1
                
            except:
                print('Cant process image %s' % image_name)
                with open("log_img2np.txt", "a") as f:
                    f.write("Couldn't read: %s \n" % os.path.join(in_dir, image_name))
                continue
            arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
            x[row] = arr
            row += 1
        samples_num = len(os.listdir(os.path.join(in_dir, folder)))
        labels = [label] * samples_num

        labels_list_train.extend(labels)
        
        print('Label %d: %s has %d samples' % (label, folder, samples_num))

    y = np.array(labels_list_train)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # If you subsample folders [1*] this will not compute mean over all training images
    x_mean = np.mean(x[:num_images], axis=0)

    # Shuffled indices
    train_indices = np.arange(num_images)
    np.random.shuffle(train_indices)
    
    curr_index = 0
    size = num_images // n

    # Create first n-1 files
    y_test = []
    for i in range(1, n):
        d = {
            'data': x[train_indices[curr_index: (curr_index + size)], :],
            'labels': y[train_indices[curr_index: (curr_index + size)]].tolist(),
            'mean': x_mean
        }
        pickle.dump(d, open(os.path.join(out_dir, 'train_data_batch_%d' % i), 'wb'))
        curr_index += size
        y_test.extend(d['labels'])

    # Create last file
    d = {
        'data': x[train_indices[curr_index:], :],
        'labels': y[train_indices[curr_index:]].tolist(),
        'mean': x_mean
    }
    pickle.dump(d, open(os.path.join(out_dir, 'train_data_batch_%d' % n), 'wb'))

    y_test.extend(d['labels'])

    count = np.zeros([1000])

    for i in y_test:
        count[i-1] += 1

    for i in range(1000):
        print('%d : %d' % (i, count[i]))
    print('A total of %d images' % num_images)
    print('A total of %d labels' % len(y_test))

if __name__ == '__main__':
    in_dir, out_dir = parse_arguments()

    print("Start program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir)
    print("Finished.")
