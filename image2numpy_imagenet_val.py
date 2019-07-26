# http://stackoverflow.com/questions/35032675/how-to-create-dataset-similar-to-cifar-10/35034287

from argparse import ArgumentParser
import numpy as np
import os
import imageio
from utils import *

# Number of classes to be subsampled
num_classes = 1000
# Number of pickle files to be created for the training set. Preferably, each file should not be too small and can fit into your memory
n = 10

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--in_dir', help="Input directory with source images")
    parser.add_argument('-o', '--out_dir', help="Output directory for pickle files")
    args = parser.parse_args()

    return args.in_dir, args.out_dir


def process_folder(in_dir, out_dir):
    label_dict = get_label_dict()
    folders = get_ordered_folders()
    val_ground_dict = get_val_ground_dict()
    folders = folders[0::1000//num_classes]
    # Table contains labels that are associated with those folders
    labels_searched = []
    for folder in folders:
        labels_searched.append(label_dict[folder])

    print("Processing folder %s" % in_dir)
    labels_list = []
    images = []
    for image_name in os.listdir(in_dir):
        # Get label for that image
        # If it was resized using 'image_resizer_imagenet.py' script then we know that it has extension '.png'
        label = val_ground_dict[image_name[:-4]]

        # Ignore if it's not one of the subsampled classes
        if label not in labels_searched:
            continue
        try:
            img = imageio.imread(os.path.join(in_dir, image_name),pilmode='RGB')
            r = img[:, :, 0].flatten()
            g = img[:, :, 1].flatten()
            b = img[:, :, 2].flatten()

        except:
            print('Cant process image %s' % os.path.join(in_dir, image_name))
            with open("log_img2np_val.txt", "a") as f:
                f.write("Couldn't read: %s" % os.path.join(in_dir, image_name))
            continue
        arr = np.array(list(r) + list(g) + list(b), dtype=np.uint8)
        images.append(arr)
        labels_list.append(label)

    data_val = np.row_stack(images)
    labels_list= np.array(labels_list)
    y_test = []
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    val_indices = np.arange(data_val.shape[0])
    np.random.shuffle(val_indices)
    curr_index = 0
    size = data_val.shape[0] // n
    for i in range(1,n):
        d_val={
        'data': data_val[val_indices[curr_index: (curr_index + size)], :],
        'labels': labels_list[val_indices[curr_index: (curr_index + size)]].tolist()
    	}
        
        pickle.dump(d_val, open(os.path.join(out_dir, 'val_data_batch_%d' % i), 'wb'))
        curr_index += size
        y_test.extend(d_val['labels'])
  
    # Create last file
    d_val = {
        'data': data_val[val_indices[curr_index:], :],
        'labels': labels_list[val_indices[curr_index:]].tolist(),
    }
    pickle.dump(d_val, open(os.path.join(out_dir, 'val_data_batch_%d' % n), 'wb'))
    y_test.extend(d_val['labels'])

    count = np.zeros([1000])
    for i in y_test:
        count[i-1] += 1

    for i in range(1000):
        print('%d : %d' % (i, count[i]))

    print('SUM: %d' % len(y_test))

if __name__ == '__main__':
    in_dir, out_dir = parse_arguments()

    print("Start program ...")
    process_folder(in_dir=in_dir, out_dir=out_dir)
    print("Finished.")
