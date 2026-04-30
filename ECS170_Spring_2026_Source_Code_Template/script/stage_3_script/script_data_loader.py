import pickle
from matplotlib import pyplot as plt
import os

# project root directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '../../'))
data_dir = os.path.join(project_root, 'data/stage_3_data')

# loading ORL dataset
if 1:
	orl_path = os.path.join(data_dir, 'ORL')
	f = open(orl_path, 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
		
# loading CIFAR-10 dataset
if 0:
	cifar_path = os.path.join(data_dir, 'CIFAR')
	f = open(cifar_path, 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix)
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break

# loading MNIST dataset
if 0:
	mnist_path = os.path.join(data_dir, 'MNIST')
	f = open(mnist_path, 'rb')
	data = pickle.load(f)
	f.close()
	for instance in data['train']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the training set
		break
		
	for instance in data['test']:
		image_matrix = instance['image']
		image_label = instance['label']
		plt.imshow(image_matrix, cmap='gray')
		plt.show()
		print(image_matrix)
		print(image_label)
		# remove the following "break" code if you would like to see more image in the testing set
		break
	
	
	