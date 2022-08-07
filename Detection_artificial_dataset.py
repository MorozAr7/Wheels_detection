import copy
import os
import cv2
import numpy as np
import random
import torch
from copy import deepcopy
import matplotlib.pyplot as plt
import csv
import time
import utils
from utils import rotate, compute_IOU
from CONFIG import (ROOT_PATH, DATASET_SPLIT, IMAGE_SIZE_WIDTH)

#IMAGE_SIZE = 1000
WHEEL_SIZE = int(250)


def reshape(image, w, h):
	"""
	Function reshapes a batch of images into 256x256 size
	:param images:
	:type ndarray:
	:return reshaped:
	:type ndarray:
	"""

	reshaped_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)

	return reshaped_image


def check_collision(data_dict, x_middle1, y_middle1, w1, h1):
	for key in list(data_dict.keys()):
		x_middle2, y_middle2, w2, h2 = data_dict[key][0], data_dict[key][1], data_dict[key][2] + 25, data_dict[key][2] + 25

		IOU = compute_IOU(torch.tensor([x_middle1, y_middle1, w1, h1]), torch.tensor([x_middle2, y_middle2, w2, h2]))
		if IOU.item() > 0.0:
			return False
	return True


def create_image():
	image_base = np.zeros((1000, 1000, 3))
	tyre = reshape(cv2.imread(ROOT_PATH + "images/tyre_black.jpg"), w=WHEEL_SIZE, h=WHEEL_SIZE)
	disc = reshape(cv2.imread(ROOT_PATH + "images/disc_black.jpg"), w=WHEEL_SIZE, h=WHEEL_SIZE)
	wheel = reshape(cv2.imread(ROOT_PATH + "images/wheel_black.jpg"), w=WHEEL_SIZE, h=WHEEL_SIZE)

	images_dict = {"0": tyre, "1": wheel, "2": disc}

	base_h, base_w = 1000, 1000
	number_images = random.randint(1, 5)

	upper_size_bound = 300
	if 4 <= number_images:
		upper_size_bound = 250

	wheels_sizes = list()

	for i in range(number_images):
		wheels_sizes.append(random.randint(175, upper_size_bound))
	wheels_sizes.sort(reverse=True)

	data_dict = dict()
	print(number_images)
	for i in range(number_images):

		wheel_type = random.randint(0, 2)
		image_add = images_dict[str(wheel_type)]
		image_copy = copy.deepcopy(image_add)
		angle = 360 * random.random()
		rotated = rotate(image_copy, angle)
		rotated = utils.remove_black(rotated)
		wheel_size = wheels_sizes[i]
		rotated = reshape(rotated, w=wheel_size, h=wheel_size)
		time1 = time.time()
		while True:
			if time.time() - time1 > 2.5:
				print("TOO MUCH TIME")
				#print(time.time() - time1)
				break
			x_min = random.randint(0, base_h - wheel_size)
			y_min = random.randint(0, base_w - wheel_size)
			if check_collision(data_dict, x_min, y_min, wheel_size):
				image_base[y_min:y_min + wheel_size, x_min:x_min + wheel_size, :] = rotated[:, :, :]
				data_dict[str(i)] = [x_min, y_min, wheel_size, wheel_type]
				break
	plt.imshow(image_base/255)
	plt.show()
	exit()
	return image_base, data_dict


def create_dataset():
	file = open("detection_dataset_description.csv", "w")
	csvwriter = csv.writer(file)
	all_rows = [["Image Path", "Labels Path"]]
	os.mkdir(ROOT_PATH + "wheels_detection_dataset")
	os.mkdir(ROOT_PATH + "wheels_detection_dataset/images")
	os.mkdir(ROOT_PATH + "wheels_detection_dataset/labels")

	for num in range(sum(DATASET_SPLIT)):
		print("IMAGE NUMBER", num)
		image, data_dict = create_image()

		path_images = ROOT_PATH + "wheels_detection_dataset/images/wheel_{}.jpeg".format(num)
		path_labels = ROOT_PATH + "wheels_detection_dataset/labels/wheel_{}.txt".format(num)
		image = utils.remove_black(image)

		image = reshape(image, IMAGE_SIZE_WIDTH, IMAGE_SIZE_WIDTH)
		cv2.imwrite(path_images, image)

		all_rows.append([path_images, path_labels])
		txt_file = open(path_labels, "w")

		for key in data_dict.keys():
			wheel_size = data_dict[key][2]
			coordX = (data_dict[key][0] + wheel_size / 2)/1000
			coordY = (data_dict[key][1] + wheel_size / 2)/1000
			bbox_size = wheel_size/1000
			label = data_dict[key][3]
			line = str(coordX) + " " + str(coordY) + " " + str(bbox_size) + " " + str(bbox_size) + " " + str(label) + "\n"
			txt_file.write(line)
		print("--------" * 20)
	csvwriter.writerows(all_rows)
	file.close()


if __name__ == "__main__":
	time_start = time.time()
	create_dataset()
	time_finish = time.time()
	print(time_finish - time_start)
