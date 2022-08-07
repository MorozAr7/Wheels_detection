import copy
import os
import cv2
import numpy as np
import random
import time
import csv
import utils
from utils import rotate
from CONFIG import (ROOT_PATH, DATASET_SPLIT, IMAGE_SIZE_WIDTH)
from Detection_artificial_dataset import check_collision

WHEEL_SIZE = int(250)


def reshape(image, w, h):
	"""
	Function reshapes a batch of images into 256x256 size
	:param images:
	:type ndarray:
	:return reshaped:
	:type ndarray:
	"""
	return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def create_image():
	image_base = np.zeros((1000, 1000, 3))
	tyre = reshape(cv2.imread(ROOT_PATH + "images/tyre_black.jpg"), w=WHEEL_SIZE, h=WHEEL_SIZE)
	disc = reshape(cv2.imread(ROOT_PATH + "images/disc_black.jpg"), w=WHEEL_SIZE, h=WHEEL_SIZE)
	wheel = reshape(cv2.imread(ROOT_PATH + "images/wheel_black.jpg"), w=WHEEL_SIZE, h=WHEEL_SIZE)

	images_dict = {"0": tyre, "1": wheel, "2": disc}

	base_h, base_w = 1000, 1000

	number_wheels = random.randint(0, 4)

	if number_wheels == 0:
		label = 3
	else:
		label = random.randint(0, 2)

	data_dict = dict()
	wheels_sizes = list()

	for i in range(number_wheels):
		wheel_size = random.randint(200, 400)
		wheels_sizes.append(wheel_size)
	wheels_sizes.sort(reverse=True)

	for i in range(number_wheels):
		wheel_size = wheels_sizes[i]
		image_add = images_dict[str(label)]
		image_copy = copy.deepcopy(image_add)
		angle = 360 * random.random()
		rotated = rotate(image_copy, angle)
		rotated = utils.remove_black(rotated)
		time_start = time.time()
		while True:
			if time.time() - time_start > 3:
				print("TOO MUCH TIME")
				break
			x_min = random.randint(0, base_h - wheel_size)
			y_min = random.randint(0, base_w - wheel_size)
			if check_collision(data_dict, x_min, y_min, wheel_size):
				image_base[y_min:y_min + wheel_size, x_min:x_min + wheel_size, :] = reshape(rotated[:, :, :], w=wheel_size, h=wheel_size)
				data_dict[str(i)] = [x_min, y_min, wheel_size]
				break

	return image_base, label


def create_classification_dataset():
	file = open("classification_dataset_description.csv", "w")
	csvwriter = csv.writer(file)
	all_rows = [["Image Path", "Labels"]]
	os.mkdir(ROOT_PATH + "wheels_classification_dataset")
	os.mkdir(ROOT_PATH + "wheels_classification_dataset/images")

	for num in range(DATASET_SPLIT[0] + DATASET_SPLIT[1]):
		print(num)
		image, label = create_image()

		path_images = ROOT_PATH + "wheels_classification_dataset/images/wheel_{}.jpeg".format(num)
		image = utils.remove_black(image)
		image = reshape(image, IMAGE_SIZE_WIDTH, IMAGE_SIZE_WIDTH)

		cv2.imwrite(path_images, image)
		all_rows.append([path_images, label])

	csvwriter.writerows(all_rows)
	file.close()


if __name__ == "__main__":
	create_classification_dataset()
