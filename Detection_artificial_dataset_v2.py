import random
import copy
from Detection_artificial_dataset import reshape, check_collision
from utils import rotate, remove_background, remove_white, remove_black
import numpy as np
import time
import cv2
import sys
import os
import csv
import matplotlib.pyplot as plt
from CONFIG import ROOT_PATH, DATASET_SPLIT, IMAGE_SIZE_HEIGHT, IMAGE_SIZE_WIDTH
import albumentations as A


np.set_printoptions(threshold=sys.maxsize)


def get_all_objects_mask(image):

	zeros = 180 * np.ones((image.shape[0], image.shape[1]))
	R_channel = image[..., 0] <= zeros
	G_channel = image[..., 1] <= zeros
	B_channel = image[..., 2] <= zeros
	object_mask = np.array(([R_channel * G_channel * B_channel]))
	return object_mask


def add_object_shadows(image, data_dict):
	object_mask = get_no_objects_mask(image)
	image_no_objects = image * (np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))

	shadow_color_base = (150, 151, 153)
	shadow_random_shift = np.random.uniform(low=-5, high=5)
	image_objects = (image - shadow_color_base - shadow_random_shift) * (1 - np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))
	shadow_color = (shadow_color_base[0] + shadow_random_shift, shadow_color_base[1] + shadow_random_shift, shadow_color_base[2] + shadow_random_shift)

	for key, value in data_dict.items():
		x_coord, y_coord = int(value[0] / (IMAGE_SIZE_WIDTH/1600)), int(value[1] / (IMAGE_SIZE_WIDTH/1600)) + 50
		magnitude = value[-2]
		angle_wheel_rotation = value[-1]

		left_shift = -int(np.abs(angle_wheel_rotation * 1.2)) - 50 - 4 * magnitude
		right_shift = int(np.abs(angle_wheel_rotation * 1.2)) + 50 + 4 * magnitude
		radius = int(value[2] * 3.25)
		if magnitude == 1:
			y_shift = 55
			y_shift_coeff = 1.25
		elif magnitude == 2:
			y_shift = 70
			y_shift_coeff = 1.35
		elif magnitude == 3:
			y_shift = 85
			y_shift_coeff = 1.5

		image_no_objects = cv2.circle(image_no_objects, center=(x_coord, y_coord - int(radius * y_shift_coeff)), radius=radius, color=shadow_color, thickness=-1)
		image_no_objects = cv2.circle(image_no_objects, center=(x_coord, y_coord + y_shift), radius=radius, color=shadow_color, thickness=-1)
		image_no_objects = cv2.circle(image_no_objects, center=(x_coord + left_shift, y_coord), radius=radius, color=shadow_color, thickness=-1)
		image_no_objects = cv2.circle(image_no_objects, center=(x_coord + right_shift, y_coord), radius=radius, color=shadow_color, thickness=-1)

	for _ in range(5):
		image_no_objects = apply_blur(image_no_objects)
	image = image_no_objects + image_objects

	return image


def apply_blur(image):
	blur_transform = A.Compose([
		A.GaussianBlur(p=1, blur_limit=(101, 101), sigma_limit=[30, 31], always_apply=True)
	])

	transformed_image = blur_transform(image=image)["image"]

	return transformed_image


def add_margin_image(image, border_width, margin_thickness):
	image = np.zeros(image.shape)
	x_min, y_min = border_width - margin_thickness, border_width - margin_thickness
	bottom_aruco_height = border_width - margin_thickness - 100

	main_margin_color = [171 - 194, 172 - 197, 167 - 197]
	random_color_shift = 30
	shadowed_margin_color = [119 - 194, 119 - 197, 114 - 197]

	shadowed_margin_thickness = 7

	image[y_min:image.shape[0] - bottom_aruco_height, x_min:x_min + margin_thickness - shadowed_margin_thickness, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=0, size=(image.shape[0] - y_min - bottom_aruco_height, margin_thickness -
		                                                                                                       shadowed_margin_thickness, 3)) #* 10

	image[y_min:y_min + margin_thickness - shadowed_margin_thickness, x_min:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=0, size=(margin_thickness - shadowed_margin_thickness, image.shape[1] - 2 * y_min, 3)) #* 10

	image[y_min:image.shape[0] - bottom_aruco_height, image.shape[1] - x_min - margin_thickness + shadowed_margin_thickness:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=0, size=(image.shape[0] - y_min - bottom_aruco_height, margin_thickness -
		                                                                                                        shadowed_margin_thickness, 3)) #* 10

	image[image.shape[0] - bottom_aruco_height - margin_thickness + shadowed_margin_thickness:image.shape[0] - bottom_aruco_height, x_min:image.shape[1] - x_min, :] = \
		np.array(main_margin_color) + np.random.uniform(low=-random_color_shift, high=0, size=(margin_thickness - shadowed_margin_thickness, image.shape[1] - 2 * x_min, 3))# * 10



	image[border_width:image.shape[0] - bottom_aruco_height - margin_thickness, x_min + margin_thickness - shadowed_margin_thickness:x_min + margin_thickness, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 15, high=random_color_shift - 15, size=(image.shape[0] - bottom_aruco_height - margin_thickness - border_width,
		                                                                                                       shadowed_margin_thickness,
		                                                                                       3))

	image[y_min + margin_thickness - shadowed_margin_thickness:y_min + margin_thickness, border_width:image.shape[1] - border_width, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 15, high=random_color_shift - 15, size=(shadowed_margin_thickness, image.shape[1] - 2 * border_width, 3))

	image[border_width:image.shape[0] - bottom_aruco_height - margin_thickness, image.shape[1] - x_min - margin_thickness:image.shape[1] - x_min - margin_thickness + shadowed_margin_thickness, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 15, high=random_color_shift - 15, size=(image.shape[0] - margin_thickness - bottom_aruco_height - border_width,
		                                                                                                       shadowed_margin_thickness,
		                                                                                       3))

	image[image.shape[0] - bottom_aruco_height - margin_thickness:image.shape[0] - bottom_aruco_height - margin_thickness + shadowed_margin_thickness, border_width:image.shape[1] - border_width, :] = \
		np.array(shadowed_margin_color) + np.random.uniform(low=-random_color_shift + 15, high=random_color_shift - 15, size=(shadowed_margin_thickness, image.shape[1] - 2 * border_width, 3))

	return image


def add_aruco_tags(image, border_width, margin_thickness):
	aruco_tags = list()
	aruco_tag_height = list()

	image = np.zeros(image.shape)
	for aruco_tag_index in range(10):
		aruco_tag = cv2.imread(ROOT_PATH + "/aruco_tags/aruco_tag{}.png".format(aruco_tag_index + 1)) #- 205
		aruco_width = border_width - margin_thickness
		if aruco_tag_index + 1 in [6, 7, 8, 9]:
			aruco_height = border_width - margin_thickness - 100
			aruco_tag = reshape(aruco_tag, w=aruco_width, h=aruco_height)
			aruco_tag_height.append(aruco_height)
		else:
			aruco_height = border_width - margin_thickness
			aruco_tag = reshape(aruco_tag, w=aruco_width, h=aruco_height)
			aruco_tag_height.append(aruco_height)
		aruco_tags.append(aruco_tag)
	aruco_tag_coords = [(0, 0), (500, 0), (1000, 0), (1600 - aruco_tag_height[3], 0), (1600 - aruco_tag_height[4], 434),
	                    (1600 - aruco_tag_height[4], 1000 - aruco_tag_height[5]), (1000, 1000 - aruco_tag_height[6]), (500, 1000 - aruco_tag_height[7]), (0, 1000 - aruco_tag_height[8]), (0, 434)]

	for index in range(len(aruco_tag_coords)):
		image[aruco_tag_coords[index][1]:aruco_tag_coords[index][1] + aruco_tag_height[index], aruco_tag_coords[index][0]:aruco_tag_coords[index][0] + aruco_tag_height[0]] = (aruco_tags[index] -
		                                                                                                                                                                       np.array([205]))

	return image


def remove_background_selected_area(image, x_middle, y_middle, height, width):
	ones = np.ones((image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[0], image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[1]))

	R_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0] <= 240 * ones
	G_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 1] <= 240 * ones
	B_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 2] <= 240 * ones

	mask = np.array([R_channel * G_channel * B_channel])
	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] * (np.transpose(np.repeat(mask, [3], axis=0), axes=[1, 2, 0]))
	return image


def remove_redundant_white(image, insertion, x_middle, y_middle, height, width):

	if insertion.shape[1] != (x_middle + width) - (x_middle - width) or insertion.shape[0] != (y_middle + height) - (y_middle - height):
		insertion = reshape(insertion, h=(y_middle + height) - (y_middle - height), w=(x_middle + width) - (x_middle - width))

	image = copy.deepcopy(image)
	ones = np.ones(
		(image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[0], image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0].shape[1]))
	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] += insertion

	R_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 0] >= 255 * ones
	G_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 1] >= 255 * ones
	B_channel = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, 2] >= 255 * ones

	mask = np.array([R_channel * G_channel * B_channel])

	image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] = image[y_middle - height:y_middle + height, x_middle - width: x_middle + width, :] - np.array([255]) * \
	                                                                                                                                                                       np.transpose(np.repeat(mask, [3], axis=0), axes=[1, 2, 0])

	return image


def plot_bbox(image, x_middle, y_middle, w, h, wheel_type):
	cv2.rectangle(image, (int(x_middle - w / 2), int(y_middle - h / 2)), (int(x_middle + w / 2), int(y_middle + h / 2)), (255, 0, 0), 1)
	cv2.putText(image, str(wheel_type), (x_middle, y_middle-25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	return image


def get_no_objects_mask(image):
	threshold = np.ones((image.shape[0], image.shape[1])) * (170)
	R_channel = image[..., 0] >= threshold
	G_channel = image[..., 1] >= threshold
	B_channel = image[..., 2] >= threshold
	no_object_mask = np.array(([R_channel * G_channel * B_channel]))

	return no_object_mask


def add_background_shadows(image):
	no_object_mask = get_no_objects_mask(image)
	random_gray_color_shift = np.random.uniform(low=-5, high=0, size=(image.shape[0], image.shape[1], 1))

	background = (image + np.repeat(random_gray_color_shift, [3], axis=-1)) * np.transpose(np.repeat(no_object_mask, [3], axis=0), axes=[1, 2, 0])

	full_image = background + image * (1 - np.transpose(np.repeat(no_object_mask, [3], axis=0), axes=[1, 2, 0]))
	return full_image


def generate_image(objects_transformed):
	background_height, background_width = 1000, 1600
	image_base = np.ones((background_height, background_width, 3)) * 255
	max_wheels = [5, 5, 4, 4, 3, 3]

	distance_array = [1, 1.05, 1.1, 1.15, 1.2, 1.25]
	distance_index = random.randint(0, 5)
	distance = distance_array[distance_index]
	wheel_width = [276, 290, 304, 318, 332, 346][distance_index]
	wheel_height_magnitude_1, wheel_height_magnitude_2, wheel_height_magnitude_3 = [304, 320, 334, 350, 364, 380][distance_index], \
	                                                                               [332, 348, 366, 382, 398, 414][distance_index], \
	                                                                               [360, 378, 396, 414, 432, 450][distance_index]

	disk_width = [235, 247, 259, 270, 282, 294][distance_index]
	disk_height_magnitude_1, disk_height_magnitude_2, disk_height_magnitude_3 = [258, 272, 284, 298, 309, 323][distance_index], \
	                                                                            [282, 296, 311, 325, 338, 352][distance_index], \
	                                                                            [306, 321, 336, 352, 367, 383][distance_index]

	border_width = int(138 * distance)
	wheel_bbox_size = [45, 47, 51, 53, 55, 57][distance_index]
	disk_bbox_size = int(wheel_bbox_size * 0.85)
	margin_thickness = [16, 17, 18, 19, 20, 21][distance_index]
	camera_central_point = (random.randint(0, background_width), random.randint(700, background_height))
	#angle_camera_perspective = (-60 / (background_width / 2)) * (camera_central_point[0] - background_width / 2)
	#top_black_point = 0.5 * (camera_central_point[0] + 2400)
	#bottom_black_point = top_black_point - 1000 * np.tan(angle_camera_perspective) if top_black_point >= 800 else top_black_point + 1000 * np.tan(angle_camera_perspective)

	disk_magnitude_1 = reshape(objects_transformed[0], w=disk_width, h=disk_height_magnitude_1)
	disk_magnitude_2 = reshape(objects_transformed[1], w=disk_width, h=disk_height_magnitude_2)
	disk_magnitude_3 = reshape(objects_transformed[2], w=disk_width, h=disk_height_magnitude_3)
	disk_dark_magnitude_1 = reshape(objects_transformed[3], w=disk_width, h=disk_height_magnitude_1)
	disk_dark_magnitude_2 = reshape(objects_transformed[4], w=disk_width, h=disk_height_magnitude_2)
	disk_dark_magnitude_3 = reshape(objects_transformed[5], w=disk_width, h=disk_height_magnitude_3)
	tyre_magnitude_1 = reshape(objects_transformed[6], w=wheel_width, h=wheel_height_magnitude_1)
	tyre_magnitude_2 = reshape(objects_transformed[7], w=wheel_width, h=wheel_height_magnitude_2)
	tyre_magnitude_3 = reshape(objects_transformed[8], w=wheel_width, h=wheel_height_magnitude_3)
	wheel_magnitude_1 = reshape(objects_transformed[9], w=wheel_width, h=wheel_height_magnitude_1)
	wheel_magnitude_2 = reshape(objects_transformed[10], w=wheel_width, h=wheel_height_magnitude_2)
	wheel_magnitude_3 = reshape(objects_transformed[11], w=wheel_width, h=wheel_height_magnitude_3)
	wheel_dark_magnitude_1 = reshape(objects_transformed[12], w=wheel_width, h=wheel_height_magnitude_1)
	wheel_dark_magnitude_2 = reshape(objects_transformed[13], w=wheel_width, h=wheel_height_magnitude_2)
	wheel_dark_magnitude_3 = reshape(objects_transformed[14], w=wheel_width, h=wheel_height_magnitude_3)

	images_dict = {"0": [tyre_magnitude_1, tyre_magnitude_2, tyre_magnitude_3],
	               "1": [disk_magnitude_1, disk_magnitude_2, disk_magnitude_3],
	               "2": [disk_dark_magnitude_1, disk_dark_magnitude_2, disk_dark_magnitude_3],
	               "3": [wheel_magnitude_1, wheel_magnitude_2, wheel_magnitude_3],
	               "4": [wheel_dark_magnitude_1, wheel_dark_magnitude_2, wheel_dark_magnitude_3]}

	number_wheels = random.randint(1, max_wheels[distance_index])
	print("Number of wheels {}".format(number_wheels))
	data_dict = dict()

	for i in range(number_wheels):
		wheel_type = random.randint(0, 4)
		wheels_set = images_dict[str(wheel_type)]
		time1 = time.time()
		while True:
			if time.time() - time1 > 2.5:
				print("TIME EXCEEDED!")
				print(time.time() - time1)
				break

			x_middle = random.randint(border_width + int(wheel_width / 2), background_width - border_width - int(wheel_width / 2))
			y_middle = random.randint(border_width + int(wheel_height_magnitude_3/2), background_height - 30 - int(wheel_height_magnitude_3/2))

			magnitude = 1
			if camera_central_point[1] - y_middle <= 287 - int(wheel_height_magnitude_3/2):
				magnitude = 1
			elif 287 - int(wheel_height_magnitude_3/2) <= camera_central_point[1] - y_middle <= 575 - int(wheel_height_magnitude_3/2):
				magnitude = 2
			elif 575 - int(wheel_height_magnitude_3/2) <= camera_central_point[1] - y_middle <= 862 - int(wheel_height_magnitude_3/2):
				magnitude = 3

			angle_rotation = (60 / 1186) * (- x_middle + camera_central_point[0])
			angle_diff = 0
			if 0 <= np.abs(angle_rotation) <= 20:
				angle_diff = 55

			rotated_wheel = rotate(wheels_set[magnitude - 1], angle_rotation)
			height_min = wheels_set[magnitude - 1].shape[0] / 2
			width_min = wheels_set[magnitude - 1].shape[1] / 2
			height_width_max = np.sqrt(height_min ** 2 + width_min ** 2)

			delta_h = int((height_width_max - height_min) * np.sin(np.deg2rad(np.abs(angle_rotation) + 60 - angle_diff)))
			delta_w = int((height_width_max - height_min) * np.sin(np.deg2rad(np.abs(angle_rotation) + 60 - angle_diff)))

			if wheel_type == 0:
				wheel_class = 0
				bbox_size = wheel_bbox_size
			elif wheel_type in [1, 2]:
				wheel_class = 1
				bbox_size = disk_bbox_size
			elif wheel_type in [3, 4]:
				wheel_class = 2
				bbox_size = wheel_bbox_size
			if angle_rotation > 0:
				if wheel_class == 1:
					if magnitude == 1:
						height_shift = int(12 / distance) + int(angle_rotation * 0.35)
						width_shift = -angle_rotation * 0.5 / distance
					elif magnitude == 2:
						height_shift = int(15 / distance) + int(angle_rotation * 0.2)
						width_shift = - angle_rotation * 0.45 / distance
					else:
						height_shift = int(1 / distance) + int(angle_rotation * 0.1)
						width_shift = -angle_rotation * 0.6 / distance
				else:
					if magnitude == 1:
						height_shift = int(12.5 / distance) + int(angle_rotation * 0.4)
						width_shift = - angle_rotation * 0.3 / distance
					elif magnitude == 2:
						height_shift = int(9.5 / distance) + int(angle_rotation * 0.3)
						width_shift = - angle_rotation * 0.45 / distance
					else:
						height_shift = int(0.1 / distance) + int(angle_rotation * 0.05)
						width_shift = - angle_rotation * 0.65 / distance
			else:
				if wheel_class == 1:
					if magnitude == 1:
						height_shift = int(20 / distance) + int(-angle_rotation * 0.45)
						width_shift = -angle_rotation * 0.5 / distance
					elif magnitude == 2:
						height_shift = int(9 / distance) + int(-angle_rotation * 0.2)
						width_shift = - angle_rotation * 0.55 / distance
					else:
						height_shift = -int(3 / distance) + int(-angle_rotation * 0.1)
						width_shift = -angle_rotation * 0.8 / distance
				else:
					if magnitude == 1:
						height_shift = int(17 / distance) + int(-angle_rotation * 0.4)
						width_shift = - angle_rotation * 0.3 / distance
					elif magnitude == 2:
						height_shift = int(8 / distance) + int(-angle_rotation * 0.3)
						width_shift = - angle_rotation * 0.75 / distance
					else:
						height_shift = -int(7 / distance) + int(-angle_rotation * 0.05)
						width_shift = - angle_rotation * 1.2 / distance

			if check_collision(data_dict, int((x_middle - angle_rotation / 2) * (IMAGE_SIZE_WIDTH/background_width)), int((IMAGE_SIZE_WIDTH/background_width) * (y_middle - height_shift)), bbox_size + 15,
			                   bbox_size + 15):
				if delta_h == 0:
					image_base = remove_redundant_white(image_base, insertion=rotated_wheel[...], x_middle=x_middle, y_middle=y_middle,
					                                    height=int(rotated_wheel.shape[0] / 2) - delta_h, width=int(rotated_wheel.shape[1] / 2) - delta_w)
				else:
					image_base = remove_redundant_white(image_base, insertion=rotated_wheel[delta_h:-delta_h, delta_w:-delta_w, :], x_middle=x_middle, y_middle=y_middle,
					                                    height=int(rotated_wheel.shape[0] / 2) - delta_h, width=int(rotated_wheel.shape[1] / 2) - delta_w)

				data_dict[str(i)] = [int((x_middle + width_shift) * (IMAGE_SIZE_WIDTH/background_width)), int((IMAGE_SIZE_WIDTH/background_width) * int(y_middle + height_shift)), int(bbox_size * 0.9),
				                     wheel_class, magnitude,
				                     angle_rotation]
				break

	image_base = remove_white(image_base)
	object_mask = get_all_objects_mask(image_base)
	image_margins = add_margin_image(image_base, border_width, margin_thickness)
	image_base = add_object_shadows(image_base, data_dict)
	image_base += image_margins * (1 - np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))
	aruco_tags = add_aruco_tags(image_base, border_width, margin_thickness)
	image_base += aruco_tags * (1 - np.transpose(np.repeat(object_mask, [3], axis=0), axes=[1, 2, 0]))
	image_base = add_background_shadows(image_base)
	image_base = reshape(image_base, IMAGE_SIZE_WIDTH, IMAGE_SIZE_HEIGHT)

	"""	for key, value in data_dict.items():
		image_base = plot_bbox(image_base, value[0], value[1], value[2], value[2], value[3])
	plt.imshow(image_base/255)
	plt.show()
	exit()"""

	return image_base, data_dict


def generate_dataset():
	disk_magnitude_1 = cv2.imread(ROOT_PATH + "images/disk_magnitude_1.jpg")
	disk_magnitude_2 = cv2.imread(ROOT_PATH + "images/disk_magnitude_2.jpg")
	disk_magnitude_3 = cv2.imread(ROOT_PATH + "images/disk_magnitude_3.jpg")
	disk_dark_magnitude_1 = cv2.imread(ROOT_PATH + "images/disk_dark_magnitude_1.jpg")
	disk_dark_magnitude_2 = cv2.imread(ROOT_PATH + "images/disk_dark_magnitude_2.jpg")
	disk_dark_magnitude_3 = cv2.imread(ROOT_PATH + "images/disk_dark_magnitude_3.jpg")
	tyre_magnitude_1 = cv2.imread(ROOT_PATH + "images/tyre_magnitude_1.jpg")
	tyre_magnitude_2 = cv2.imread(ROOT_PATH + "images/tyre_magnitude_2.jpg")
	tyre_magnitude_3 = cv2.imread(ROOT_PATH + "images/tyre_magnitude_3.jpg")
	wheel_magnitude_1 = cv2.imread(ROOT_PATH + "images/wheel_magnitude_1.jpg")
	wheel_magnitude_2 = cv2.imread(ROOT_PATH + "images/wheel_magnitude_2.jpg")
	wheel_magnitude_3 = cv2.imread(ROOT_PATH + "images/wheel_magnitude_3.jpg")
	wheel_dark_magnitude_1 = cv2.imread(ROOT_PATH + "images/wheel_dark_magnitude_1.jpg")
	wheel_dark_magnitude_2 = cv2.imread(ROOT_PATH + "images/wheel_dark_magnitude_2.jpg")
	wheel_dark_magnitude_3 = cv2.imread(ROOT_PATH + "images/wheel_dark_magnitude_3.jpg")

	file = open("detection_dataset_description.csv", "w")
	csvwriter = csv.writer(file)
	all_rows = [["Image Path", "Labels Path"]]
	os.mkdir(ROOT_PATH + "wheels_detection_dataset")
	os.mkdir(ROOT_PATH + "wheels_detection_dataset/images")
	os.mkdir(ROOT_PATH + "wheels_detection_dataset/labels")

	for num in range(sum(DATASET_SPLIT)):
		print("Image number {}".format(num))
		image, data_dictionary = generate_image([disk_magnitude_1,
		                                         disk_magnitude_2,
		                                         disk_magnitude_3,
		                                         disk_dark_magnitude_1,
		                                         disk_dark_magnitude_2,
		                                         disk_dark_magnitude_3,
		                                         tyre_magnitude_1,
		                                         tyre_magnitude_2,
		                                         tyre_magnitude_3,
		                                         wheel_magnitude_1,
		                                         wheel_magnitude_2,
		                                         wheel_magnitude_3,
		                                         wheel_dark_magnitude_1,
		                                         wheel_dark_magnitude_2,
		                                         wheel_dark_magnitude_3])

		path_images = ROOT_PATH + "wheels_detection_dataset/images/wheel_{}.jpeg".format(num)
		path_labels = ROOT_PATH + "wheels_detection_dataset/labels/wheel_{}.txt".format(num)
		cv2.imwrite(path_images, image)
		all_rows.append([path_images, path_labels])
		txt_file = open(path_labels, "w")

		for key in data_dictionary.keys():
			coord_x = data_dictionary[key][0] / IMAGE_SIZE_WIDTH
			coord_y = data_dictionary[key][1] / IMAGE_SIZE_HEIGHT
			bbox_size_x, bbox_size_y = data_dictionary[key][2] / IMAGE_SIZE_HEIGHT, data_dictionary[key][2] / IMAGE_SIZE_HEIGHT
			label = data_dictionary[key][3]
			line = str(coord_x) + " " + str(coord_y) + " " + str(bbox_size_x) + " " + str(bbox_size_y) + " " + str(label) + "\n"
			txt_file.write(line)

	csvwriter.writerows(all_rows)
	file.close()


if __name__ == "__main__":
	generate_dataset()
