import albumentations as A
import cv2
import matplotlib.pyplot as plt
from CONFIG import IMAGE_SIZE_WIDTH

"""image = cv2.cvtColor(cv2.imread("/Users/artemmoroz/Desktop/CIIRC_projects/WheelsDetectionModified/wheels_detection_dataset/images/wheel_6.jpeg"), cv2.COLOR_BGR2RGB)
saved_bboxes = [[0.864, 0.63, 0.25, 0.25, 2],
[0.527, 0.831, 0.25, 0.25, 2],
[0.368, 0.387, 0.25, 0.25, 2],
[0.619, 0.27, 0.25, 0.25, 2],
]
plt.imshow(image)
plt.show()
"""
from albumentations.pytorch import ToTensorV2

from torchvision.models import efficientnet_b0
#a = efficientnet_b0()
#print(a)
def plot_bboxes(img, bboxes):
	for bbox in bboxes:
		center_x, center_y, width, height, object_class = bbox[0] * IMAGE_SIZE_WIDTH, bbox[1] * IMAGE_SIZE_WIDTH, bbox[2] * IMAGE_SIZE_WIDTH, bbox[3] * IMAGE_SIZE_WIDTH, str(bbox[4])

		xmin = round(center_x - width/2)
		xmax = round(center_x + width/2)
		ymin = round(center_y - height/2)
		ymax = round(center_y + height/2)

		cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
		cv2.putText(img, object_class, (xmin, ymin + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
	plt.imshow(img)
	plt.show()


transformations_classification = A.Compose(
	[
		#A.Resize(width=256, height=256, p=0),
		#A.RandomCrop(width=140, height=140, p=0),
		#A.Rotate(limit=360, crop_border=True, p=1),
		#A.Resize(width=256, height=256, p=1),
		A.HorizontalFlip(p=1),
		A.VerticalFlip(p=1),
		A.RGBShift(r_shift_limit=100, g_shift_limit=100, b_shift_limit=100, p=1),
		A.Blur(blur_limit=5, p=1),
		A.ColorJitter(p=1),
		A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
		ToTensorV2()
	]
)

transformations_detection = A.Compose(
	[
		#A.Resize(width=256, height=256, p=1),
		#A.RandomCrop(width=200, height=200, p=1),
		#A.Rotate(limit=360, crop_border=True, p=1),
		#A.Resize(width=256, height=256, p=1),
		#A.ChannelShuffle(p=0.5),
		#A.CLAHE(p=0.5, clip_limit=10, tile_grid_size=(10,10)),
		#A.HorizontalFlip(p=0.5),
		#A.VerticalFlip(p=0.5),
		A.OneOf([
			#A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
			#A.ISONoise(color_shift=(0.01, 1), intensity=(0.01, 1), p=0.5),
			#A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True, p=0.5),
			#A.PixelDropout(dropout_prob=0.01, per_channel=True, p=0.5),
		], p=0.5),
		#A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1),
		#A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, p=0.5),
		#A.ISONoise(color_shift=(0.01, 1), intensity=(0.01, 1), p=0.5),
		#A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=True, elementwise=True),
		#A.PixelDropout(dropout_prob=0.01, per_channel=True),
		#A.ColorJitter(p=0.5, brightness=[0.5, 1], contrast=0, saturation=0, hue=0),
		A.OneOf([
			#A.ChannelShuffle(p=0.5),
			#A.RGBShift(r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=0.5),
		], p=0.5),
		A.OneOf([
			#A.Blur(p=0.5, blur_limit=15),
			#A.AdvancedBlur(p=0.5, blur_limit=(3, 15), sigmaX_limit=(0.1, 3), sigmaY_limit=(0.1, 3), rotate_limit=90, beta_limit=(0.5, 8), noise_limit=(0.75, 1.25)),
			#A.GaussianBlur(p=0.5),
			#A.MedianBlur(p=0.5, blur_limit=7),
			#A.MotionBlur(p=0.5, blur_limit=7)
		], p=0.6),
		A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
		ToTensorV2()
	], bbox_params=A.BboxParams(format="yolo", min_visibility=0.3)
)

"""
augmented = transformations_detection(image=image, bboxes=saved_bboxes)
augmented_image = augmented['image']
augmented_bboxes = augmented['bboxes']
plot_bboxes(augmented_image, augmented_bboxes)

print(augmented_bboxes)

"""
