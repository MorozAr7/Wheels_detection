import torch
import pandas as pd
import cv2


class ClassificationDataset(torch.utils.data.Dataset):
	def __init__(self, csv_file, transformations=None):
		super(ClassificationDataset, self).__init__()
		self.images_labels_dataframe = pd.read_csv(csv_file)
		self.transformations = transformations

	def __len__(self):
		return len(self.images_labels_dataframe)

	def __getitem__(self, index):
		image_path = self.images_labels_dataframe.iloc[index, 0]
		label = int(self.images_labels_dataframe.iloc[index, 1])

		image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
		if self.transformations:
			augmentations = self.transformations(image=image)
			image = augmentations["image"]

		return image, label
