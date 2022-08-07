import torch
import torchvision
from CONFIG import *
from torchvision.models import ResNet18_Weights, ResNet34_Weights, MobileNet_V2_Weights


class ClassificationModel(torch.nn.Module):
	def __init__(self, number_classes, chosen_model):
		super(ClassificationModel, self).__init__()
		self.number_classes = number_classes
		self.Softmax = torch.nn.Softmax()
		self.chosen_model = chosen_model
		if chosen_model == 0:
			self.convolutional_layers = self._create_conv_layers_ResNet18()
			self.fully_connected_layers = self._create_fully_connected_layers(number_classes=self.number_classes, num_input_params=512)
		elif chosen_model == 1:
			self.convolutional_layers = self._create_conv_layers_ResNet34()
			self.fully_connected_layers = self._create_fully_connected_layers(number_classes=self.number_classes, num_input_params=512)
		elif chosen_model == 2:
			self.convolutional_layers = self._create_conv_layers_MobileNetV2()
			self.fully_connected_layers = self._create_fully_connected_layers(number_classes=self.number_classes, num_input_params=1280)

	def forward(self, x):
		x = self.convolutional_layers(x)
		x = self.fully_connected_layers(x)
		return x

	@staticmethod
	def _create_convolutional_layers_ResNet18():
		resnet18 = torchvision.models.resnet18(ResNet18_Weights.IMAGENET1K_V1)
		resnet18_layers = torch.nn.Sequential(*list(resnet18.children())[:-1])
		return resnet18_layers

	@staticmethod
	def _create_convolutional_layers_ResNet34():
		resnet34 = torchvision.models.resnet34(ResNet34_Weights.IMAGENET1K_V1)
		resnet34_layers = torch.nn.Sequential(*list(resnet34.children())[:-1])
		return resnet34_layers

	@staticmethod
	def _create_conv_layers_MobileNetV2():
		mobilenetV2 = torchvision.models.mobilenet_v2(MobileNet_V2_Weights.IMAGENET1K_V2)
		mobilenetV2_layers = torch.nn.Sequential(*list(mobilenetV2.children())[:-1], torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
		return mobilenetV2_layers

	@staticmethod
	def _create_fully_connected_layers(number_classes, num_input_params):
		return torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(num_input_params, 4096), torch.nn.Dropout(DROPOUT), torch.nn.LeakyReLU(0.1), torch.nn.Linear(4096, number_classes + 1))
