import torch


class LossFunction(torch.nn.Module):
	def __init__(self, grid_size, number_classes, anchor_box):
		super(LossFunction, self).__init__()

		self.grid_size = grid_size
		self.number_classes = number_classes
		self.anchor_box = torch.tensor(anchor_box).reshape(1, 1, 1, 2)
		self.MSE = torch.nn.MSELoss()
		self.BCE = torch.nn.BCELoss()
		self.CE = torch.nn.CrossEntropyLoss()
		self.sigmoid = torch.nn.Sigmoid()
		self.lambda_ = 5

	def forward(self, predictions, labels):
		#   prediction.shape = [Batch_size, Grid_size, Grid_size, 8] where the last dimension is [p_c1, p_c2, p_c3, confidence, x, y, w, h]
		#   labels.shape = [Batch_size, Grid_size, Grid_size, 8] where the last dimension is [p_c1, p_c2, p_c3, confidence, x, y, w, h]
		#exists_box = labels[..., self.number_classes].unsqueeze(3)
		#print(predictions.shape, labels.shape)
		object_mask = labels[..., self.number_classes] == 1
		no_object_mask = labels[..., self.number_classes] == 0

		no_object_loss = self.MSE(torch.flatten(predictions[..., self.number_classes:self.number_classes + 1][no_object_mask]),
		                          torch.flatten(labels[..., self.number_classes:self.number_classes + 1][no_object_mask]))

		object_loss = self.MSE(torch.flatten(predictions[..., self.number_classes:self.number_classes + 1][object_mask]),
		                       torch.flatten(labels[..., self.number_classes:self.number_classes + 1][object_mask]))

		bbox_loss1 = self.MSE(torch.flatten(predictions[..., -4:-2][object_mask], end_dim=-2), torch.flatten(labels[..., -4:-2][object_mask], end_dim=-2))
		bbox_loss2 = self.MSE(torch.flatten(predictions[..., -2:][object_mask], end_dim=-2), torch.flatten((labels[..., -2:][object_mask]), end_dim=-2))
		#class_loss = self.CE(predictions[..., :self.number_classes] * exists_box, labels[..., :self.number_classes] * exists_box)

		class_loss = self.MSE(torch.flatten(predictions[..., :self.number_classes][object_mask], end_dim=-2), torch.flatten(labels[..., :self.number_classes][object_mask], end_dim=-2))
		return object_loss + 2 * no_object_loss + self.lambda_ * (bbox_loss1 + bbox_loss2 + class_loss/5)

