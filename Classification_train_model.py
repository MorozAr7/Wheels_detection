import torch
from CONFIG import *
import numpy as np
from Classification_models import ClassificationModel
from Classification_dataset import ClassificationDataset
from utils import init_model_weights, save_pretraining
from data_augmentation import transformations_classification
import matplotlib.pyplot as plt

DEVICE = CHOOSE_CUDA if torch.cuda.is_available() else "cpu"


def one_epoch(data_loader, model, optimizer, loss_function, is_training=True):
	model.train() if is_training else model.eval()

	epoch_loss = 0

	epoch_predictions = torch.tensor([])
	epoch_labels = torch.tensor([])
	if not is_training:
		with torch.no_grad():
			for batch_index, (images, labels) in enumerate(data_loader):
				images = images.to(DEVICE).float()
				plt.imshow(np.array(images[0, ...].permute(1, 2, 0)))
				plt.show()
				labels = labels.to(DEVICE)
				print(batch_index)
				predictions = model(images)

				loss = loss_function(predictions, labels)
				_, argmax = torch.max(predictions, dim=1)
				epoch_predictions = torch.cat([epoch_predictions, argmax], dim=0)
				epoch_labels = torch.cat([epoch_labels, labels], dim=0)

				optimizer.zero_grad()

				torch.cuda.empty_cache()
				epoch_loss += loss.item()
	else:
		for batch_index, (images, labels) in enumerate(data_loader):
			images = images.to(DEVICE).float()

			labels = labels.to(DEVICE)
			print(batch_index)
			predictions = model(images)

			loss = loss_function(predictions, labels)
			_, argmax = torch.max(predictions, dim=1)
			epoch_predictions = torch.cat([epoch_predictions, argmax], dim=0)
			epoch_labels = torch.cat([epoch_labels, labels], dim=0)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			torch.cuda.empty_cache()
			epoch_loss += loss.item()

	return epoch_loss, (torch.sum((epoch_predictions == epoch_labels))/epoch_predictions.shape[0]).item()


def main(train_data_loader, test_data_loader, model, optimizer, loss_function):

	best_validation_accuracy = 0
	training_loss = list()
	validation_loss = list()

	for epoch in range(NUMBER_EPOCHS):
		training_epoch_loss, training_accuracy = one_epoch(data_loader=train_data_loader, model=model, optimizer=optimizer, loss_function=loss_function)
		validation_epoch_loss, validation_accuracy = one_epoch(data_loader=test_data_loader, model=model, optimizer=optimizer, loss_function=loss_function, is_training=False)
		training_loss.append(training_epoch_loss)
		validation_loss.append(validation_epoch_loss)

		print("EPOCH: {}, TRAINING LOSS: {}, VALIDATION_LOSS: {}, TRAINING ACCURACY: {}, VALIDATION ACCURACY: {}".format(epoch, training_epoch_loss, validation_epoch_loss, training_accuracy, validation_accuracy))

		if validation_accuracy > best_validation_accuracy:
			best_validation_accuracy = validation_accuracy
			print("SAVING MODEL")
			save_pretraining(model.state_dict())
			print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":

	model = ClassificationModel(number_classes=NUMBER_CLASSES, chosen_model=CHOSEN_MODEL).to(DEVICE)
	#print(model.state_dict())
	#exit()
	model = init_model_weights(model=model)

	dataset = ClassificationDataset(csv_file=CSV_FILE_CLASSIFICATION, transformations=transformations_classification)
	optimizer = torch.optim.Adam(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, params=model.parameters())
	loss_function = torch.nn.CrossEntropyLoss()

	train_subset, test_subset = torch.utils.data.random_split(dataset, DATASET_SPLIT)
	train_data_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=BATCH_SIZE_TRAINING, shuffle=True)
	test_data_loader = torch.utils.data.DataLoader(dataset=test_subset, batch_size=BATCH_SIZE_TESTING, shuffle=True)

	main(train_data_loader=train_data_loader, test_data_loader=test_data_loader, model=model, optimizer=optimizer, loss_function=loss_function)
