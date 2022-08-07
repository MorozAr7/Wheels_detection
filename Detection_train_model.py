import copy
import numpy as np
import time
import torch
import utils
from CONFIG import *
from Detection_models import DetectionModel
from Detection_dataset import DetectionDataset
from loss_function import LossFunction
from utils import *
from data_augmentation import transformations_detection

#torch.set_printoptions(edgeitems=1000, linewidth=2000)


def test_model(test_data_loader, loss_fcn):
    model = torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/WheelsDetectionModified/RESNET_34_mAP_saved.pt", map_location=torch.device("cpu"))
    model.eval()
    index_image = 0
    for batch_id, (images, labels) in enumerate(test_data_loader):
        images = images.to(DEVICE).float()
        labels = labels.to(DEVICE)
        predictions = model(images)
        predictions[..., 4:6] = torch.sigmoid(predictions[..., 4:6])
        predictions[..., 6:] = torch.tensor(ANCHOR_BOX).reshape(1, 1, 1, 2).to(DEVICE) * torch.sigmoid(predictions[..., 6:])

        predicted_bboxes_list, ground_truth_bboxes_list = get_bboxes(predicted_bboxes=predictions, ground_truth_bboxes=labels, image_index=index_image)
        index_image += images.shape[0]
        plot_bboxes(images=np.ascontiguousarray(images.permute(0, 2, 3, 1), dtype=np.float), predicted_bboxes=predicted_bboxes_list, ground_truth_bboxes=ground_truth_bboxes_list)
        time.sleep(1)


def one_epoch(model, optimizer, loss_function, data_loader, is_training=True, epoch=0):
    model.train() if is_training else model.eval()

    epoch_loss = 0

    all_predicted_bboxes = list()
    all_ground_truth_bboxes = list()
    index_image = 0

    for batch_index, (images, labels) in enumerate(data_loader):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        print(batch_index)
        #print(labels)
        predictions = model(images.float())
        predictions[..., 4:6] = utils.modified_sigmoid(predictions[..., 4:6], coefficient=0.75)
        predictions[..., 6:] = torch.tensor(ANCHOR_BOX).reshape(1, 1, 1, 2).to(DEVICE) * utils.modified_sigmoid(predictions[..., 6:], coefficient=0.75)

        loss = loss_function(predictions, labels)

        if is_training:
            loss.backward()
            optimizer.step()
        optimizer.zero_grad()

        predicted_bboxes_list, ground_truth_bboxes_list = get_bboxes(predicted_bboxes=predictions, ground_truth_bboxes=labels, image_index=index_image)
        index_image += images.shape[0]
        all_predicted_bboxes += predicted_bboxes_list
        all_ground_truth_bboxes += ground_truth_bboxes_list
        if is_training and (epoch + 1) % 25 == 0:
            #print(predictions)
            plot_bboxes(images.permute(0, 2, 3, 1), predicted_bboxes_list)
        torch.cuda.empty_cache()
        epoch_loss += loss.item()

    if is_training:

        return epoch_loss
    else:
        mAP = compute_mean_average_precision(predicted_bboxes=all_predicted_bboxes, ground_truth_bboxes=all_ground_truth_bboxes)
        return epoch_loss, mAP


def main(model, optimizer, loss_function, train_data_loader, validation_data_loader):
    best_vlidation_mAP = 0
    training_loss = list()
    validation_loss = list()

   # test_model(test_data_loader, loss_fcn=loss_function)
    #exit()

    for epoch in range(NUMBER_EPOCHS):

        train_epoch_loss = one_epoch(model=model, optimizer=optimizer, loss_function=loss_function, data_loader=train_data_loader, epoch=epoch)
        validation_epoch_loss, validation_mAP = one_epoch(model=model, optimizer=optimizer, loss_function=loss_function, data_loader=validation_data_loader, is_training=False, epoch=epoch)

        change_learning_rate(optimizer=optimizer, epoch=epoch)

        training_loss.append(train_epoch_loss)
        validation_loss.append(validation_epoch_loss)

        print("Epoch: {}, Training loss: {}, Validation loss: {}, Validation mAP: {}".format(epoch + 1, train_epoch_loss, validation_epoch_loss, validation_mAP))

        if validation_mAP > best_vlidation_mAP:
            best_vlidation_mAP = validation_mAP
            print("SAVING MODEL...")
            torch.save(model, "{}_saved.pt".format(MODEL_NAME))
            print("MODEL WAS SUCCESSFULLY SAVED!")


if __name__ == "__main__":
    torch.set_printoptions(linewidth=2000, threshold=10000)
    model = DetectionModel(number_classes=NUMBER_CLASSES, grid_size_width=GRID_SIZE_WIDTH, grid_size_height=GRID_SIZE_HEIGHT, chosen_model=CHOSEN_MODEL).to(DEVICE)
    #model = init_model_weights(model)
    #model.load_state_dict(torch.load("/Users/artemmoroz/Desktop/CIIRC_projects/WheelsDetectionModified/MOBILENET_V2_CLASSIFICATION_PRETRAINED.pt"))

    optimizer = torch.optim.Adam(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, params=model.parameters())
    criterion = LossFunction(grid_size=GRID_SIZE_WIDTH, number_classes=NUMBER_CLASSES, anchor_box=ANCHOR_BOX)
    dataset = DetectionDataset(csv_file=CSV_FILE_DETECTION, grid_size_width=GRID_SIZE_WIDTH, grid_size_height=GRID_SIZE_HEIGHT, number_classes=NUMBER_CLASSES,
                               transformations=transformations_detection)

    train_subset, validation_subset = torch.utils.data.random_split(dataset, DATASET_SPLIT)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_subset, batch_size=BATCH_SIZE_TRAINING, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(dataset=validation_subset, batch_size=BATCH_SIZE_TESTING, shuffle=True)

    main(model=model, optimizer=optimizer, loss_function=criterion, train_data_loader=train_data_loader, validation_data_loader=test_data_loader)