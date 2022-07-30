import os

import torch
import torch.optim as optim

### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from torchvision import transforms

from pytorch_metric_learning import distances, losses, miners, reducers, testers
from pytorch_metric_learning.utils.accuracy_calculator import AccuracyCalculator

from climbing_thing import ROOT_DIR
from climbing_thing.metric_learning.dataset import ClassificationDataset


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
from climbing_thing.metric_learning.models import Net


### MNIST code originally from https://github.com/pytorch/examples/blob/master/mnist/main.py ###
def train(model, loss_func, mining_func, device, train_loader, optimizer, epoch, print_freq):
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        embeddings = model(data)
        indices_tuple = mining_func(embeddings, labels)
        loss = loss_func(embeddings, labels, indices_tuple)
        loss.backward()
        optimizer.step()
        if epoch%print_freq and batch_idx == 0:
            print(
                "Epoch {} Iteration {}: Loss = {}, Number of mined triplets = {}".format(
                    epoch, batch_idx, loss, mining_func.num_triplets
                )
            )


### convenient function from pytorch-metric-learning ###
def get_all_embeddings(dataset, model):
    tester = testers.BaseTester()
    return tester.get_all_embeddings(dataset, model)


### compute accuracy using AccuracyCalculator from pytorch-metric-learning ###
def test(train_set, test_set, model, accuracy_calculator):
    train_embeddings, train_labels = get_all_embeddings(train_set, model)
    test_embeddings, test_labels = get_all_embeddings(test_set, model)
    train_labels = train_labels.squeeze(1)
    test_labels = test_labels.squeeze(1)
    print("Computing accuracy")
    accuracies = accuracy_calculator.get_accuracy(
        test_embeddings, train_embeddings, test_labels, train_labels, False
    )
    print("Test set accuracy (Precision@1) = {}".format(accuracies["precision_at_1"]))

if __name__ == "__main__":

    device = torch.device("cuda")

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.Normalize(mean=[0.2617867887020111, 0.24093492329120636, 0.21575430035591125], std=[0.22014980018138885, 0.20301464200019836, 0.1867351531982422]),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
    ])

    image_dir = os.path.join(ROOT_DIR, "data/instance_images/test2_masked/")
    label_path = os.path.join(image_dir, "test2_annotations.csv")
    dataset1 = ClassificationDataset(label_path, image_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=8, shuffle=True)
    # test_loader = torch.utils.data.DataLoader(dataset2, batch_size=256)
    test_loader = train_loader


    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 250


    ### pytorch-metric-learning stuff ###
    distance = distances.CosineSimilarity()
    reducer = reducers.ThresholdReducer(low=0)
    loss_func = losses.TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    mining_func = miners.TripletMarginMiner(
        margin=0.2, distance=distance, type_of_triplets="semihard"
    )
    accuracy_calculator = AccuracyCalculator(include=("precision_at_1",), k=1)
    ### pytorch-metric-learning stuff ###


    for epoch in range(1, num_epochs + 1):
        print_loss = epoch%20 == 0
        train(model, loss_func, mining_func, device, train_loader, optimizer, epoch, print_freq=20)
        # test(dataset1, dataset1, model, accuracy_calculator)
    weights_path = os.path.join(ROOT_DIR, "metric_learning/weights.pth")
    torch.save(model.state_dict(), weights_path)
