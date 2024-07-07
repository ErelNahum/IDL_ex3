########################################################################
########################################################################
##                                                                    ##
##                      ORIGINAL _ DO NOT PUBLISH                     ##
##                                                                    ##
########################################################################
########################################################################

import os
import torch
import numpy as np
import torch.nn as nn
import loaders.loader as ld

from models.gru import ExGRU
from models.rnn import ExRNN
from models.mlp import ExMLP
from matplotlib import pyplot as plt
from models.restricted_attention import ExRestSelfAtten

batch_size = 32
output_size = 2

run_recurrent = False  # else run Token-wise MLP
use_RNN = False  # otherwise GRU
atten_size = 5  # atten > 0 means using restricted self atten

reload_model = False
num_epochs = 10
learning_rate = 0.001
test_interval = 50

WEIGHTS_DIR_PREFIX = 'weights/'
RESULTS_DIR_PREFIX = 'results/'

# Loading sataset, use toy = True for obtaining a smaller dataset

train_dataset, test_dataset, num_words, input_size = ld.get_data_set(batch_size)


# prints portion of the review (20-30 first words), with the sub-scores each word obtained
# prints also the final scores, the softmaxed prediction values and the true label values

def print_review(rev_text, sbs1, sbs2, lbl1, lbl2):
    print('*********************')
    for i in range(20):
        print(f'Word: {rev_text[i]}, Sub Score: [{sbs1[i]}, {sbs2[i]}]')
    print(f'\nFinal score: [{sbs1.sum()}, {sbs2.mean()}]')
    scores = torch.tensor(np.stack([sbs1, sbs2]))
    softmax_scores = torch.nn.functional.softmax(scores.sum(1), dim=0)
    print(f'Softmaxed prediction: [{softmax_scores[0]}, {softmax_scores[1]}]')
    print(f'True label: [{lbl1}, {lbl2}]')
    print('*********************')


def select_model(hidden_size):
    if run_recurrent:
        if use_RNN:
            model = ExRNN(input_size, output_size, hidden_size)
        else:
            model = ExGRU(input_size, output_size, hidden_size)
    else:
        if atten_size > 0:
            model = ExRestSelfAtten(input_size, output_size, hidden_size, atten_size)
        else:
            model = ExMLP(input_size, output_size, hidden_size)
    print("Using model: " + model.name())
    return model


def load_model(hidden_size):
    model = select_model(hidden_size)
    model_file_path = get_model_weights_path(model.name(), hidden_size)
    if reload_model and os.path.exists(model_file_path):
        print("Reloading model")
        model.load_state_dict(torch.load(model_file_path))
    return model


def run_token_wise_model(model, reviews):
    sub_score = []
    if atten_size > 0:
        # MLP + atten
        sub_score, atten_weights = model(reviews)
    else:
        # MLP
        sub_score = model(reviews)
    output = torch.mean(sub_score, 1)
    return output, sub_score


def get_model_weights_path(model_name, hidden_size):
    return WEIGHTS_DIR_PREFIX + model_name + f"_h{hidden_size}.pth"


def post_test_iter(epoch, itr, labels, reviews_text, sub_score, test_loss, train_loss):
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], "
        f"Step [{itr + 1}/{len(train_dataset)}], "
        f"Train Loss: {train_loss:.4f}, "
        f"Test Loss: {test_loss:.4f}"
    )
    if not run_recurrent:
        nump_subs = sub_score.detach().numpy()
        labels = labels.detach().numpy()
        print_review(reviews_text[0], nump_subs[0, :, 0], nump_subs[0, :, 1], labels[0, 0], labels[0, 1])


def run_recurrent_model(labels, model, reviews):
    output = None
    hidden_state = model.init_hidden(int(labels.shape[0]))
    for i in range(num_words):
        output, hidden_state = model(reviews[:, i, :], hidden_state)  # HIDE
    return output, hidden_state


def epoch_iteration(criterion, epoch, model, optimizer,
                    test_losses, train_losses, sub_score=None):
    correct_train, train_total_samples = 0, 0
    correct_test, test_total_samples = 0, 0
    itr = 0  # iteration counter within each epoch
    train_loss = train_losses[-1]
    for labels, reviews, reviews_text in train_dataset:  # getting training batches

        itr = itr + 1

        if (itr + 1) % test_interval == 0:
            test_iter = True
            labels, reviews, reviews_text = next(iter(test_dataset))  # get a test batch
            test_total_samples += len(labels)
        else:
            test_iter = False
            train_total_samples += len(labels)

        # Recurrent nets (RNN/GRU)
        if run_recurrent:
            output, hidden_state = run_recurrent_model(labels, model, reviews)
        # Token-wise networks (MLP / MLP + Atten.)
        else:
            output, sub_score = run_token_wise_model(model, reviews)

        # cross-entropy loss
        loss = criterion(output, labels)

        # optimize in training iterations
        if not test_iter:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # averaged losses
        if test_iter:
            correct_test += get_correct_amount(labels, output)
            test_losses.append(0.8 * float(loss.detach()) + 0.2 * test_losses[-1])
            train_losses.append(train_loss)
        else:
            correct_train += get_correct_amount(labels, output)
            train_loss = 0.9 * float(loss.detach()) + 0.1 * train_loss

        if test_iter:
            post_test_iter(epoch, itr, labels, reviews_text,
                           sub_score, test_losses[-1], train_losses[-1])
    return correct_train / train_total_samples, correct_test / test_total_samples


def get_correct_amount(labels, output):
    preds = torch.argmax(output, dim=1)
    labels = torch.argmax(labels, dim=1)
    return (labels == preds).sum().item()


def plot_losses(model_name, test_losses, train_losses, hidden_size):
    plt.plot(train_losses, label=f'{model_name} train loss (hidden size = {hidden_size})', color='orange')
    plt.plot(test_losses, label=f'{model_name} test loss (hidden size = {hidden_size})', color='blue')
    plt.title(f"{model_name} loss (hidden size = {hidden_size})")
    plt.xlabel('Test Iteration')
    plt.ylabel('Loss')
    plt.legend(["train", "test"], loc="upper right")
    plt.savefig(RESULTS_DIR_PREFIX + model_name + f"_losses_hs_{hidden_size}.png")
    plt.show()


def plot_accuracy(model_name, test_accuracies, train_accuracies, hidden_size):
    plt.plot(train_accuracies, color='orange')
    plt.plot(test_accuracies, color='blue')
    plt.title(f"{model_name} accuracy (hidden size = {hidden_size})")
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(["train", "test"], loc="lower right")
    plt.savefig(RESULTS_DIR_PREFIX + model_name + f"_accuracy_hs_{hidden_size}.png")
    plt.show()


def plot_graphs(model_name, train_losses, test_losses, train_accuracies, test_accuracies, hidden_size):
    plot_losses(model_name, test_losses, train_losses, hidden_size)
    plot_accuracy(model_name, test_accuracies, train_accuracies, hidden_size)


def main():
    hidden_size = 128  # to experiment with

    model = load_model(hidden_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = [1.0]
    test_losses = [1.0]
    train_accuracies, test_accuracies = [], []

    # training steps in which a test step is executed every test_interval
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}\n---------------------")
        train_accuracy, test_accuracy = epoch_iteration(criterion, epoch, model, optimizer, test_losses, train_losses)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test accuracy: {test_accuracy:.4f},"
              f" Train accuracy: {train_accuracy:.4f}\n")
        test_accuracies.append(test_accuracy)
        train_accuracies.append(train_accuracy)

    # saving the model
    torch.save(model, get_model_weights_path(model.name(), hidden_size))

    plot_graphs(model.name(), train_losses, test_losses, train_accuracies, test_accuracies, hidden_size)


if __name__ == '__main__':
    main()
