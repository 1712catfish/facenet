import matplotlib.pyplot as plt

import config


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig('loss.png')
    plt.show()


def plot_history(history):
    plt.plot(history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(config.SAVE_PLOT + '/accuracy.jpg')

    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig(config.SAVE_PLOT + '/loss.jpg')
