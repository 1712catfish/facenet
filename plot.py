import matplotlib.pyplot as plt


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
    plt.show(block=True)

    plt.plot(history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show(block=True)

# show_plot(counter, loss_history)
