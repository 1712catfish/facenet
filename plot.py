import matplotlib.pyplot as plt


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig('loss.png')
    plt.show()

def plot_history(history):
    plt.plot(history.history['accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# show_plot(counter, loss_history)
