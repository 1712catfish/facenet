import matplotlib.pyplot as plt


def show_plot(iteration, loss):
    plt.plot(iteration, loss)
    plt.savefig('loss.png')
    plt.show()


# show_plot(counter, loss_history)
