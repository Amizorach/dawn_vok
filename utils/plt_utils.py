class PlotUtils:
    def __init__(self):
        self.fig, self.axs = plt.subplots(2, 1)

    def plot_embedding(self, embedding, output, epoch, loss_history):
        self.axs[0].plot(embedding, output)
        self.axs[1].plot(loss_history)
        self.fig.savefig(f"epoch_{epoch}.png")
        plt.close()
