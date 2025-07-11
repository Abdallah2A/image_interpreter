from zenml.steps import step
import matplotlib.pyplot as plt
import os


def plot_loss(history, output_path):
    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_path)
    plt.close()


@step
def evaluation_step(history):
    os.makedirs('reports', exist_ok=True)
    plot_loss(history, 'reports/loss_curve.png')
