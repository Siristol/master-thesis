#Plot loss and accuracy curve
import matplotlib.pyplot as plt
import numpy as np
import re
import os


def plot_training_curves(log_file_path, save_plots=False, output_dir="./plots"):
    """
    Plot training loss and accuracy curves from log file.
    
    Args:
        log_file_path (str): Path to the log file
        save_plots (bool): Whether to save plots to disk
        output_dir (str): Directory to save plots
    """
    # Read and parse the log file
    epochs = []
    train_losses = []
    val_losses = []
    accuracies = []
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse each epoch line
    for line in lines:
        if line.startswith('Epoch:'):
            # Extract values using regex
            epoch_match = re.search(r'Epoch:(\d+)', line)
            train_loss_match = re.search(r'Train_loss:([0-9.e-]+)', line)
            val_loss_match = re.search(r'Val_loss:([0-9.e-]+)', line)
            acc_match = re.search(r'Acc:([0-9.e-]+)', line)
            
            if all([epoch_match, train_loss_match, val_loss_match, acc_match]):
                epochs.append(int(epoch_match.group(1)))
                train_losses.append(float(train_loss_match.group(1)))
                val_losses.append(float(val_loss_match.group(1)))
                accuracies.append(float(acc_match.group(1)))
    
    # Extract model name from file path
    model_name = os.path.basename(log_file_path).replace('.txt', '').replace('_log', '')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss curves
    ax1.plot(epochs, train_losses, label='Training Loss', color='blue', linewidth=2)
    ax1.plot(epochs, val_losses, label='Validation Loss', color='red', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Training and Validation Loss - {model_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracy curve
    ax2.plot(epochs, accuracies, label='Validation Accuracy', color='green', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'Validation Accuracy - {model_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Set y-axis limits for better visualization
    ax2.set_ylim(0, max(accuracies) * 1.1)
    
    plt.tight_layout()
    
    # Save plots if requested
    if save_plots:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        plt.savefig(f"{output_dir}/{model_name}_curves.png", dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_dir}/{model_name}_curves.png")
    
    plt.show()
    
    # Print final statistics
    print(f"\nTraining Statistics for {model_name}:")
    print(f"Final Training Loss: {train_losses[-1]:.6f}")
    print(f"Final Validation Loss: {val_losses[-1]:.6f}")
    print(f"Final Accuracy: {accuracies[-1]:.4f}")
    print(f"Best Accuracy: {max(accuracies):.4f} at epoch {epochs[accuracies.index(max(accuracies))]}")


# Example usage
if __name__ == "__main__":
    # Plot curves for a single model
    plot_training_curves("runs/mobilenetv1-coco_log.txt", save_plots=True)
    pass