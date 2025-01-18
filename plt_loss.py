import re
import matplotlib.pyplot as plt

def parse_losses(log_file_path):
    losses = []
    # pattern matching for the avg loss
    pattern = re.compile(r'avg\.loss:\s*([0-9]*\.[0-9]+|[0-9]+)')

    with open(log_file_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            # we'll just strip each line and search using regex.
            line = line.strip()
            match = pattern.search(line)
            if match:
                # extract loss as a float
                loss_value = float(match.group(1))
                losses.append(loss_value)
    
    return losses

def plot_losses(losses, output_image="loss_plot.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss over Time')
    plt.xlabel('Iteration (index of captured loss)')
    plt.ylabel('Average Loss')
    plt.grid(True)
    plt.legend()
    # save the plot
    plt.savefig(output_image)
    plt.show()

if __name__ == "__main__":
    log_file = "training_log.txt"
    losses = parse_losses(log_file)

    if not losses:
        print("No losses found in the log file.")
    else:
        plot_losses(losses)
        print("Loss plot saved as loss_plot.png")

