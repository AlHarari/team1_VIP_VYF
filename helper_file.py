import re
# import matplotlib.pyplot as plt

def get_arguments(system_arguments):
    try:
        with open("input_arguments.txt", "r") as f:
            script_name = system_arguments[0]
            if script_name == "train.py":
                task_id = int(system_arguments[1])
                line = f.readlines()[task_id].split()
                lr = float(line[0])
                args = [lr] + list(map(int, line[1:]))
                return args
            elif script_name == "after_train.py": 
                lines = f.readlines()
                args_list = []
                for line in lines:
                    split_arguments = line.split()
                    args_list.append(
                            [float(split_arguments[0])] +        # lr
                            list(map(int, split_arguments[1:]))  # rest of the arguments
                    )
                return args_list
            else:
                raise Exception("Wrong file called 'get_arguments' function.")
    except FileNotFoundError:
        print("No 'input_arguments.txt' file found.")


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

#def plot_losses(losses, output_image="loss_plot.png"):
#    plt.figure(figsize=(10, 6))
#    plt.plot(losses, label='Training Loss')
#    plt.title('Training Loss over Time')
#    plt.xlabel('Iteration (index of captured loss)')
#    plt.ylabel('Average Loss')
#    plt.grid(True)
#    plt.legend()
#    # save the plot
#    plt.savefig(output_image)
#    plt.show()

#if __name__ == "__main__":
#    log_file = "training_log.txt"
#    losses = parse_losses(log_file)

#    if not losses:
#        print("No losses found in the log file.")
#    else:
#        plot_losses(losses)
#        print("Loss plot saved as loss_plot.png")

