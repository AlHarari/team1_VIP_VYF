import re
import pickle
import sys

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

if __name__ == "__main__":
    PICKLED_FILE_PATH = "~/scratch/pickled_inputs.pkl"
    print("GETTING minimum loss point and corresponding args.")
    try: 
        with open(PICKLED_FILE_PATH, "rb") as inputs_file:
            k = 5 if len(sys.argv) == 1 else int(sys.argv[1])
            if len(sys.argv) > 2:
                print("Usage: python helper_file.py <OPTIONAL_NUMBER_OF_PAIRS>.")
                sys.exit(1)
            print(f"k={k}")

            args_to_loss_map = pickle.load(inputs_file)
            print(f"Number of unique args: {len(args_to_loss_map)}")
            sorted_args_to_loss = sorted(args_to_loss_map.items(), key=lambda entry: sum(entry[1][-20:])/20)[:k]
            for pair in sorted_args_to_loss:
                print(f"{pair[0]}: {sum(pair[1][-20:])/20}")
    except FileNotFoundError:
        print("There isn't a 'pickled_inputs.pkl' file.") 
