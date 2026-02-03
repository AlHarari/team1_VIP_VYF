import matplotlib.pyplot as plt
from helper_file import parse_losses, get_arguments
import os
import sys # This is so we could get the job_id to not have to worry about naming the plot.
import pickle

PICKLE_FILE_PATH = "/storage/ice-shared/vip-vyf/embeddings_team/pickled_inputs.pkl"

if len(sys.argv) != 2:
    print("Usage: python after_train.py <JOB_ID>.")
    sys.exit(1)

# Collect different arguments used by reading the input_arguments.txt file.
args_list = get_arguments(sys.argv)
ARRAY_JOB_ID = sys.argv[1]

# Collect all loss files. Hopefully, we'll be able to properly name our loss files so as to preserve order.
loss_files = sorted(list(filter(lambda file_name: "slurm" in file_name, os.listdir("../training_logs/")))) # sorted() to keep track of order. Won't have affected visualizations, but would've probably affected pickled file.
losses = [parse_losses("../training_logs/" + file_name) for file_name in loss_files]

if len(losses) != len(args_list):
    raise Exception("Number of argument tuples != loss files generated. Can't create correspondence.")

# Now, we assume that the first filename in loss_files corresponds to the model trained on the first tuple of arguments in input_arguments, and so on.
if not os.path.exists(PICKLE_FILE_PATH):
    # Create dict and dump.
    from_args_to_loss_curve = {tuple(args): loss for args, loss in zip(args_list, losses)}
    with open(PICKLE_FILE_PATH, "wb") as f:
        pickle.dump(from_args_to_loss_curve, f)
    print("CREATED NEW PICKLE FILE")
else:
    # Load up dict to update it then dump it back.
    with open(PICKLE_FILE_PATH, "rb") as f:
        from_args_to_loss_curve = pickle.load(f)
    # Assuming new argument tuples were used.
    for i, args in enumerate(args_list):
        from_args_to_loss_curve[tuple(args)] = losses[i]
    with open(PICKLE_FILE_PATH, "wb") as f:
        pickle.dump(from_args_to_loss_curve, f)
    print("UPDATED PICKLE FILE")

# Plot
plt.figure(figsize=(20, 12))
plt.title(f"(Average) Loss Values From Parameters of Job_{ARRAY_JOB_ID}")
plt.xlabel("Iteration (Proportional To Epoch #)")
plt.ylabel("(Average) Loss Value")

for i, loss_list in enumerate(losses):
    plt.plot(list(range(len(loss_list))), loss_list, label=loss_files[i])

plt.legend(prop={'size': 10})
plt.savefig(f"/storage/ice-shared/vip-vyf/embeddings_team/models/array_models_{ARRAY_JOB_ID}/compare_plots_{ARRAY_JOB_ID}.png") # 

# Delete all loss files after we've saved them to the from_args_to_loss_curve dict
# for file_name in loss_files:
#    os.remove("../training_logs/" + file_name)

# Delete files in Report folder as well
for file_name in os.listdir("../Reports/"):
    os.remove("../Reports/" + file_name)
