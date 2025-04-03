import matplotlib.pyplot as plt
from plt_loss import parse_losses
import os


loss_files = list(filter(lambda file_name: "slurm" in file_name, os.listdir("../training_logs/")))
losses = [parse_losses("../training_logs/" + loss_files[i]) for i in range(len(loss_files))]

plt.figure(figsize=(20, 12))
plt.title("(Average) Loss Values From Parameters")
plt.xlabel("Iteration (Proprtional To Epoch #)")
plt.ylabel("(Average) Loss Value")

for i, loss_list in enumerate(losses):
    plt.plot(list(range(len(loss_list))), loss_list, label=loss_files[i])


# plt.plot(list(range(len(group_1_loss))), group_1_loss, label="$E = 250, m = 3, M = 10, s = 14, n_w = 2$")
# plt.plot(list(range(len(group_2_loss))), group_2_loss, label="$E = 250, m = 3, M = 10, s = 14, n_w = 6$")
# plt.plot(list(range(len(group_3_loss))), group_3_loss, label="$E = 350, m = 3, M = 10, s = 20, n_w = 2$")

plt.legend(prop={'size': 10})
plt.savefig("../visualizations/new_compare_plots_slurm.png")

