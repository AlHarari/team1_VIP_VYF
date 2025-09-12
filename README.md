# How to train.
- Start by inputting your arguments into the `input_arguments.txt` file. The format should be such that each row corresponds to a string of arguments to 
train a single model on. More specifically, each row is of the format
LEARNING_RATE DIMENSION WINDOW_SIZE EPOCH MINCOUNT MINN MAXN NEG WORDN.
- Run `. whole_train.sh` on the shell.
