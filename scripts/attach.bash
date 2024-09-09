#!/bin/bash

sudo killall tmux:\ server

# Start a new tmux session called "my_tpu_session"
tmux new-session -d -s my_tpu_session

# Create the first pane and connect to the first server (worker 0)
tmux send-keys "gcloud compute tpus tpu-vm ssh carrot --zone=us-central2-b --worker=0" C-m

# Split the window horizontally (creates two panes side-by-side)
tmux split-window -h
tmux send-keys "gcloud compute tpus tpu-vm ssh carrot --zone=us-central2-b --worker=1" C-m

# Select the first pane again
tmux select-pane -t 0

# Split the first pane vertically (creates a 2x2 grid)
tmux split-window -v
tmux send-keys "gcloud compute tpus tpu-vm ssh carrot --zone=us-central2-b --worker=2" C-m

# Select the second pane (which was split horizontally)
tmux select-pane -t 1

# Split this pane vertically (completes the 2x2 grid)
tmux split-window -v
tmux send-keys "gcloud compute tpus tpu-vm ssh carrot --zone=us-central2-b --worker=3" C-m

# Finally, attach to the tmux session
tmux select-layout tiled
tmux attach-session -t my_tpu_session
