rm -rf ~/.cache
mkdir ~/.cache
sudo mount -t tmpfs -o size=200G tmpfs ~/.cache
python3.11 prepare_openwebtext.py
python3.11 train.py
