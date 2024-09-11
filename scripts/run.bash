sudo mount -t tmpfs -o size=300G tmpfs ~
python3.11 prepare_openwebtext.py
python3.11 train.py
