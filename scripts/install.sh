sudo apt install neofetch git build-essential -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv -y

wget https://bootstrap.pypa.io/get-pip.py
python3.11 get-pip.py

python3.11 -m pip install -r tpu-requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
