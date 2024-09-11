sudo mount -t tmpfs -o size=150G tmpfs ~/.cache
cd ~

sudo apt install neofetch git build-essential -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.11 python3.11-venv -y

curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
python3.11 get-pip.py

python3.11 -m pip install -r nano_jax_gpt/tpu-requirements.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html

sudo mount -t tmpfs -o size=100G tmpfs ~/nano_jax_gpt
git clone https://github.com/knyazer/nano_jax_gpt

cd ~/nano_jax_gpt/
python3.11 prepare_openwebtext.py
