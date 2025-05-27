Hello

# Create environment
We need to use python3.10 or 3.8 to use segmantation-models-pytorch
To ensure we use pytorch with GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.10-1-Linux-x86_64.sh 
./Anaconda3-2024.10-1-Linux-x86_64.sh 
export PATH="$HOME/anaconda3/bin:$PATH"
source ~/.bashrc   # or ~/.zshrc, depending on what you're using
conda --version

conda create -n <your virtual env name> python=3.10 
conda activate <your virtual env name>
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
conda install conda-forge::black
```

# How to run
### Split dataset
`python3 dataset_split.py` 
### Training
`python3 train.py --e 50 --mn model --lr 0.0001 --bs 50 --lossmse --sche 0` 
### Test
`python3 test.py --idx 5 --mn model_name.pth`



# How to add libralies
Add the libraly name to requirements.txt
Then run this: `pip install -r requirements.txt`