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
```

# How to run
`python3 train.py --e 50 --mn model_name.pth --lr 1e-4` 
`python3 prediction.py --idx 5 --mn model_name.pth`

* idx: A index of input image that you wanna predict
* mn: A model name when you save the model


# How to add libralies
Add the libraly name to requirements.txt
Then run this: `pip install -r requirements.txt`