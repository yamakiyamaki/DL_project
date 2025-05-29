# FACE TO BALL (USING SYNTHETIC DATA)

## Create environment
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

## How to run
### Split dataset
Execute the following command: 
`python3 dataset_split.py --data_dir <path_to_dataset> --train_ratio <float between 0 and 1> --val_ratio <float between 0 and 1> --random_seed <int>`

Inside the dataset folder specified in the command, you should have a subfolder named `renders` with all the renders extracted from blender.

Example:
`python3 dataset_split.py --data_dir ./data/dataset_256px_16f_100im/ --train_ratio 0.75 --val_ratio 0.15 --random_seed 189`

The split will create 3 folders, 1 for training, 1 for testing and 1 for validation, but their contents won't be synced into git.

### Training
`python3 train.py --e 50 --mn model --lr 0.0001 --bs 50 --loss mse --sche 0` 
>Here you can find the training files for two approaches:
>1. `train.py`- Frozen ecoder with imagenet weigts (trains only decoder)
>2. `train_full.py`- Train both encoder and decoder
### Test
`python3 test.py --idx 5 --mn model_name.pth`



## How to add libralies
Add the libraly name to requirements.txt
Then run this: `pip install -r requirements.txt`

## Data generator
To generate the data you can use the blender file [DLCV_data_generator.blend](./DLCV_data_generator.blend). For that, keep in mind that you will need to specify where do you want your renders, by filling in the variable `output_dir`. You should be able to see it in the python code once you open it.

If you want to add or remove faces, just move them from the collection `Faces` to the `NotUsed` (or viceversa).

If you add new faces, keep in mind that they should have the orientation using <ins>X</ins>YZ Euler, not quaternions.

When you import a new face, just keep in mind to move it around to match the ball.

NOTE: Known issue -> once you use a face for rendering, you can no longer see it in the inspector. We don't know why this happens.