ENV_NAME="${1:-"Deep_learning_WSI_tutorial"}"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE"/etc/profile.d/conda.sh

conda create -n "$ENV_NAME" python=3.7 -y
conda activate "$ENV_NAME"

# NB: CUDA 10.2 is older and likely to be compatible with more systems, but CUDA 11.0 is preferable
# See: https://pytorch.org/get-started/previous-versions/
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch -y
conda install matplotlib==3.* pandas==1.* -y
conda install -c conda-forge notebook==6.* ipywidgets==7.* -y
pip install openslide-python==1.*
conda install -c anaconda scipy==1.* scikit-learn==0.23.* tqdm==4.* pixman==0.40.0 -y
