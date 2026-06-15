<div align="center">

# NIV: Neural Axis Variations for Variable Font Generation

Nadav Benedek, Ariel Shamir, Ohad Fried

<br>

<a href="https://arxiv.org/abs/2606.05261"><strong>Paper</strong></a>
&nbsp;|&nbsp;
<a href="https://ndvbd.github.io/NIV/"><strong>Project Page</strong></a>

</div>

---

This repository is the official implementation of: NIV: Neural Axis Variations for Variable Font Generation.

## Structure

- `environment.yml`: Conda environment specification.
- `code/`: Python source files for training and generation.
- `model/`: Trained model checkpoint folder (used by `ttf_to_var.py`)
- `dataset_sample/`: Sample of the prepared dataset (split files, dataset license, and a small set of XML files).
  To generate the full dataset, run the two commands in **Quick Start**:
  1. `data_get_variable_google_fonts.py`
  2. `data_prepare_training_data.py`

## Environment Setup

Set up the conda environment. This may take a few minutes.

```bash
conda env create -f environment.yml
conda activate niv_env
```


## Quick Start


### Download variable fonts

Use this script to clone the Google Fonts repository into google-fonts subdirectory, and extract only variable-font TTF files into google-fonts-variable. This takes a few minutes.

```bash
python code/data_get_variable_google_fonts.py
```

### Prepare training data

Use this step to convert and preprocess a folder of `.ttf` fonts into XML training files in the dataset folder. This should take around 15 minutes.

Example:

```bash
python code/data_prepare_training_data.py \
  --fonts-dir google-fonts-variable \
  --out-dir dataset \
  --skip-composites \
  --normalize-upm \
  --only-axes wght,wdth,slnt,opsz
```

Alternatively, you can download the prepared dataset from Hugging Face:

```bash
pip install -U huggingface_hub
hf download ndvb/NIV \
  --repo-type dataset \
  --local-dir dataset
```

### Training

Train the model using the dataset directory and the predefined train/test split which is specified in dataset_sample for consistent results. The default argument for best performance is to load dataset into RAM (around 50GB of RAM).
You can skip this step and use a trained model.

```bash
python code/niv_model_train.py \
  --data-dir dataset \
  --fixed-split-dir dataset_sample \
  --seed 42 \
  --output-dir model_output \
  --train-loss mse \
  --eval-loss mse \
  --split-method font \
  --eval-steps 20000 \
  --only-axes wght,wdth,slnt,opsz \
  --epochs 80
```
### Evaluation

Evaluate on font-split:

```bash
python code/niv_model_train.py \
  --data-dir dataset \
  --fixed-split-dir dataset_sample \
  --load-checkpoint model \
  --seed 42 \
  --train-loss mse \
  --eval-loss mse \
  --split-method font \
  --only-axes wght,wdth,slnt,opsz \
  --only-eval
```


### Generate a variable font

```bash
python code/ttf_to_var.py /path/to/font.ttf --model /path/to/model/best --axes wght,wdth,slnt,opsz --unicode-range 0x20-0x7E --delta-level 4
```

Output is saved next to the source font as `*_var.ttf`.
You can use a model you trained, or take a (font-split) trained model from `model/`.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{benedek2026niv,
  title={NIV: Neural Axis Variations for Variable Font Generation},
  author={Benedek, Nadav and Shamir, Ariel and Fried, Ohad},
  journal={arXiv preprint arXiv:2606.05261},
  year={2026}
}
```
