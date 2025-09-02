# Image Generation with Variational Autoencoders on CelebA

This project focuses on training a VAE and VQVAE on the CelebA dataset to generate high-fidelity images of human faces.

The project includes scripts for:
- Training the VQ-VAE and VAE models.
- Evaluating the models using reconstruction metrics (PSNR, MSE) and perceptual metrics (Fréchet Inception Distance - FID).
- Generating new images from the learned latent space.

## Models

Two main models are implemented in this project:

### VQ-VAE (`models/vqvae.py`)
The VQ-VAE consists of:
- **Encoder**: A convolutional neural network that maps input images to a lower-dimensional latent representation. It uses residual blocks.
- **VectorQuantizer**: This layer maps the continuous output of the encoder to a discrete set of embedding vectors from a codebook.
- **Decoder**: A transposed convolutional neural network that reconstructs the image from the quantized latent representation.

### VAE (`models/vae.py`)
A standard Variational Autoencoder is also provided. It consists of:
- **Encoder**: Maps the input image to the parameters (mean and log-variance) of a Gaussian distribution in the latent space.
- **Decoder**: Reconstructs the image from a sample of the latent distribution.
- It uses the reparameterization trick for training.

## Dataset

The model is trained on the **CelebA dataset**, which is a large-scale face attributes dataset with more than 200K celebrity images.

## Dependencies

The project uses Python and several deep learning libraries. You can install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

The main libraries used are:
- PyTorch
- Torchvision
- Matplotlib
- NumPy
- SciPy
- pytorch-fid (for FID calculation)
- tqdm

## Usage

### Training

To train the VQ-VAE model, you can run the `training_vqvae.py` script.

```bash
python training_vqvae.py --data_root <path_to_celeba> --steps 200000 --batch 64 --lr 1e-4 --out checkpoints
```

You can customize training parameters such as batch size, learning rate, and number of training steps via command-line arguments.

### Evaluation

After training, you can evaluate the model using `Evaluations/eval.py`. This script can generate reconstructions, calculate PSNR/MSE, and compute the FID score.

```bash
python Evaluations/eval.py \
    --model_path checkpoints/vqvae_final.pt \
    --data_root <path_to_celeba> \
    --batch 64 \
    --calculate_fid
```

This will save reconstruction grids, generated samples, and evaluation results in the `evaluation_results` directory.

## HPC System Usage

The evaluation process, especially the FID calculation, can be computationally intensive. The project includes an example script (`Evaluations/eval.csh`) for running evaluation jobs on a High-Performance Computing (HPC) cluster that uses the SLURM workload manager.

The script contains `#SBATCH` directives to configure the job resources:
```csh
#SBATCH -J eval_vqvae          # Job name
#SBATCH --time=12:00:00        # Time limit
#SBATCH --nodes=1              # Number of nodes
#SBATCH --gres=gpu:1           # Number of GPUs
#SBATCH --mem=80G              # Memory
#SBATCH --cpus-per-task=16     # Number of CPUs
```

To submit the job, you would typically use a command like:
```bash
sbatch Evaluations/eval.csh
```
The script activates the project's virtual environment and then executes the `eval.py` script with the appropriate parameters. This setup is suitable for running long-running experiments on a shared computing cluster. Training can be done in a similar fashion.
