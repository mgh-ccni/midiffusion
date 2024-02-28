
# midiffusion

`midiffusion` is a comprehensive framework designed for experimenting with diffusion models. It provides a modular architecture for training, sampling, and evaluating diffusion models on various datasets.

## Features

- **Flexible Configuration**: Customize experiments with a comprehensive set of configuration files.
- **Diverse Datasets Support**: Integrated support for multiple datasets, facilitating easy experimentation.
- **Modular Design**: Separation of concerns between model definitions, data processing, and training routines.
- **Extensible**: Easily add new models, datasets, and sampling methods.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/midiffusion-main.git
cd midiffusion-main
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Training

To start a training session, you can use the provided shell scripts:

```bash
./train_commands.sh
```

For training with specific settings, modify the `train_commands.sh` script or run `main.py` with custom arguments.

### Sampling

To generate samples from a trained model:

```bash
./run_sampling.sh
```

### Testing

To evaluate a model or perform tests:

```bash
python test.py
```

## Project Structure

- `configs/`: Configuration files for training and evaluation.
- `datasets/`: Place your dataset files here or scripts to download datasets.
- `functions/`: Core functionalities including model components and utilities.
- `models/`: Definitions of diffusion models and architectures.
- `runners/`: Scripts for different stages of the experiment lifecycle.
- `utils/`: Utility functions and helpers.

## Contributing

Contributions to `midiffusion` are welcome! Please follow the standard fork and pull request workflow. If you plan to propose a major change, please discuss it in an issue first.

## License

Specify your project's license here.
