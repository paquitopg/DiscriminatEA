# DiscriminatEA

**DiscriminatEA** is a Python package for entity alignment on highly heterogeneous knowledge graphs. This tool implements advanced machine learning techniques to identify corresponding entities across different knowledge graphs, even when they have significantly different structures and characteristics.

## 🚀 Features

- **Entity Alignment**: Identify corresponding entities across heterogeneous knowledge graphs
- **Multiple Embedding Methods**: Support for various knowledge graph embedding techniques
- **Flexible Architecture**: Modular design allowing easy integration of new methods
- **Docker Support**: Containerized deployment options for easy setup
- **Cluster Execution**: Support for SLURM-based cluster computing
- **Comprehensive Evaluation**: Built-in evaluation metrics and alignment scoring

## 📋 Requirements

- Python >= 3.10
- PyTorch (GPU support recommended)
- NumPy, Pandas, NetworkX
- Transformers, SentencePiece
- Gensim, SciPy

## 🛠️ Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/paquitopg/DiscriminatEA.git
cd DiscriminatEA

# Install dependencies using Poetry
poetry install
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/paquitopg/DiscriminatEA.git
cd DiscriminatEA

# Install dependencies
pip install -r requirements.txt
```

## 🏗️ Project Structure

```
discriminatEA/
├── discriminatEA/              # Main package
│   ├── feature_perprocessing/  # Embedding generation modules
│   │   ├── openke/            # OpenKE framework integration
│   │   ├── RDGCN-master/      # RDGCN implementation
│   │   └── longterm/          # Long-term embedding methods
│   ├── main_discriminatEA.py  # Main training script
│   ├── model.py              # Core model implementation
│   ├── utils.py              # Utility functions
│   └── evaluate.py           # Evaluation metrics
├── launchers/                 # Deployment configurations
│   ├── launcher_predict_alignment/
│   ├── launcher_pretrain_model/
│   └── launcher_produce_embeddings/
├── pyproject.toml            # Poetry configuration
└── requirements.txt          # pip dependencies
```

## 🚀 Quick Start

### Basic Usage

```python
from discriminatEA.model import Simple_HHEA
from discriminatEA.main_discriminatEA import train

# Initialize model
model = Simple_HHEA(embedding_dim=300, ...)

# Train the model
train(model, alignment_pairs, dev_alignments, epochs=1500)
```

### Command Line Usage

```bash
# Run main training pipeline
python discriminatEA/main_discriminatEA.py

# Generate embeddings
python discriminatEA/produce_embedding.py

# Predict alignments
python discriminatEA/predict_alignment.py
```

## 🐳 Docker Deployment

The project includes Docker configurations for easy deployment:

```bash
# Build and run with Docker Compose
cd launchers/launcher_predict_alignment/
docker-compose up --build
```

## 🖥️ Cluster Computing

For SLURM-based clusters, use the provided batch scripts:

```bash
# Submit job to cluster
sbatch launchers/launcher_predict_alignment/sbatch.sh
```

## 📊 Supported Datasets

The framework supports various knowledge graph datasets including:
- ICEWS-WIKI
- ICEWS-YAGO
- Airelle
- Custom datasets

## 🔧 Configuration

Key configuration options can be found in:
- `pyproject.toml` - Package dependencies and metadata
- `launchers/` - Deployment-specific configurations
- Model parameters in `discriminate_ea/model.py`

## 📈 Evaluation Metrics

The framework provides comprehensive evaluation including:
- Hit@K accuracy (K=1, 5, 10)
- Mean Reciprocal Rank (MRR)
- CSLS-based similarity scoring

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Paco Goze** - [@paquitopg](https://github.com/paquitopg)

For more detailed documentation and examples, please refer to the individual module documentation and example scripts in the repository.
