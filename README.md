# Biometric Verification System

Biometric verification system for research and education purposes. This system implements multiple biometric modalities including fingerprint, face, and voice recognition with proper evaluation metrics and anti-spoofing capabilities.

## Features

- **Multiple Biometric Modalities**: Fingerprint, face, and voice recognition
- **Anti-Spoofing Detection**: Liveness detection and spoofing prevention
- **Comprehensive Evaluation**: EER, minDCF, ROC/DET curves, FAR/FRR metrics
- **Interactive Demo**: Streamlit-based web interface for enrollment and verification
- **Privacy-First Design**: PII protection and data anonymization
- **Research-Ready**: Clean code, type hints, comprehensive documentation

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Biometric-Verification-System.git
cd Biometric-Verification-System

# Install dependencies
pip install -e .

# Install development dependencies (optional)
pip install -e ".[dev]"
```

### Basic Usage

```python
from src.models.biometric_verifier import BiometricVerifier
from src.data.synthetic_data import generate_synthetic_dataset

# Initialize the verifier
verifier = BiometricVerifier()

# Generate synthetic biometric data
data = generate_synthetic_dataset(n_samples=1000)

# Train the system
verifier.train(data)

# Verify a biometric sample
result = verifier.verify(sample_biometric_data)
print(f"Verification result: {result}")
```

### Interactive Demo

```bash
# Launch the Streamlit demo
streamlit run demo/app.py
```

## Dataset Schemas

### Biometric Templates
- **Fingerprint**: 128-dimensional feature vectors
- **Face**: 512-dimensional embeddings
- **Voice**: 256-dimensional speaker embeddings

### Data Format
```python
{
    "user_id": "user_001",
    "modality": "fingerprint",  # fingerprint, face, voice
    "template": np.array([...]),  # biometric template
    "metadata": {
        "quality_score": 0.95,
        "timestamp": "2024-01-01T00:00:00Z",
        "device_id": "device_001"
    }
}
```

## Training and Evaluation

### Training
```bash
python scripts/train.py --config configs/fingerprint_config.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --model_path models/fingerprint_model.pkl --test_data data/test/
```

## Metrics and Limitations

### Evaluation Metrics
- **EER (Equal Error Rate)**: Point where FAR = FRR
- **minDCF (Minimum Detection Cost Function)**: Cost-weighted error rate
- **ROC/DET Curves**: Performance visualization
- **FAR/FRR**: False Acceptance/Rejection Rates

### Current Limitations
- Uses synthetic data for demonstration purposes
- Not suitable for production security operations
- Limited to research and educational use cases
- Performance may not reflect real-world biometric systems

## Security and Privacy

### Privacy Safeguards
- All biometric templates are anonymized
- No PII is stored or transmitted
- Synthetic data generation for demonstrations
- Secure template storage and comparison

### Security Disclaimers
- This system is for research and education only
- Not intended for production security operations
- May not provide adequate security for real-world applications
- Users should not rely on this system for actual authentication

## Project Structure

```
biometric-verification-system/
├── src/                    # Source code
│   ├── data/              # Data processing modules
│   ├── features/          # Feature extraction
│   ├── models/            # Biometric models
│   ├── defenses/          # Anti-spoofing defenses
│   ├── eval/              # Evaluation metrics
│   ├── viz/               # Visualization tools
│   └── utils/             # Utility functions
├── data/                  # Data storage
├── configs/               # Configuration files
├── scripts/               # Training/evaluation scripts
├── notebooks/             # Jupyter notebooks
├── tests/                 # Unit tests
├── assets/                # Generated assets
├── demo/                  # Interactive demo
└── README.md              # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This biometric verification system is designed for research and educational purposes only. It is not intended for production security operations or real-world authentication systems. The system uses synthetic data and simplified models that may not reflect the complexity and security requirements of actual biometric systems. Users should not rely on this system for any security-critical applications.
# Biometric-Verification-System
