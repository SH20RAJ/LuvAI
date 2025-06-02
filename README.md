# LuvAI 💝 - Romantic Chatbot Training Pipeline

Welcome to LuvAI - A comprehensive Flan-T5 based romantic chatbot training system that creates emotionally intelligent conversational AI models specialized in romantic communication.

## 🎯 Project Overview

LuvAI is an advanced machine learning project that fine-tunes Google's Flan-T5 model on curated romantic conversation datasets. The system creates chatbots capable of engaging in meaningful, emotionally aware romantic conversations with bilingual support (English/Hindi-Hinglish).

### Key Features

- 🤖 **Flan-T5 Fine-tuning**: Complete training pipeline for Google's Flan-T5-small model
- 💬 **Romantic Conversations**: Specialized dataset of 300+ romantic chat conversations
- 🌍 **Bilingual Support**: English and Hindi-Hinglish conversation capabilities
- 📊 **Comprehensive Evaluation**: ROUGE score metrics and interactive testing
- 🚀 **Production Ready**: Model saving, loading, and deployment guidelines
- 📱 **Interactive Interface**: Real-time chat testing functionality

## 📂 Project Structure

```
AI/
├── README.md                           # Project documentation
├── luvai_training.ipynb               # Complete training pipeline notebook
├── luvai_training.py                  # Python script version of training pipeline
├── requirements.txt                   # Python dependencies
├── run_examples.py                    # Usage examples and commands
├── prompt.md                          # Dataset generation prompts
├── LOVE/                              # Romantic conversation datasets
│   ├── luvai_chat_data.json          # Original dataset (10 conversations)
│   ├── romantic_chat_01.json         # Extended dataset (50+ conversations)
│   ├── romantic_chat_02.json         # Extended dataset (50+ conversations)
│   ├── romantic_chat_03.json         # Extended dataset (50+ conversations)
│   ├── romantic_chat_04.json         # Extended dataset (50+ conversations)
│   ├── romantic_chat_05.json         # New dataset (50 conversations)
│   ├── romantic_chat_06.json         # New dataset (50 conversations)
│   ├── generate_datasets.py          # Dataset generation scripts
│   ├── generate_romantic_datasets.py # Romantic dataset generator
│   └── simple_generator.py           # Simple conversation generator
├── data/                              # Additional data files
└── .venv/                            # Python virtual environment
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended) or CPU
- 8GB+ RAM (16GB+ recommended for GPU training)
- Virtual environment (recommended)

### Installation

1. **Clone or download the project**:

```bash
cd /path/to/AI/project
```

2. **Create and activate virtual environment**:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**:

```bash
# Install from requirements file (recommended)
pip install -r requirements.txt

# Or install manually
pip install transformers datasets torch accelerate sentencepiece evaluate rouge_score
pip install --upgrade huggingface_hub

# Optional: For faster training
pip install deepspeed
```

### Training Options

#### Option 1: Python Script (Recommended)

```bash
# Basic training
python luvai_training.py --data_path ./LOVE

# Training with custom parameters
python luvai_training.py --data_path ./LOVE --model_name google/flan-t5-base --interactive

# Interactive examples and help
python run_examples.py
```

#### Option 2: Jupyter Notebook

```bash
jupyter notebook luvai_training.ipynb
```

## 📚 Dataset Information

### Dataset Overview

- **Total Conversations**: 300+ romantic chat conversations
- **Format**: JSON files with conversation pairs
- **Languages**: English and Hindi-Hinglish mix
- **Structure**: Each file contains conversation arrays with natural romantic dialogues

### Sample Conversation Format

```json
{
  "conversations": [
    {
      "input": "I've been thinking about you all day",
      "output": "Aww, that's so sweet! You've been on my mind too. What were you thinking about?"
    }
  ]
}
```

### Dataset Files

- `luvai_chat_data.json` - Original romantic conversations (10 pairs)
- `romantic_chat_01.json` - Extended romantic dialogues (50+ pairs)
- `romantic_chat_02.json` - Sweet and caring conversations (50+ pairs)  
- `romantic_chat_03.json` - Playful romantic exchanges (50+ pairs)
- `romantic_chat_04.json` - Deep emotional conversations (50+ pairs)
- `romantic_chat_05.json` - Mixed romantic scenarios (50 pairs)
- `romantic_chat_06.json` - Bilingual romantic chats (50 pairs)

## 🎓 Training Pipeline

The complete training pipeline is available in `luvai_training.ipynb` with the following steps:

### 1. Environment Setup

- Install required dependencies
- Import necessary libraries
- Configure GPU/CPU settings

### 2. Data Loading & Preprocessing

- Load multiple JSON conversation files
- Combine and clean datasets
- Format data for T5 training

### 3. Model Configuration

- Initialize Flan-T5-small model and tokenizer
- Set up training arguments
- Configure data collators

### 4. Training Execution

- Fine-tune the model on romantic conversations
- Monitor training progress
- Apply early stopping

### 5. Evaluation & Testing

- Calculate ROUGE scores
- Interactive chat testing
- Model performance analysis

### 6. Model Persistence

- Save trained model and tokenizer
- Load models for inference
- Export for production deployment

## 🏃‍♂️ Usage Examples

### Python Script Usage

#### Basic Training

```bash
# Train with default settings
python luvai_training.py --data_path ./LOVE

# Train with custom model
python luvai_training.py --data_path ./LOVE --model_name google/flan-t5-base

# Train and start interactive chat
python luvai_training.py --data_path ./LOVE --interactive
```

#### Advanced Options

```bash
# Custom training parameters
python luvai_training.py \
  --data_path ./LOVE \
  --model_name google/flan-t5-base \
  --output_dir ./my_training \
  --save_dir ./my_model \
  --test_size 0.15

# Load existing model for chat only
python luvai_training.py --load_model ./flan-t5-romantic-chat-final
```

#### Interactive Examples

```bash
# Run interactive examples with guided setup
python run_examples.py
```

### Programmatic Usage

#### Basic Training in Code

```python
from luvai_training import RomanticChatTrainer

# Initialize trainer
trainer = RomanticChatTrainer(model_name="google/flan-t5-small")

# Load and process data
conversations = trainer.load_chat_data("./LOVE")
processed_data = trainer.preprocess_conversations(conversations)

# Setup and train
trainer.setup_model_and_tokenizer()
datasets = trainer.create_datasets(processed_data)
tokenized_datasets = trainer.tokenize_datasets(datasets)
trainer.setup_trainer(tokenized_datasets)

# Train and save
trainer.train_model()
trainer.save_model("./my_romantic_model")
```

#### Interactive Chat

```python
# Load trained model for chat
trainer = RomanticChatTrainer()
trainer.interactive_chat("./flan-t5-romantic-chat-final")
```

## 📊 Model Performance

### Training Metrics

- **Model**: google/flan-t5-small (77M parameters)
- **Training Steps**: Configurable (default: 500-1000)
- **Evaluation**: ROUGE-1, ROUGE-2, ROUGE-L scores
- **Batch Size**: 8 (adjustable based on GPU memory)
- **Learning Rate**: 5e-5 with warmup

### Expected Results

- **ROUGE-1**: 0.45+ (measures word overlap)
- **ROUGE-2**: 0.25+ (measures bigram overlap)  
- **ROUGE-L**: 0.40+ (measures longest common subsequence)

## 🚀 Production Deployment

### Model Export

```python
# Save model for production
model.save_pretrained("./production_model")
tokenizer.save_pretrained("./production_model")
```

### Inference API

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

def romantic_chatbot_api(user_input):
    model = T5ForConditionalGeneration.from_pretrained("./production_model")
    tokenizer = T5Tokenizer.from_pretrained("./production_model")
    
    input_text = f"romantic chat: {user_input}"
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=100, temperature=0.8)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response
```

## 🛠️ Customization

### Adding New Datasets

1. Create JSON files in the `LOVE/` directory
2. Follow the conversation format structure
3. Run the data loading functions to include new data

### Hyperparameter Tuning

- Modify `TrainingArguments` in the notebook
- Adjust `max_length`, `temperature`, and `num_beams` for generation
- Experiment with different learning rates and batch sizes

### Model Variants

- Switch to `flan-t5-base` or `flan-t5-large` for better performance
- Adjust training steps based on model size
- Consider using LoRA for parameter-efficient fine-tuning

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add new conversation datasets or improve training pipeline
4. Test your changes thoroughly
5. Submit a pull request

### Dataset Contribution Guidelines

- Follow existing JSON format
- Ensure conversations are appropriate and romantic in nature
- Include diverse scenarios and emotional contexts
- Test conversations for quality and coherence

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google**: For the Flan-T5 model architecture
- **Hugging Face**: For the transformers library and model hosting
- **Contributors**: All dataset creators and testers

## 📞 Support

For questions, issues, or contributions:

- Create an issue in the repository
- Check the training notebook for detailed implementation
- Review dataset formats for proper structure

---

**Made with ❤️ for creating emotionally intelligent AI conversations**
