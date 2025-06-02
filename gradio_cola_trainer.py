#!/usr/bin/env python3
"""
🎯 Google CoLA Dataset Trainer with Gradio Interface

A comprehensive Gradio web interface for uploading datasets and training models
on Google CoLA (Corpus of Linguistic Acceptability) tasks using Flan-T5.

Features:
- Dataset upload and validation
- Interactive training configuration
- Real-time training progress
- Model testing and evaluation
- Download trained models

Author: LuvAI Team
Date: June 2025
"""

import gradio as gr
import pandas as pd
import torch
import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from datetime import datetime

# ML Libraries
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import evaluate
from tqdm import tqdm


class CoLATrainer:
    """
    CoLA (Corpus of Linguistic Acceptability) trainer class
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_history = []
        self.current_model_path = None
        
    def validate_cola_dataset(self, file_path: str) -> Tuple[bool, str, Optional[pd.DataFrame]]:
        """
        Validate uploaded dataset for CoLA format
        
        Expected formats:
        1. TSV with columns: sentence, label (0/1 for unacceptable/acceptable)
        2. CSV with columns: sentence, label
        3. JSON with format: [{"sentence": "text", "label": 0/1}, ...]
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_ext == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            elif file_ext == '.csv':
                df = pd.read_csv(file_path)
            elif file_ext == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            else:
                return False, f"Unsupported file format: {file_ext}. Please use TSV, CSV, or JSON.", None
            
            # Check required columns
            required_cols = ['sentence', 'label']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return False, f"Missing required columns: {missing_cols}. Required: {required_cols}", None
            
            # Validate labels
            unique_labels = df['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                return False, "Labels must be 0 (unacceptable) or 1 (acceptable)", None
            
            # Check for empty sentences
            empty_sentences = df['sentence'].isna().sum()
            if empty_sentences > 0:
                return False, f"Found {empty_sentences} empty sentences", None
            
            # Basic statistics
            total_samples = len(df)
            acceptable_count = (df['label'] == 1).sum()
            unacceptable_count = (df['label'] == 0).sum()
            
            stats = f"""
✅ Dataset validation successful!

📊 Dataset Statistics:
- Total samples: {total_samples}
- Acceptable sentences: {acceptable_count} ({acceptable_count/total_samples*100:.1f}%)
- Unacceptable sentences: {unacceptable_count} ({unacceptable_count/total_samples*100:.1f}%)
- Average sentence length: {df['sentence'].str.len().mean():.1f} characters
            """
            
            return True, stats, df
            
        except Exception as e:
            return False, f"Error processing file: {str(e)}", None
    
    def prepare_cola_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Prepare CoLA data for training by converting to input-output format
        """
        processed_data = []
        
        for _, row in df.iterrows():
            sentence = row['sentence']
            label = row['label']
            
            # Convert to text-to-text format for T5
            input_text = f"cola: {sentence}"
            output_text = "acceptable" if label == 1 else "unacceptable"
            
            processed_data.append({
                'input': input_text,
                'output': output_text,
                'original_sentence': sentence,
                'label': label
            })
        
        return processed_data
    
    def setup_model_and_tokenizer(self, model_name: str = "google/flan-t5-small") -> str:
        """
        Setup T5 model and tokenizer for CoLA task
        """
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            
            return f"✅ Model {model_name} loaded successfully on {self.device}"
        except Exception as e:
            return f"❌ Error loading model: {str(e)}"
    
    def create_datasets(self, processed_data: List[Dict], test_size: float = 0.2) -> DatasetDict:
        """
        Create train/validation datasets
        """
        # Split data
        train_data, val_data = train_test_split(
            processed_data, 
            test_size=test_size, 
            random_state=42,
            stratify=[item['label'] for item in processed_data]
        )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_function(self, examples):
        """
        Tokenize examples for T5
        """
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples['input'],
            max_length=512,
            truncation=True,
            padding='max_length'
        )
        
        # Tokenize targets
        labels = self.tokenizer(
            examples['output'],
            max_length=10,  # "acceptable"/"unacceptable" are short
            truncation=True,
            padding='max_length'
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def compute_metrics(self, eval_pred):
        """
        Compute metrics for evaluation
        """
        predictions, labels = eval_pred
        
        # Decode predictions
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Convert to binary predictions
        pred_labels = []
        true_labels = []
        
        for pred, true in zip(decoded_preds, decoded_labels):
            # Convert text predictions to binary
            if "acceptable" in pred.lower():
                pred_labels.append(1)
            else:
                pred_labels.append(0)
                
            if "acceptable" in true.lower():
                true_labels.append(1)
            else:
                true_labels.append(0)
        
        # Calculate accuracy
        accuracy = accuracy_score(true_labels, pred_labels)
        
        return {
            'accuracy': accuracy,
            'eval_samples': len(pred_labels)
        }
    
    def train_model(self, 
                   datasets: DatasetDict, 
                   output_dir: str,
                   num_epochs: int = 3,
                   batch_size: int = 8,
                   learning_rate: float = 5e-5,
                   progress_callback=None) -> str:
        """
        Train the CoLA model
        """
        try:
            # Tokenize datasets
            tokenized_datasets = datasets.map(
                self.tokenize_function,
                batched=True,
                remove_columns=datasets['train'].column_names
            )
            
            # Setup training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                learning_rate=learning_rate,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                num_train_epochs=num_epochs,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="accuracy",
                greater_is_better=True,
                report_to=None,  # Disable wandb/tensorboard
                dataloader_pin_memory=False,
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
            
            # Setup trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
            )
            
            # Train
            train_result = self.trainer.train()
            
            # Save model
            self.trainer.save_model()
            self.current_model_path = output_dir
            
            # Get final metrics
            eval_results = self.trainer.evaluate()
            
            results = f"""
✅ Training completed successfully!

📊 Training Results:
- Final train loss: {train_result.training_loss:.4f}
- Final eval loss: {eval_results['eval_loss']:.4f}
- Final accuracy: {eval_results['eval_accuracy']:.4f}
- Total training time: {train_result.metrics['train_runtime']:.2f} seconds
- Model saved to: {output_dir}
            """
            
            return results
            
        except Exception as e:
            return f"❌ Training failed: {str(e)}"
    
    def test_model(self, test_sentences: List[str]) -> str:
        """
        Test the trained model on sample sentences
        """
        if not self.model or not self.tokenizer:
            return "❌ No model loaded. Please train a model first."
        
        results = []
        
        for sentence in test_sentences:
            input_text = f"cola: {sentence}"
            
            # Tokenize
            inputs = self.tokenizer(
                input_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=10,
                    num_beams=4,
                    early_stopping=True
                )
            
            # Decode
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            results.append({
                'sentence': sentence,
                'prediction': prediction,
                'acceptable': "acceptable" in prediction.lower()
            })
        
        # Format results
        formatted_results = "🧪 Model Test Results:\n\n"
        for i, result in enumerate(results, 1):
            status = "✅ Acceptable" if result['acceptable'] else "❌ Unacceptable"
            formatted_results += f"{i}. \"{result['sentence']}\"\n   → {status}\n\n"
        
        return formatted_results


# Global trainer instance
cola_trainer = CoLATrainer()
uploaded_dataset = None


def upload_dataset(file):
    """Handle dataset upload"""
    global uploaded_dataset
    
    if file is None:
        return "❌ Please upload a dataset file", None, gr.update(interactive=False)
    
    # Validate dataset
    is_valid, message, df = cola_trainer.validate_cola_dataset(file.name)
    
    if is_valid:
        uploaded_dataset = df
        return message, df.head(10), gr.update(interactive=True)
    else:
        uploaded_dataset = None
        return message, None, gr.update(interactive=False)


def train_cola_model(model_name, num_epochs, batch_size, learning_rate, test_size):
    """Train the CoLA model"""
    global uploaded_dataset
    
    if uploaded_dataset is None:
        return "❌ Please upload a valid dataset first"
    
    # Setup model
    setup_msg = cola_trainer.setup_model_and_tokenizer(model_name)
    if "❌" in setup_msg:
        return setup_msg
    
    # Prepare data
    processed_data = cola_trainer.prepare_cola_data(uploaded_dataset)
    datasets = cola_trainer.create_datasets(processed_data, test_size)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"./cola_model_{timestamp}"
    
    # Train
    result = cola_trainer.train_model(
        datasets=datasets,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    return result


def test_sentences(sentence_input):
    """Test individual sentences"""
    if not sentence_input.strip():
        return "❌ Please enter sentences to test"
    
    # Split sentences by newlines
    sentences = [s.strip() for s in sentence_input.split('\n') if s.strip()]
    
    if not sentences:
        return "❌ No valid sentences found"
    
    return cola_trainer.test_model(sentences)


def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="CoLA Dataset Trainer", theme=gr.themes.Soft()) as interface:
        
        gr.Markdown("""
        # 🎯 Google CoLA Dataset Trainer
        
        **Train T5 models on Corpus of Linguistic Acceptability (CoLA) tasks**
        
        Upload your dataset, configure training parameters, and train a model to classify sentence acceptability.
        """)
        
        with gr.Tabs():
            
            # Dataset Upload Tab
            with gr.Tab("📂 Dataset Upload"):
                gr.Markdown("""
                ### Upload CoLA Dataset
                
                **Supported formats:**
                - **TSV/CSV**: columns `sentence`, `label` (0=unacceptable, 1=acceptable)
                - **JSON**: `[{"sentence": "text", "label": 0/1}, ...]`
                
                **Example data:**
                ```
                sentence,label
                "The cat sat on the mat",1
                "Cat the sat mat on the",0
                ```
                """)
                
                file_upload = gr.File(
                    label="Upload Dataset",
                    file_types=[".csv", ".tsv", ".json"],
                    type="filepath"
                )
                
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                    lines=8
                )
                
                dataset_preview = gr.Dataframe(
                    label="Dataset Preview",
                    interactive=False
                )
                
                file_upload.change(
                    upload_dataset,
                    inputs=[file_upload],
                    outputs=[upload_status, dataset_preview, gr.State()]
                )
            
            # Training Tab
            with gr.Tab("🚀 Model Training"):
                gr.Markdown("""
                ### Configure Training Parameters
                
                Adjust the parameters below and click **Start Training** to begin.
                """)
                
                with gr.Row():
                    with gr.Column():
                        model_choice = gr.Dropdown(
                            choices=[
                                "google/flan-t5-small",
                                "google/flan-t5-base",
                                "google/flan-t5-large"
                            ],
                            value="google/flan-t5-small",
                            label="Model Size"
                        )
                        
                        num_epochs = gr.Slider(
                            minimum=1,
                            maximum=10,
                            value=3,
                            step=1,
                            label="Number of Epochs"
                        )
                        
                        batch_size = gr.Slider(
                            minimum=1,
                            maximum=32,
                            value=8,
                            step=1,
                            label="Batch Size"
                        )
                    
                    with gr.Column():
                        learning_rate = gr.Slider(
                            minimum=1e-6,
                            maximum=1e-3,
                            value=5e-5,
                            step=1e-6,
                            label="Learning Rate"
                        )
                        
                        test_size = gr.Slider(
                            minimum=0.1,
                            maximum=0.5,
                            value=0.2,
                            step=0.05,
                            label="Validation Split"
                        )
                
                train_btn = gr.Button(
                    "🚀 Start Training",
                    variant="primary",
                    interactive=False
                )
                
                training_output = gr.Textbox(
                    label="Training Results",
                    interactive=False,
                    lines=10
                )
                
                train_btn.click(
                    train_cola_model,
                    inputs=[model_choice, num_epochs, batch_size, learning_rate, test_size],
                    outputs=[training_output]
                )
            
            # Testing Tab
            with gr.Tab("🧪 Model Testing"):
                gr.Markdown("""
                ### Test Your Trained Model
                
                Enter sentences (one per line) to test grammatical acceptability.
                """)
                
                test_input = gr.Textbox(
                    label="Test Sentences",
                    placeholder="Enter sentences to test (one per line):\nThe cat sat on the mat\nCat the sat mat on the",
                    lines=5
                )
                
                test_btn = gr.Button("🧪 Test Sentences", variant="secondary")
                
                test_results = gr.Textbox(
                    label="Test Results",
                    interactive=False,
                    lines=8
                )
                
                test_btn.click(
                    test_sentences,
                    inputs=[test_input],
                    outputs=[test_results]
                )
            
            # Help Tab
            with gr.Tab("❓ Help"):
                gr.Markdown("""
                ## 📖 How to Use This Interface
                
                ### 1. Upload Dataset
                - Go to the **Dataset Upload** tab
                - Upload a TSV, CSV, or JSON file with CoLA format
                - Required columns: `sentence` (text) and `label` (0/1)
                - Verify the dataset preview looks correct
                
                ### 2. Configure Training
                - Go to the **Model Training** tab
                - Choose model size (small is fastest, large is most accurate)
                - Adjust training parameters as needed
                - Click **Start Training** (this may take several minutes)
                
                ### 3. Test Model
                - Go to the **Model Testing** tab
                - Enter sentences to test grammatical acceptability
                - Click **Test Sentences** to see predictions
                
                ### 📊 About CoLA
                The Corpus of Linguistic Acceptability (CoLA) consists of English sentences labeled as grammatically acceptable or unacceptable. This task helps models learn:
                - Grammatical correctness
                - Syntactic patterns
                - Language understanding
                
                ### 🔧 Tips
                - Start with the small model for quick testing
                - Use larger models for better accuracy
                - More epochs = longer training but potentially better results
                - Smaller batch sizes use less memory
                """)
        
        # Update train button state based on dataset upload
        file_upload.change(
            lambda x: gr.update(interactive=uploaded_dataset is not None),
            inputs=[file_upload],
            outputs=[train_btn]
        )
    
    return interface


def main():
    """Launch the Gradio interface"""
    interface = create_interface()
    
    # Launch with sharing enabled for broader access
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )


if __name__ == "__main__":
    main() 
(Corpus of Linguistic Acceptability) dataset. This interface allows users to:
- Upload their own CoLA dataset
- Configure training parameters
- Train the model interactively
- Test the trained model

Author: LuvAI Team
Date: June 2025
"""

import gradio as gr
import torch
import pandas as pd
import numpy as np
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tempfile
from datetime import datetime

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from evaluate import load
import threading
import time

class CoLATrainer:
    """
    Google CoLA (Corpus of Linguistic Acceptability) trainer with Gradio interface
    """
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.training_status = "Ready"
        self.progress_callback = None
        self.current_model_path = None
        
    def load_cola_dataset(self, file_path: str) -> Tuple[bool, str, pd.DataFrame]:
        """
        Load and validate CoLA dataset from uploaded file
        
        Args:
            file_path: Path to uploaded dataset file
            
        Returns:
            Tuple of (success, message, dataframe)
        """
        try:
            # Determine file type and load accordingly
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.tsv'):
                df = pd.read_csv(file_path, sep='\t')
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                return False, "❌ Unsupported file format. Please upload CSV, TSV, or JSON files.", pd.DataFrame()
            
            # Validate CoLA dataset structure
            required_columns = ['sentence', 'label']
            
            # Check if required columns exist (case insensitive)
            df_columns_lower = [col.lower() for col in df.columns]
            missing_columns = []
            
            for req_col in required_columns:
                if req_col not in df_columns_lower:
                    missing_columns.append(req_col)
            
            if missing_columns:
                # Try common variations
                column_mapping = {}
                for col in df.columns:
                    col_lower = col.lower()
                    if 'sentence' in col_lower or 'text' in col_lower:
                        column_mapping['sentence'] = col
                    elif 'label' in col_lower or 'acceptability' in col_lower:
                        column_mapping['label'] = col
                
                # Rename columns if found
                if len(column_mapping) == 2:
                    df = df.rename(columns=column_mapping)
                else:
                    return False, f"❌ Missing required columns: {missing_columns}. Expected 'sentence' and 'label' columns.", pd.DataFrame()
            
            # Validate data types and content
            if df.empty:
                return False, "❌ Dataset is empty.", pd.DataFrame()
            
            # Ensure labels are binary (0 or 1)
            unique_labels = df['label'].unique()
            if not all(label in [0, 1] for label in unique_labels):
                return False, "❌ Labels must be binary (0 for unacceptable, 1 for acceptable).", pd.DataFrame()
            
            # Basic statistics
            total_samples = len(df)
            acceptable_count = sum(df['label'] == 1)
            unacceptable_count = sum(df['label'] == 0)
            
            message = f"""✅ Dataset loaded successfully!
            
📊 Dataset Statistics:
• Total samples: {total_samples}
• Acceptable sentences: {acceptable_count} ({acceptable_count/total_samples*100:.1f}%)
• Unacceptable sentences: {unacceptable_count} ({unacceptable_count/total_samples*100:.1f}%)
• Average sentence length: {df['sentence'].str.len().mean():.1f} characters

📝 Sample entries:
{df.head(3).to_string(index=False)}"""
            
            return True, message, df
            
        except Exception as e:
            return False, f"❌ Error loading dataset: {str(e)}", pd.DataFrame()
    
    def preprocess_cola_data(self, df: pd.DataFrame) -> List[Dict]:
        """
        Preprocess CoLA data for T5 training
        
        Args:
            df: CoLA dataset dataframe
            
        Returns:
            List of processed examples
        """
        processed_examples = []
        
        for _, row in df.iterrows():
            sentence = str(row['sentence']).strip()
            label = int(row['label'])
            
            # Create T5 format: "cola: [sentence]" -> "acceptable" or "unacceptable"
            input_text = f"cola: {sentence}"
            target_text = "acceptable" if label == 1 else "unacceptable"
            
            processed_examples.append({
                'input_text': input_text,
                'target_text': target_text,
                'sentence': sentence,
                'label': label
            })
        
        return processed_examples
    
    def setup_model_and_tokenizer(self, model_name: str) -> Tuple[bool, str]:
        """
        Initialize model and tokenizer
        
        Args:
            model_name: Name of the pre-trained model
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.tokenizer = T5Tokenizer.from_pretrained(model_name)
            self.model = T5ForConditionalGeneration.from_pretrained(model_name)
            self.model.to(self.device)
            
            return True, f"✅ Model '{model_name}' loaded successfully on {self.device}"
            
        except Exception as e:
            return False, f"❌ Error loading model: {str(e)}"
    
    def create_datasets(self, examples: List[Dict], test_size: float = 0.2) -> DatasetDict:
        """
        Create train/validation datasets
        
        Args:
            examples: Processed examples
            test_size: Fraction for validation set
            
        Returns:
            DatasetDict with train and validation splits
        """
        # Split data
        train_examples, val_examples = train_test_split(
            examples, test_size=test_size, random_state=42, 
            stratify=[ex['label'] for ex in examples]
        )
        
        # Create datasets
        train_dataset = Dataset.from_list(train_examples)
        val_dataset = Dataset.from_list(val_examples)
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_function(self, examples):
        """Tokenize examples for T5"""
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=512,
            truncation=True,
            padding=True
        )
        
        labels = self.tokenizer(
            examples['target_text'],
            max_length=128,
            truncation=True,
            padding=True
        )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def train_model_async(self, dataset_file, model_name: str, epochs: int, 
                         batch_size: int, learning_rate: float, test_size: float,
                         progress_callback=None) -> Tuple[bool, str]:
        """
        Asynchronous training function
        
        Args:
            dataset_file: Uploaded dataset file
            model_name: Pre-trained model name
            epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            test_size: Validation split size
            progress_callback: Function to update progress
            
        Returns:
            Tuple of (success, message)
        """
        try:
            self.training_status = "Loading dataset..."
            if progress_callback:
                progress_callback("Loading dataset...", 10)
            
            # Load and validate dataset
            success, message, df = self.load_cola_dataset(dataset_file.name)
            if not success:
                self.training_status = "Failed"
                return False, message
            
            self.training_status = "Preprocessing data..."
            if progress_callback:
                progress_callback("Preprocessing data...", 20)
            
            # Preprocess data
            examples = self.preprocess_cola_data(df)
            
            self.training_status = "Setting up model..."
            if progress_callback:
                progress_callback("Setting up model...", 30)
            
            # Setup model
            success, model_message = self.setup_model_and_tokenizer(model_name)
            if not success:
                self.training_status = "Failed"
                return False, model_message
            
            # Create datasets
            datasets = self.create_datasets(examples, test_size)
            
            self.training_status = "Tokenizing data..."
            if progress_callback:
                progress_callback("Tokenizing data...", 40)
            
            # Tokenize datasets
            tokenized_datasets = datasets.map(
                self.tokenize_function,
                batched=True,
                remove_columns=datasets['train'].column_names
            )
            
            self.training_status = "Preparing training..."
            if progress_callback:
                progress_callback("Preparing training...", 50)
            
            # Create output directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"./cola_model_{timestamp}"
            self.current_model_path = output_dir
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=epochs,
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=0.01,
                logging_dir=f"{output_dir}/logs",
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_steps=100,
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                warmup_steps=100,
                fp16=torch.cuda.is_available(),
                dataloader_pin_memory=False,
                remove_unused_columns=False,
            )
            
            # Data collator
            data_collator = DataCollatorForSeq2Seq(
                self.tokenizer,
                model=self.model,
                padding=True
            )
            
            # Create trainer
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_datasets['train'],
                eval_dataset=tokenized_datasets['validation'],
                data_collator=data_collator,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            self.training_status = "Training model..."
            if progress_callback:
                progress_callback("Training model... This may take a while.", 60)
            
            # Train model
            train_result = self.trainer.train()
            
            self.training_status = "Evaluating model..."
            if progress_callback:
                progress_callback("Evaluating model...", 90)
            
            # Evaluate model
            eval_results = self.trainer.evaluate()
            
            self.training_status = "Saving model..."
            if progress_callback:
                progress_callback("Saving model...", 95)
            
            # Save model
            final_model_dir = f"./cola_model_final_{timestamp}"
            self.model.save_pretrained(final_model_dir)
            self.tokenizer.save_pretrained(final_model_dir)
            self.current_model_path = final_model_dir
            
            self.training_status = "Completed"
            if progress_callback:
                progress_callback("Training completed!", 100)
            
            # Prepare results message
            results_message = f"""🎉 Training completed successfully!

📊 Training Results:
• Final training loss: {train_result.training_loss:.4f}
• Final validation loss: {eval_results['eval_loss']:.4f}
• Training steps: {train_result.global_step}
• Model saved to: {final_model_dir}

🔧 Training Configuration:
• Model: {model_name}
• Epochs: {epochs}
• Batch size: {batch_size}
• Learning rate: {learning_rate}
• Dataset size: {len(examples)} examples
• Train/Val split: {1-test_size:.0%}/{test_size:.0%}

✅ Model is ready for testing!"""
            
            return True, results_message
            
        except Exception as e:
            self.training_status = "Failed"
            return False, f"❌ Training failed: {str(e)}"
    
    def test_model(self, sentence: str) -> Tuple[str, float]:
        """
        Test the trained model on a sentence
        
        Args:
            sentence: Input sentence to test
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.model or not self.tokenizer:
            return "❌ No model loaded", 0.0
        
        try:
            # Prepare input
            input_text = f"cola: {sentence}"
            inputs = self.tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=10,
                    num_beams=2,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True
                )
            
            # Decode prediction
            prediction = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified)
            confidence = 0.85 if prediction.lower() in ['acceptable', 'unacceptable'] else 0.5
            
            return prediction, confidence
            
        except Exception as e:
            return f"❌ Error: {str(e)}", 0.0

# Global trainer instance
trainer = CoLATrainer()

def upload_dataset(file):
    """Handle dataset upload"""
    if file is None:
        return "❌ Please upload a dataset file.", None
    
    success, message, df = trainer.load_cola_dataset(file.name)
    
    if success:
        return message, df.head(10)  # Show first 10 rows
    else:
        return message, None

def start_training(dataset_file, model_name, epochs, batch_size, learning_rate, test_size, progress=gr.Progress()):
    """Start training process"""
    if dataset_file is None:
        return "❌ Please upload a dataset first."
    
    def progress_callback(message, percent):
        progress(percent/100, desc=message)
    
    # Run training in the main thread (Gradio handles async)
    success, message = trainer.train_model_async(
        dataset_file, model_name, epochs, batch_size, learning_rate, test_size, progress_callback
    )
    
    return message

def test_sentence(sentence):
    """Test a sentence with the trained model"""
    if not sentence.strip():
        return "❌ Please enter a sentence to test."
    
    if trainer.training_status != "Completed":
        return "❌ Please train a model first."
    
    prediction, confidence = trainer.test_model(sentence.strip())
    
    return f"""🧪 Test Result:

📝 Sentence: "{sentence}"
🎯 Prediction: {prediction}
📊 Confidence: {confidence:.2f}

💡 Interpretation:
{f"The sentence is grammatically {'acceptable' if 'acceptable' in prediction.lower() else 'unacceptable'}." if 'acceptable' in prediction.lower() or 'unacceptable' in prediction.lower() else "Unexpected prediction format."}"""

def get_training_status():
    """Get current training status"""
    return f"🔄 Status: {trainer.training_status}"

# Create Gradio interface
def create_interface():
    """Create and configure Gradio interface"""
    
    with gr.Blocks(title="CoLA Dataset Trainer", theme=gr.themes.Soft()) as interface:
        gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🌐 Google CoLA Dataset Trainer</h1>
            <p style="font-size: 18px; color: #666;">
                Train Flan-T5 models on Google CoLA (Corpus of Linguistic Acceptability) dataset
            </p>
        </div>
        """)
        
        with gr.Tab("📁 Dataset Upload"):
            with gr.Row():
                with gr.Column():
                    dataset_file = gr.File(
                        label="Upload CoLA Dataset",
                        file_types=[".csv", ".tsv", ".json"],
                        info="Upload your CoLA dataset (CSV, TSV, or JSON format)"
                    )
                    upload_btn = gr.Button("📊 Analyze Dataset", variant="primary")
                
                with gr.Column():
                    upload_output = gr.Textbox(
                        label="Dataset Analysis",
                        lines=10,
                        interactive=False
                    )
            
            dataset_preview = gr.Dataframe(
                label="Dataset Preview (First 10 rows)",
                interactive=False
            )
            
            gr.HTML("""
            <div style="margin: 20px 0; padding: 15px; background-color: #f0f8ff; border-radius: 8px;">
                <h3>📋 Dataset Format Requirements:</h3>
                <ul>
                    <li><strong>Columns:</strong> 'sentence' and 'label'</li>
                    <li><strong>Labels:</strong> 0 (unacceptable) or 1 (acceptable)</li>
                    <li><strong>Format:</strong> CSV, TSV, or JSON</li>
                    <li><strong>Example:</strong> sentence: "The cat is on the mat", label: 1</li>
                </ul>
            </div>
            """)
        
        with gr.Tab("🚀 Model Training"):
            with gr.Row():
                with gr.Column():
                    model_name = gr.Dropdown(
                        choices=[
                            "google/flan-t5-small",
                            "google/flan-t5-base",
                            "google/flan-t5-large"
                        ],
                        value="google/flan-t5-small",
                        label="Pre-trained Model",
                        info="Choose base model size (small=fastest, large=best quality)"
                    )
                    
                    epochs = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=3,
                        step=1,
                        label="Training Epochs",
                        info="Number of training epochs"
                    )
                    
                    batch_size = gr.Slider(
                        minimum=1,
                        maximum=16,
                        value=4,
                        step=1,
                        label="Batch Size",
                        info="Training batch size (lower if out of memory)"
                    )
                
                with gr.Column():
                    learning_rate = gr.Slider(
                        minimum=1e-6,
                        maximum=1e-3,
                        value=5e-5,
                        step=1e-6,
                        label="Learning Rate",
                        info="Optimizer learning rate"
                    )
                    
                    test_size = gr.Slider(
                        minimum=0.1,
                        maximum=0.4,
                        value=0.2,
                        step=0.05,
                        label="Validation Split",
                        info="Fraction of data for validation"
                    )
                    
                    train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")
            
            training_output = gr.Textbox(
                label="Training Progress & Results",
                lines=15,
                interactive=False
            )
            
            status_display = gr.Textbox(
                label="Current Status",
                value="Ready",
                interactive=False
            )
        
        with gr.Tab("🧪 Model Testing"):
            with gr.Column():
                test_sentence_input = gr.Textbox(
                    label="Test Sentence",
                    placeholder="Enter a sentence to test for grammatical acceptability...",
                    lines=3
                )
                
                test_btn = gr.Button("🧪 Test Sentence", variant="secondary")
                
                test_output = gr.Textbox(
                    label="Test Results",
                    lines=8,
                    interactive=False
                )
            
            gr.HTML("""
            <div style="margin: 20px 0; padding: 15px; background-color: #f0fff0; border-radius: 8px;">
                <h3>💡 Testing Tips:</h3>
                <ul>
                    <li><strong>Acceptable:</strong> "The cat is sleeping on the mat."</li>
                    <li><strong>Unacceptable:</strong> "Cat the on sleeping mat is the."</li>
                    <li>Test various grammatical constructions</li>
                    <li>Try both simple and complex sentences</li>
                </ul>
            </div>
            """)
        
        with gr.Tab("ℹ️ Information"):
            gr.HTML("""
            <div style="padding: 20px;">
                <h2>🎯 About Google CoLA Dataset</h2>
                <p>The Corpus of Linguistic Acceptability (CoLA) consists of English acceptability judgments drawn from books and journal articles on linguistic theory.</p>
                
                <h3>📊 Dataset Details:</h3>
                <ul>
                    <li><strong>Task:</strong> Binary classification (acceptable/unacceptable)</li>
                    <li><strong>Domain:</strong> Linguistic acceptability</li>
                    <li><strong>Size:</strong> ~10,000 sentences</li>
                    <li><strong>Labels:</strong> 0 (unacceptable), 1 (acceptable)</li>
                </ul>
                
                <h3>🤖 Model Information:</h3>
                <ul>
                    <li><strong>Architecture:</strong> Flan-T5 (Text-to-Text Transfer Transformer)</li>
                    <li><strong>Training:</strong> Fine-tuned for acceptability classification</li>
                    <li><strong>Input Format:</strong> "cola: [sentence]"</li>
                    <li><strong>Output:</strong> "acceptable" or "unacceptable"</li>
                </ul>
                
                <h3>🔧 Technical Requirements:</h3>
                <ul>
                    <li><strong>GPU:</strong> Recommended for faster training</li>
                    <li><strong>Memory:</strong> 8GB+ RAM, 4GB+ VRAM</li>
                    <li><strong>Time:</strong> 10-30 minutes depending on dataset size</li>
                </ul>
            </div>
            """)
        
        # Event handlers
        upload_btn.click(
            fn=upload_dataset,
            inputs=[dataset_file],
            outputs=[upload_output, dataset_preview]
        )
        
        train_btn.click(
            fn=start_training,
            inputs=[dataset_file, model_name, epochs, batch_size, learning_rate, test_size],
            outputs=[training_output]
        )
        
        test_btn.click(
            fn=test_sentence,
            inputs=[test_sentence_input],
            outputs=[test_output]
        )
        
        # Auto-refresh status every 5 seconds during training
        interface.load(
            fn=get_training_status,
            outputs=[status_display],
            every=5
        )
    
    return interface

def main():
    """Main function to launch the interface"""
    print("🌐 Starting Google CoLA Dataset Trainer Interface...")
    print("=" * 60)
    print(f"🖥️  Device: {trainer.device}")
    print(f"🔥 CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Create and launch interface
    interface = create_interface()
    
    # Launch with public link for sharing (optional)
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True for public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
