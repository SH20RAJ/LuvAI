#!/usr/bin/env python3
"""
💕 Flan-T5 Romantic Chat Training Pipeline

Complete script to fine-tune Google's Flan-T5 model on romantic chat conversations.
This script provides data loading, preprocessing, training, evaluation, and inference capabilities.

Author: LuvAI Team
Date: June 2025
"""

import json
import os
import glob
import argparse
import sys
from pathlib import Path
import pandas as pd
import torch
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
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from evaluate import load


class RomanticChatTrainer:
    """
    Main class for training Flan-T5 on romantic chat data
    """
    
    def __init__(self, model_name: str = "google/flan-t5-small"):
        """
        Initialize the trainer
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.rouge_metric = load("rouge")
        
        print(f"🚀 Initializing LuvAI Romantic Chat Trainer")
        print(f"Using device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    def load_chat_data(self, folder_path: str) -> List[Dict]:
        """
        Load all JSON chat data from a folder
        
        Args:
            folder_path: Path to folder containing JSON files
        
        Returns:
            List of chat conversation dictionaries
        """
        all_conversations = []
        
        # Find all JSON files in the folder
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        
        print(f"📂 Found {len(json_files)} JSON files:")
        for file_path in json_files:
            print(f"  - {os.path.basename(file_path)}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_conversations.extend(data)
                    print(f"    Loaded {len(data)} conversations")
            except Exception as e:
                print(f"    ❌ Error loading {file_path}: {e}")
        
        return all_conversations
    
    def explore_dataset(self, conversations: List[Dict]) -> None:
        """
        Explore and analyze the chat dataset
        
        Args:
            conversations: List of conversation dictionaries
        """
        print("\n🔍 Dataset Analysis:")
        print(f"Total conversations: {len(conversations)}")
        
        # Analyze input and response lengths
        input_lengths = []
        response_lengths = []
        
        for conv in conversations:
            if 'input' in conv and 'response' in conv:
                input_lengths.append(len(conv['input']))
                response_lengths.append(len(conv['response']))
        
        if input_lengths and response_lengths:
            print(f"\n📏 Text Length Statistics:")
            print(f"Input - Mean: {np.mean(input_lengths):.1f}, Max: {max(input_lengths)}, Min: {min(input_lengths)}")
            print(f"Response - Mean: {np.mean(response_lengths):.1f}, Max: {max(response_lengths)}, Min: {min(response_lengths)}")
        
        # Language analysis (basic)
        hindi_count = 0
        english_count = 0
        mixed_count = 0
        
        for conv in conversations:
            text = conv.get('input', '') + ' ' + conv.get('response', '')
            # Simple heuristic for language detection
            if any(ord(char) > 127 for char in text):  # Contains non-ASCII (likely Hindi)
                if any(char.isascii() and char.isalpha() for char in text):
                    mixed_count += 1
                else:
                    hindi_count += 1
            else:
                english_count += 1
        
        print(f"\n🌐 Language Distribution:")
        print(f"English: {english_count}")
        print(f"Hindi/Hinglish: {hindi_count}")
        print(f"Mixed: {mixed_count}")
        
        # Show sample conversations
        print(f"\n📝 Sample Conversations:")
        for i, conv in enumerate(conversations[:3]):
            print(f"\nExample {i+1}:")
            print(f"Input: {conv.get('input', 'N/A')}")
            print(f"Response: {conv.get('response', 'N/A')}")
    
    def preprocess_conversations(self, conversations: List[Dict]) -> List[Dict]:
        """
        Preprocess conversations for T5 training format
        
        Args:
            conversations: List of conversation dictionaries
        
        Returns:
            Processed conversations ready for training
        """
        processed_data = []
        
        for conv in conversations:
            if 'input' in conv and 'response' in conv:
                # Extract human input and AI response
                human_input = conv['input']
                ai_response = conv['response']
                
                # Remove "Human: " and "AI: " prefixes if present
                if human_input.startswith("Human: "):
                    human_input = human_input[7:]
                if ai_response.startswith("AI: "):
                    ai_response = ai_response[4:]
                
                # Format for T5: input should be a task description + context
                # T5 is trained with task prefixes
                formatted_input = f"romantic chat: {human_input.strip()}"
                formatted_output = ai_response.strip()
                
                processed_data.append({
                    'input_text': formatted_input,
                    'target_text': formatted_output
                })
        
        return processed_data
    
    def setup_model_and_tokenizer(self) -> None:
        """
        Load the Flan-T5 model and tokenizer
        """
        print(f"\n🤖 Loading model: {self.model_name}")
        
        # Load tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        print(f"✅ Tokenizer loaded. Vocab size: {self.tokenizer.vocab_size}")
        
        # Load model
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
        print(f"✅ Model loaded and moved to {self.device}")
        print(f"📊 Model parameters: {self.model.num_parameters():,}")
        
        # Test tokenization
        test_input = "romantic chat: Hello beautiful, how are you? 💕"
        test_tokens = self.tokenizer(test_input, return_tensors="pt")
        print(f"\n🧪 Tokenization test:")
        print(f"Input: {test_input}")
        print(f"Tokens: {test_tokens['input_ids'].shape}")
        decoded = self.tokenizer.decode(test_tokens['input_ids'][0], skip_special_tokens=True)
        print(f"Decoded: {decoded}")
    
    def create_datasets(self, processed_conversations: List[Dict], test_size: float = 0.2) -> DatasetDict:
        """
        Create train and validation datasets
        
        Args:
            processed_conversations: List of processed conversation dictionaries
            test_size: Fraction of data to use for validation
        
        Returns:
            DatasetDict with train and validation splits
        """
        # Split the data
        train_data, val_data = train_test_split(
            processed_conversations, 
            test_size=test_size, 
            random_state=42
        )
        
        print(f"\n📊 Dataset split:")
        print(f"  Training: {len(train_data)} conversations")
        print(f"  Validation: {len(val_data)} conversations")
        
        # Create HuggingFace datasets
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data)
        
        # Combine into DatasetDict
        dataset_dict = DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
        
        return dataset_dict
    
    def tokenize_function(self, examples):
        """
        Tokenize the input and target texts
        """
        # Tokenize inputs
        model_inputs = self.tokenizer(
            examples['input_text'],
            max_length=512,
            truncation=True,
            padding=False  # We'll pad later in the data collator
        )
        
        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                examples['target_text'],
                max_length=512,
                truncation=True,
                padding=False
            )
        
        model_inputs['labels'] = labels['input_ids']
        return model_inputs
    
    def tokenize_datasets(self, datasets: DatasetDict) -> DatasetDict:
        """
        Apply tokenization to datasets
        
        Args:
            datasets: DatasetDict with train and validation splits
        
        Returns:
            Tokenized DatasetDict
        """
        print("\n🔄 Tokenizing datasets...")
        tokenized_datasets = datasets.map(
            self.tokenize_function,
            batched=True,
            remove_columns=datasets['train'].column_names
        )
        
        print("✅ Tokenization complete")
        print(f"Tokenized train dataset: {len(tokenized_datasets['train'])} examples")
        print(f"Tokenized validation dataset: {len(tokenized_datasets['validation'])} examples")
        
        # Show tokenized example
        print(f"\n📋 Tokenized Example:")
        example = tokenized_datasets['train'][0]
        print(f"Input IDs shape: {np.array(example['input_ids']).shape}")
        print(f"Labels shape: {np.array(example['labels']).shape}")
        print(f"Decoded input: {self.tokenizer.decode(example['input_ids'], skip_special_tokens=True)}")
        print(f"Decoded label: {self.tokenizer.decode(example['labels'], skip_special_tokens=True)}")
        
        return tokenized_datasets
    
    def compute_metrics(self, eval_pred):
        """
        Compute ROUGE metrics for evaluation
        """
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 in labels (used for padding) with tokenizer.pad_token_id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Compute ROUGE scores
        result = self.rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True
        )
        
        # Extract ROUGE scores
        result = {key: value * 100 for key, value in result.items()}
        
        return {
            "rouge1": result["rouge1"],
            "rouge2": result["rouge2"],
            "rougeL": result["rougeL"],
            "rougeLsum": result["rougeLsum"]
        }
    
    def setup_trainer(self, tokenized_datasets: DatasetDict, output_dir: str = "./flan-t5-romantic-chat") -> None:
        """
        Set up the trainer with training arguments and data collator
        
        Args:
            tokenized_datasets: Tokenized training and validation datasets
            output_dir: Directory to save training outputs
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,  # Start with 3 epochs
            per_device_train_batch_size=4,  # Adjust based on GPU memory
            per_device_eval_batch_size=4,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=200,
            save_strategy="steps",
            save_steps=500,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=None,  # Disable wandb reporting
            seed=42,
            fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
            gradient_accumulation_steps=2,  # Effective batch size = 4 * 2 = 8
            lr_scheduler_type="linear",
            learning_rate=5e-5,
            remove_unused_columns=False,
        )
        
        print(f"\n⚙️ Training Configuration:")
        print(f"  Output directory: {output_dir}")
        print(f"  Epochs: {training_args.num_train_epochs}")
        print(f"  Batch size: {training_args.per_device_train_batch_size}")
        print(f"  Learning rate: {training_args.learning_rate}")
        print(f"  Mixed precision (FP16): {training_args.fp16}")
        print(f"  Gradient accumulation: {training_args.gradient_accumulation_steps}")
        
        # Data collator for dynamic padding
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True,
            return_tensors="pt"
        )
        
        print("✅ Data collator configured for dynamic padding")
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        print("✅ Trainer configured successfully")
        print(f"📊 Training will start with {len(tokenized_datasets['train'])} training examples")
        print(f"📊 Evaluation will use {len(tokenized_datasets['validation'])} validation examples")
    
    def train_model(self) -> Dict:
        """
        Train the model and return training results
        
        Returns:
            Training results dictionary
        """
        # Check model before training
        print("\n🔍 Pre-training model check:")
        test_input = "romantic chat: I miss you so much 💔"
        test_encoding = self.tokenizer(test_input, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                test_encoding.input_ids,
                max_length=50,
                num_beams=2,
                early_stopping=True
            )
            pre_training_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Input: {test_input}")
            print(f"Pre-training output: {pre_training_output}")
        
        print("\n🚀 Starting training...")
        print("This may take a while depending on your hardware.")
        print("📊 Training progress will be shown below:")
        
        # Start training
        train_result = self.trainer.train()
        
        print("\n🎉 Training completed!")
        print(f"📊 Training Results:")
        print(f"  Final loss: {train_result.training_loss:.4f}")
        print(f"  Training time: {train_result.metrics['train_runtime']:.2f} seconds")
        print(f"  Samples per second: {train_result.metrics['train_samples_per_second']:.2f}")
        
        return train_result
    
    def evaluate_model(self) -> Dict:
        """
        Evaluate the trained model
        
        Returns:
            Evaluation results dictionary
        """
        print("\n📊 Evaluating trained model...")
        eval_results = self.trainer.evaluate()
        
        print(f"\n📈 Evaluation Results:")
        for key, value in eval_results.items():
            if key.startswith('eval_'):
                metric_name = key.replace('eval_', '')
                print(f"  {metric_name}: {value:.4f}")
        
        return eval_results
    
    def test_model_samples(self) -> None:
        """
        Test the model with sample inputs
        """
        print("\n🧪 Testing trained model with sample inputs:")
        
        test_cases = [
            "romantic chat: Good morning beautiful! ☀️",
            "romantic chat: I'm feeling sad today 😢", 
            "romantic chat: Tumhe pata hai main tumse kitna pyaar karta hun? 💕",
            "romantic chat: What would you do if I was there with you?",
            "romantic chat: Mujhe tumhari yaad aa rahi hai 💭"
        ]
        
        for test_input in test_cases:
            # Tokenize input
            inputs = self.tokenizer(test_input, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=100,
                    num_beams=3,
                    early_stopping=True,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode response
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"\n💬 Input: {test_input.replace('romantic chat: ', '')}")
            print(f"🤖 Response: {generated_text}")
    
    def save_model(self, save_directory: str = "./flan-t5-romantic-chat-final") -> None:
        """
        Save the trained model and tokenizer
        
        Args:
            save_directory: Directory to save the model
        """
        os.makedirs(save_directory, exist_ok=True)
        
        print(f"\n💾 Saving model to {save_directory}...")
        
        # Save model and tokenizer
        self.trainer.save_model(save_directory)
        self.tokenizer.save_pretrained(save_directory)
        
        print("✅ Model and tokenizer saved successfully!")
        print(f"📁 Model location: {save_directory}")
        
        # Also save training configuration
        eval_results = self.trainer.evaluate()
        config_info = {
            "model_name": self.model_name,
            "training_examples": len(self.trainer.train_dataset),
            "validation_examples": len(self.trainer.eval_dataset), 
            "epochs": self.trainer.args.num_train_epochs,
            "batch_size": self.trainer.args.per_device_train_batch_size,
            "learning_rate": self.trainer.args.learning_rate,
            "final_eval_loss": eval_results.get('eval_loss', 'N/A'),
            "final_rouge1": eval_results.get('eval_rouge1', 'N/A')
        }
        
        with open(f"{save_directory}/training_info.json", 'w') as f:
            json.dump(config_info, f, indent=2)
        
        print("📋 Training configuration saved to training_info.json")
    
    def load_trained_model(self, model_path: str) -> Tuple[T5Tokenizer, T5ForConditionalGeneration]:
        """
        Load a saved model and tokenizer
        
        Args:
            model_path: Path to the saved model
        
        Returns:
            Tuple of (tokenizer, model)
        """
        print(f"\n🔄 Loading model from {model_path}...")
        
        # Load tokenizer and model
        loaded_tokenizer = T5Tokenizer.from_pretrained(model_path)
        loaded_model = T5ForConditionalGeneration.from_pretrained(model_path)
        loaded_model.to(self.device)
        
        print("✅ Model loaded successfully!")
        
        return loaded_tokenizer, loaded_model
    
    def generate_romantic_response(self, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer, user_input: str) -> str:
        """
        Generate a romantic response using the trained model
        
        Args:
            model: The trained model
            tokenizer: The tokenizer
            user_input: User's input message
        
        Returns:
            Generated response
        """
        # Format input for the model
        formatted_input = f"romantic chat: {user_input}"
        
        # Tokenize input
        inputs = tokenizer(formatted_input, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=150,
                num_beams=4,
                early_stopping=True,
                temperature=0.8,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode and return response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    
    def interactive_chat(self, model_path: Optional[str] = None) -> None:
        """
        Start an interactive chat session
        
        Args:
            model_path: Optional path to load a specific model
        """
        if model_path:
            tokenizer, model = self.load_trained_model(model_path)
        else:
            tokenizer, model = self.tokenizer, self.model
        
        print("\n💕 Welcome to your Romantic AI Chatbot!")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("💬 You: ")
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("🤖 AI: Goodbye my love! 💕")
                break
            
            if user_input.strip():
                response = self.generate_romantic_response(model, tokenizer, user_input)
                print(f"🤖 AI: {response}\n")
            else:
                print("🤖 AI: Please say something, darling! 💖\n")


def main():
    """
    Main function to run the training pipeline
    """
    parser = argparse.ArgumentParser(description="LuvAI Romantic Chat Training Pipeline")
    parser.add_argument("--data_path", type=str, default="./LOVE", 
                        help="Path to folder containing JSON conversation files")
    parser.add_argument("--model_name", type=str, default="google/flan-t5-small",
                        help="Pre-trained model name")
    parser.add_argument("--output_dir", type=str, default="./flan-t5-romantic-chat",
                        help="Output directory for training")
    parser.add_argument("--save_dir", type=str, default="./flan-t5-romantic-chat-final",
                        help="Directory to save final model")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Fraction of data to use for validation")
    parser.add_argument("--interactive", action="store_true",
                        help="Start interactive chat after training")
    parser.add_argument("--load_model", type=str, default=None,
                        help="Path to load a pre-trained model for inference only")
    
    args = parser.parse_args()
    
    print("🚀 Starting LuvAI Romantic Chat Training Pipeline")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RomanticChatTrainer(model_name=args.model_name)
    
    if args.load_model:
        # Load existing model for inference
        print(f"📂 Loading existing model from {args.load_model}")
        trainer.interactive_chat(args.load_model)
        return
    
    # Load and explore data
    print("\n📂 Loading conversation data...")
    conversations = trainer.load_chat_data(args.data_path)
    
    if not conversations:
        print("❌ No conversations loaded. Please check your data path.")
        return
    
    trainer.explore_dataset(conversations)
    
    # Preprocess data
    print("\n🛠️ Preprocessing conversations...")
    processed_conversations = trainer.preprocess_conversations(conversations)
    print(f"✅ Processed {len(processed_conversations)} conversations")
    
    # Setup model and tokenizer
    trainer.setup_model_and_tokenizer()
    
    # Create and tokenize datasets
    datasets = trainer.create_datasets(processed_conversations, test_size=args.test_size)
    tokenized_datasets = trainer.tokenize_datasets(datasets)
    
    # Setup trainer
    trainer.setup_trainer(tokenized_datasets, output_dir=args.output_dir)
    
    # Train model
    train_result = trainer.train_model()
    
    # Evaluate model
    eval_results = trainer.evaluate_model()
    
    # Test with samples
    trainer.test_model_samples()
    
    # Save model
    trainer.save_model(save_directory=args.save_dir)
    
    print("\n🎉 Training pipeline completed successfully!")
    print(f"📁 Model saved to: {args.save_dir}")
    
    # Interactive chat if requested
    if args.interactive:
        trainer.interactive_chat()


if __name__ == "__main__":
    main()
