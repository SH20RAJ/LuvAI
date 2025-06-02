#!/usr/bin/env python3
"""
LuvAI Usage Examples

This script demonstrates various ways to use the LuvAI training pipeline.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and print its description"""
    print(f"\n🔥 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    print("💕 LuvAI Training Pipeline - Usage Examples")
    print("=" * 60)
    
    # Check if the training script exists
    if not os.path.exists("luvai_training.py"):
        print("❌ Error: luvai_training.py not found in current directory")
        return
    
    print("\n📚 Available Usage Examples:")
    print("1. Install requirements")
    print("2. Basic training")
    print("3. Training with custom parameters")
    print("4. Interactive chat with trained model")
    print("5. Load existing model for chat")
    print("6. Training with different model size")
    
    while True:
        choice = input("\n💬 Enter your choice (1-6) or 'q' to quit: ").strip()
        
        if choice.lower() == 'q':
            print("👋 Goodbye!")
            break
        
        if choice == "1":
            # Install requirements
            run_command(
                "pip install -r requirements.txt",
                "Installing required packages"
            )
        
        elif choice == "2":
            # Basic training
            run_command(
                "python luvai_training.py --data_path ./LOVE",
                "Basic training with default parameters"
            )
        
        elif choice == "3":
            # Training with custom parameters
            custom_command = """python luvai_training.py \\
  --data_path ./LOVE \\
  --model_name google/flan-t5-base \\
  --output_dir ./my_romantic_model \\
  --save_dir ./my_final_model \\
  --test_size 0.15"""
            
            run_command(
                custom_command,
                "Training with custom parameters (larger model, different split)"
            )
        
        elif choice == "4":
            # Interactive chat after training
            run_command(
                "python luvai_training.py --data_path ./LOVE --interactive",
                "Training followed by interactive chat session"
            )
        
        elif choice == "5":
            # Load existing model for chat
            model_path = input("📂 Enter model path (or press Enter for default): ").strip()
            if not model_path:
                model_path = "./flan-t5-romantic-chat-final"
            
            run_command(
                f"python luvai_training.py --load_model {model_path}",
                f"Loading existing model from {model_path} for chat"
            )
        
        elif choice == "6":
            # Training with different model sizes
            print("\n🤖 Available model sizes:")
            print("  - google/flan-t5-small (77M params, fastest)")
            print("  - google/flan-t5-base (250M params, balanced)")
            print("  - google/flan-t5-large (780M params, best quality)")
            
            model_choice = input("🎯 Choose model (small/base/large): ").strip().lower()
            
            if model_choice in ['small', 'base', 'large']:
                model_name = f"google/flan-t5-{model_choice}"
                run_command(
                    f"python luvai_training.py --data_path ./LOVE --model_name {model_name}",
                    f"Training with {model_name}"
                )
            else:
                print("❌ Invalid model choice")
        
        else:
            print("❌ Invalid choice. Please enter 1-6 or 'q'")

def print_advanced_usage():
    """Print advanced usage examples"""
    print("\n🔧 Advanced Usage Examples:")
    print("=" * 40)
    
    examples = [
        {
            "title": "Training with GPU optimization",
            "command": "CUDA_VISIBLE_DEVICES=0 python luvai_training.py --data_path ./LOVE"
        },
        {
            "title": "Training with custom batch size (if running out of memory)",
            "command": "python luvai_training.py --data_path ./LOVE # Edit batch_size in script"
        },
        {
            "title": "Quick test with small dataset",
            "command": "python luvai_training.py --data_path ./LOVE --test_size 0.5"
        },
        {
            "title": "Production training (base model)",
            "command": "python luvai_training.py --data_path ./LOVE --model_name google/flan-t5-base --save_dir ./production_model"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print(f"   {example['command']}")

if __name__ == "__main__":
    main()
    
    # Ask if user wants to see advanced examples
    show_advanced = input("\n🚀 Show advanced usage examples? (y/n): ").strip().lower()
    if show_advanced == 'y':
        print_advanced_usage()
