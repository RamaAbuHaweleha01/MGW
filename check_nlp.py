import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification

def check_env():
    print("--- Environment Check ---")
    
    # 1. Check Torch & CUDA
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available (GPU): {cuda_available}")
    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")

    # 2. Check Transformers & RoBERTa
    try:
        model_name = "roberta-base"
        tokenizer = RobertaTokenizer.from_pretrained(model_name)
        print(f"\nRoBERTa Tokenizer: Successfully loaded '{model_name}'")
        
        # Test tokenization
        test_text = "Subject: Urgent account verification needed"
        tokens = tokenizer.encode(test_text)
        print(f"Test Tokenization: {tokens[:5]}... (Success)")
        
    except Exception as e:
        print(f"\nError loading RoBERTa: {e}")

if __name__ == "__main__":
    check_env()
