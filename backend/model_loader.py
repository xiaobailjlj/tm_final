import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Path settings
BASE_MODEL = "roberta-base"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'sentiment_analysis_using_roberta_classification.bin')

# Make sure the model directory exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def load_model():
    """
    Load the pre-trained RoBERTa model and tokenizer
    Returns: tokenizer, model, device
    """
    try:
        # Set device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        # Load tokenizer from the base model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        print("Tokenizer loaded successfully")

        # Load fine-tuned model from file
        if os.path.exists(MODEL_PATH):
            model = torch.load(MODEL_PATH, map_location=device)
            model.eval()  # Set to evaluation mode
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file not found at {MODEL_PATH}. Loading base model instead.")
            # If the fine-tuned model is not available, load the base model
            model = AutoModelForSequenceClassification.from_pretrained(
                BASE_MODEL,
                num_labels=2,
                id2label={0: 0, 1: 1},
                label2id={0: 0, 1: 1}
            )
            model.to(device)
            print("Base model loaded as fallback")

        return tokenizer, model, device

    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None


def classify_text(tokenizer, model, device, text_list, batch_size=16):
    """
    Classify a list of texts for bias using the loaded model

    Args:
        tokenizer: The RoBERTa tokenizer
        model: The loaded RoBERTa model
        device: The torch device (cuda or cpu)
        text_list: List of texts to classify
        batch_size: Size of batches to process

    Returns:
        List of dictionaries with classification results
    """
    if not model or not tokenizer:
        print("Model or tokenizer not available. Cannot classify text.")
        return []

    results = []

    # Process sentences in batches
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]

        # Tokenize the sentences
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

        # Convert to numpy for easier handling
        probs_np = probabilities.cpu().numpy()
        preds_np = predictions.cpu().numpy()

        # Add results
        for j, text in enumerate(batch):
            is_biased = bool(preds_np[j])
            # Use the probability of the positive class (biased) as the bias score
            bias_score = float(probs_np[j][1])

            results.append({
                'text': text,
                'is_biased': is_biased,
                'bias_score': bias_score
            })

    return results


# Test function
if __name__ == "__main__":
    # Load model and tokenizer
    tokenizer, model, device = load_model()

    if model and tokenizer:
        # Test with some example sentences
        test_sentences = [
            "The study reported a 5% increase in unemployment during the last quarter.",
            "Those corrupt politicians are destroying our country with their terrible policies.",
            "The report showed mixed results with both positive and negative outcomes.",
            "Everyone knows that this political party is full of liars and cheats."
        ]

        results = classify_text(tokenizer, model, device, test_sentences)

        # Print results
        print("\nBias Classification Results:")
        print("=" * 50)
        for result in results:
            bias_status = "BIASED" if result['is_biased'] else "UNBIASED"
            print(f"{bias_status} ({result['bias_score']:.4f}): {result['text']}")