from flask import Flask, request, jsonify
import nltk
import logging
from nltk.tokenize import sent_tokenize
import csv
import os
import uuid
import yaml
from datetime import datetime
import torch
from transformers import AutoTokenizer
from flask_cors import CORS

# Download necessary NLTK data
nltk.download('punkt', quiet=True)


# Load configuration from YAML file
def load_config(config_file='./config.yaml'):
    config_path = os.path.join(os.path.dirname(__file__), config_file)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


# Initialize application with configuration
config = load_config()

app = Flask(__name__)
CORS(app,
     resources=config['cors']['resources'],
     methods=config['cors']['methods'])

# Configure logging from config
logging.basicConfig(
    level=getattr(logging, config['logging']['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Set up directories
DATA_DIR = './data'

for directory in [DATA_DIR]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

# Path to pretrained RoBERTa model and tokenizer
MODEL_PATH = '../models/sentiment_analysis_using_roberta_classification.bin'
BASE_MODEL = "roberta-base"  # Base model for the tokenizer

# Load the model and tokenizer
try:
    # Load tokenizer from the base model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    logger.info("Tokenizer loaded successfully")

    # Load the model from the saved file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = torch.load(MODEL_PATH, map_location=device)
    model.eval()  # Set to evaluation mode
    model_loaded = True
    logger.info("RoBERTa model loaded successfully!")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    logger.warning("Model not found. Using dummy classifier.")
    model_loaded = False


def dummy_classify(sentences):
    """Dummy classifier for testing when no model is available"""
    # This just randomly classifies sentences as biased or unbiased
    import random
    logger.info(f"Using dummy classifier for {len(sentences)} sentences")
    results = []
    for sentence in sentences:
        bias_score = random.uniform(0, 1)
        is_biased = bias_score > 0.5
        results.append({
            'sentence': sentence,
            'is_biased': is_biased,
            'bias_score': bias_score
        })
    return results


def classify_with_roberta(sentences):
    """Classify sentences using the fine-tuned RoBERTa model"""
    logger.info(f"Classifying {len(sentences)} sentences with RoBERTa model")
    results = []

    # Process sentences in batches to avoid memory issues
    batch_size = 16
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        logger.debug(f"Processing batch {i // batch_size + 1} with {len(batch)} sentences")

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
        for j, sentence in enumerate(batch):
            is_biased = bool(preds_np[j])
            # Use the probability of the positive class (biased) as the bias score
            bias_score = float(probs_np[j][1])

            results.append({
                'sentence': sentence,
                'is_biased': is_biased,
                'bias_score': bias_score
            })

    logger.debug(f"Classification completed. Found {sum(1 for r in results if r['is_biased'])} biased sentences")
    return results


def save_to_csv(article_id, sentences, results):
    """Save sentences and their classification results to CSV"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{article_id}_{timestamp}.csv"
    filepath = os.path.join(DATA_DIR, filename)

    logger.info(f"Saving results to CSV: {filepath}")

    try:
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['sentence_id', 'sentence', 'is_biased', 'bias_score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for i, (sentence, result) in enumerate(zip(sentences, results)):
                writer.writerow({
                    'sentence_id': i,
                    'sentence': sentence,
                    'is_biased': result['is_biased'],
                    'bias_score': result['bias_score']
                })
        logger.debug(f"Successfully wrote {len(sentences)} rows to CSV file")
    except Exception as e:
        logger.error(f"Error saving to CSV: {e}")
        raise

    return filepath


@app.route('/bias/analyze', methods=['POST'])
def analyze_article():
    """API endpoint to analyze an article for bias"""
    logger.info("Received request to /api/analyze")

    if not request.is_json:
        logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()

    if 'article_text' not in data:
        logger.warning("Missing article_text parameter")
        return jsonify({"error": "Missing article_text parameter"}), 400

    article_text = data['article_text']
    article_id = data.get('article_id', str(uuid.uuid4()))
    logger.info(f"Processing article with ID: {article_id}")

    # Break the article into sentences
    sentences = sent_tokenize(article_text)
    logger.info(f"Article broken into {len(sentences)} sentences")

    if not sentences:
        logger.warning("Could not extract any sentences from the text")
        return jsonify({"error": "Could not extract any sentences from the text"}), 400

    # Classify sentences for bias
    if model_loaded:
        # Use the RoBERTa model for classification
        logger.info("Using RoBERTa classification")
        results = classify_with_roberta(sentences)
    else:
        # Use dummy classifier if model isn't loaded
        logger.info("Using dummy classification")
        results = dummy_classify(sentences)

    # Save to CSV
    try:
        csv_path = save_to_csv(article_id, sentences, results)
        biased_count = sum(1 for r in results if r['is_biased'])
        logger.info(f"Analysis complete. Found {biased_count} biased sentences out of {len(sentences)}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return jsonify({"error": "Failed to save results"}), 500

    # Return results to client
    response = {
        "article_id": article_id,
        "total_sentences": len(sentences),
        "biased_sentences": biased_count,
        # "csv_path": os.path.basename(csv_path),
        "sentences": results
    }

    return jsonify(response)


# Enable CORS for frontend integration
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


if __name__ == '__main__':
    server_config = config['server']
    app.run(
        debug=server_config['debug'],
        host=server_config['host'],
        port=server_config['port']
    )