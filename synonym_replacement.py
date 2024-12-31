import nltk
import pandas as pd
import torch
from nltk.corpus import wordnet
import random
import spacy
from transformers import pipeline

# 下载所需的资源
nltk.download('wordnet')
nltk.download('omw-1.4')
nlp = spacy.load("en_core_web_sm")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 加载 BERT 模型用于语义相似度计算
similarity_model = pipeline("feature-extraction", model="bert-base-uncased", tokenizer="bert-base-uncased", device=device)

def get_synonyms(word, pos=None):
    """
    获取单词的同义词列表，支持按词性筛选。
    """
    synonyms = []
    for syn in wordnet.synsets(word):
        if pos and syn.pos() != pos:  # 按词性过滤
            continue
        for lemma in syn.lemmas():
            synonym = lemma.name()
            if synonym != word and '-' not in synonym:  # 排除自身和复合词
                synonyms.append(synonym.replace('_', ' '))  # 替换下划线为空格
    return list(set(synonyms))

def semantic_similarity(sentence1, sentence2):
    """
    计算两句话的语义相似度。
    """
    vec1 = similarity_model(sentence1)[0]
    vec2 = similarity_model(sentence2)[0]
    similarity = sum([sum(a * b for a, b in zip(v1, v2)) for v1, v2 in zip(vec1, vec2)]) / len(vec1)
    return similarity

def synonym_replacement(sentence, n=1, num_sentences=5, similarity_threshold=0.8):
    """
    改进的同义词替换，生成多句增强句子。
    Args:
        sentence (str): 输入的句子。
        n (int): 每句替换的单词数量。
        num_sentences (int): 生成的增强句子数量。
        similarity_threshold (float): 替换后句子与原句的最低语义相似度。
    Returns:
        list: 增强后的句子列表。
    """
    doc = nlp(sentence)
    words = [token.text for token in doc]
    pos_tags = [token.pos_ for token in doc]

    # 可替换的单词列表
    replaceable_words = [(word, pos) for word, pos in zip(words, pos_tags) if pos in {"NOUN", "VERB", "ADJ"}]
    if not replaceable_words:
        return [sentence]  # 没有可替换单词，返回原句

    augmented_sentences = []
    for _ in range(num_sentences):
        new_words = words[:]
        random.shuffle(replaceable_words)
        words_to_replace = replaceable_words[:n]

        for word, pos in words_to_replace:
            pos_map = {"NOUN": "n", "VERB": "v", "ADJ": "a"}
            synonyms = get_synonyms(word, pos=pos_map.get(pos))
            if synonyms:
                synonym = random.choice(synonyms)
                new_words = [synonym if w == word else w for w in new_words]

        new_sentence = " ".join(new_words)
        # 计算语义相似度
        if semantic_similarity(sentence, new_sentence) >= similarity_threshold:
            if new_sentence[0] != '\"':
                new_sentence = '\"' + new_sentence
            if new_sentence[-1] != '\"':
                new_sentence += '\"'
            augmented_sentences.append(new_sentence)

    return list(set(augmented_sentences))  # 去重后返回

# 加载CSV文件
data = pd.read_csv('news_bias_dataset/preprocessed_dataset.csv')
unique_sentences = data[['source_bias', 'id_event', 'id_article', 'id_sentence', 'sentence_text']].drop_duplicates(subset=['sentence_text']).to_dict(orient='records')
augmented_data = []

for sentence in unique_sentences:
    augmented_sentences = synonym_replacement(sentence['sentence_text'], n=2, num_sentences=4)
    for aug_sentence in augmented_sentences:
        new_row = sentence.copy()
        new_row['sentence_text'] = aug_sentence
        augmented_data.append(new_row)

augmented_df = pd.DataFrame(augmented_data)
augmented_df.to_csv('news_bias_dataset/augmented_dataset.csv', index=False)