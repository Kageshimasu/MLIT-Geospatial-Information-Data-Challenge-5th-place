import pandas as pd
import gensim
from gensim.models import CoherenceModel

file_paths = [
    './tokyo_data_with_topic5.csv',
    './tokyo_data_with_topic7.csv',
    './tokyo_data_with_topic10.csv',
    './tokyo_data_with_topic13.csv',
    './tokyo_data_with_topic15.csv',
    './tokyo_data_with_topic17.csv',
    './tokyo_data_with_topic20.csv'
]

# 結果を保存する辞書
coherence_scores = {}

for file_path in file_paths:
    num_topics = int(file_path.split('_topic')[-1].split('.')[0])
    tokyo_data = pd.read_csv(file_path)

    topic_columns = [f'topic_{i}' for i in range(num_topics)]
    non_binary_columns = ['target_ym', 'money_room', 'lat', 'lon', 'unit_area', 'addr1_1', 'addr1_2']
    binary_columns = tokyo_data.columns.difference(topic_columns + non_binary_columns)
    
    texts = []
    for _, row in tokyo_data[binary_columns].iterrows():
        words_in_doc = row[row == 1].index.tolist()
        texts.append(words_in_doc)
    
    top_words_per_topic = []
    for i in range(num_topics):
        top_words = tokyo_data[binary_columns].corrwith(tokyo_data[f'topic_{i}']).abs().sort_values(ascending=False).index[:10].tolist()
        top_words_per_topic.append(top_words)

    dictionary = gensim.corpora.Dictionary(texts)

    coherence_model = CoherenceModel(
        topics=top_words_per_topic, 
        texts=texts, 
        dictionary=dictionary, 
        coherence='c_v',
        processes=1
    )

    coherence_score = coherence_model.get_coherence()
    coherence_scores[num_topics] = coherence_score

    print(f"Coherence Score for {num_topics} topics: {coherence_score}")

print("\nAll Coherence Scores:")
for num_topics, score in sorted(coherence_scores.items()):
    print(f"Topics: {num_topics}, Coherence Score: {score}")
