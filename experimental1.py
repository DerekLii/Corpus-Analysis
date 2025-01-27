import re
import gensim
from gensim import corpora
import sys
import pyLDAvis
import pyLDAvis.gensim_models

def remove_punctuation(line):
    # remove these special dashes and underscores
    line = re.sub('[—_]', ' ', line)
        # reg ex means match anything that's not(^) a word character(\w) or whitespace(\s)
    line = re.sub(r'[^\w\s]','', line)
    return line
    
def remove_numbers(line):
    # remove numbers
    line = re.sub(r'\d+', '', line)
    return line

def remove_stop_words(text):

    stopwords_list = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"}

    # Split text into words
    words = text.split()

    # Filter out stopwords
    filtered_words = [word for word in words if word.lower() not in stopwords_list]

    # Join the remaining words back into a single string
    cleaned_text = " ".join(filtered_words)
    return cleaned_text

def remove_unnecessary(text):
    unnecessary = ["their", "said", "one", "say", "would", "like", "may", "thou", "yet", "shall", "thee", "must", "upon", "much", "make", "us", "many", "know", "good", "might", "see", "come", "made", "could", "without", "well", "man", "went", "ye", "gutenberg"]
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in unnecessary]
    cleaned_text = " ".join(filtered_words)

    return cleaned_text

def simple_stem(word):
    # Apply basic stemming rules with regular expressions
    word = re.sub(r'(ing|ed|ly|es|s|er)$', '', word)  # Remove common suffixes
    return word

def apply_stemming(words):
    # Stem each word in the list
    stemmed_words = [simple_stem(word) for word in words]
    return stemmed_words

def process(text):
    # Process text before doing bag of words stuff
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = remove_unnecessary(text)
    words = text.split()
    words = apply_stemming(words) 
    return words

def run_lda(corpus, dictionary, num_topics=3):
    # Train LDA model using Gensim
    lda_model = gensim.models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda_model

def print_topics(lda_model, num_words=5):
    # Display the topics with their top words
    topics = lda_model.print_topics(num_words=num_words)
    for topic in topics:
        print(topic)

def calculate_average_topic_distribution(lda_model, corpus):
    topic_distributions = [lda_model.get_document_topics(doc, minimum_probability=0) for doc in corpus]
    num_topics = lda_model.num_topics

    # Average the topic probabilities for each topic
    average_distribution = [0] * num_topics
    for doc_distribution in topic_distributions:
        for topic_id, probability in doc_distribution:
            average_distribution[topic_id] += probability

    average_distribution = [prob / len(corpus) for prob in average_distribution]

    return average_distribution

def visualize_lda(lda_model, corpus, dictionary, filename):
    # Generate pyLDAvis visualization
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis_data, filename)
    print(f"LDA visualization saved to {filename}")

# Function to split a book into chapters
def chapter_split(text):
    # Split the text into chapters using "Chapter X" as a delimiter
    chapters = re.split(r'chapter \d+', text, flags=re.IGNORECASE)
    # Remove the first element (usually empty or preface text)
    chapters = chapters[1:]
    return chapters

def process_book(filename):
    # Read and split the book into chapters
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        chapters = chapter_split(text)  # Split into chapters
    return chapters

def main():
    if len(sys.argv) < 3:
        print("Error: Provide two text files as arguments.")
        return

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    # Process books and split into chapters
    chapters1 = process_book(filename1)
    chapters2 = process_book(filename2)

    # Process each chapter separately for Book 1
    chapters_words1 = [process(chap) for chap in chapters1]

    # Process each chapter separately for Book 2
    chapters_words2 = [process(chap) for chap in chapters2]

    # Process Book 1 and Book 2 separately with LDA

    print("Processing Book 1:")
    # Create dictionary for Book 1
    dictionary1 = corpora.Dictionary(chapters_words1)
    # Create corpus for Book 1
    corpus1 = [dictionary1.doc2bow(doc) for doc in chapters_words1]
    lda_model1 = run_lda(corpus1, dictionary1)
    print("Top topics for Book 1:")
    print_topics(lda_model1)

    # visualize_lda(lda_model1, corpus1, dictionary1, "book1_lda.html")

    print("\nAverage Topic Distribution for Book 1:")
    avg_distribution1 = calculate_average_topic_distribution(lda_model1, corpus1)
    sorted_topics1 = sorted(enumerate(avg_distribution1), key=lambda x: x[1], reverse=True)[:5]
    for topic_id, avg_prob in sorted_topics1:
        print(f"Topic {topic_id}: {avg_prob:.4f} - {lda_model1.print_topic(topic_id)}")

    print("\nProcessing Book 2:")
    # Create dictionary for Book 2
    dictionary2 = corpora.Dictionary(chapters_words2)
    # Create corpus for Book 2
    corpus2 = [dictionary2.doc2bow(doc) for doc in chapters_words2]
    lda_model2 = run_lda(corpus2, dictionary2)
    print("Top topics for Book 2:")
    print_topics(lda_model2)

    # visualize_lda(lda_model2, corpus2, dictionary2, "book2_lda.html")

    print("\nAverage Topic Distribution for Book 2:")
    avg_distribution2 = calculate_average_topic_distribution(lda_model2, corpus2)
    sorted_topics2 = sorted(enumerate(avg_distribution2), key=lambda x: x[1], reverse=True)[:5]
    for topic_id, avg_prob in sorted_topics2:
        print(f"Topic {topic_id}: {avg_prob:.4f} - {lda_model2.print_topic(topic_id)}")

if __name__ == "__main__":
    main()
