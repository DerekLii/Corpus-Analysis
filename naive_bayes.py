import sys
import re
import math
from collections import Counter

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

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords_list]
    cleaned_text = " ".join(filtered_words)
    
    return cleaned_text

def remove_unnecessary(text):
    unnecessary = ["their", "said", "one", "say", "would", "like", "may", "thou", "yet", "shall", "thee", "must", "upon", "much", "make", "us", "many", "know", "good", "might", "see", "come", "made", "could", "without", "well", ]
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in unnecessary]
    cleaned_text = " ".join(filtered_words)

    return cleaned_text

# pre process and then generate bag of words using counter
def bag_of_words(text):
    text = text.lower()  # Convert to lowercase
    text = remove_punctuation(text)
    text = remove_numbers(text)
    text = remove_stop_words(text)
    text = remove_unnecessary(text)
    words = text.split()
    return Counter(words)

# P(w|c)
def probabilities(word_counts, vocab):
    total_words = sum(word_counts.values())
    probabilities = {}
    for word, count in word_counts.items():
        # smoothing
        probabilities[word] = (count + 1)/ (total_words + vocab)
    return probabilities

# Likelihood Ratio LLR
def compute_llr(prob_w_c, prob_w_co):
    if prob_w_c == 0 or prob_w_co == 0:
        return float('-inf')  # Avoid log(0)
    return math.log(prob_w_c) - math.log(prob_w_co)

# final step to compare the distinguisihed words between the two books based on LLR
def compare_probabilities(probabilities1, probabilities2):
    all_words = set(probabilities1.keys()).union(set(probabilities2.keys()))
    llr_scores = {}
    vocab = len(all_words)
    
    for word in all_words:
        prob_1 = probabilities1.get(word, 1 / (sum(probabilities1.values()) + vocab))
        prob_2 = probabilities2.get(word, 1 / (sum(probabilities2.values()) + vocab))
        llr_scores[word] = compute_llr(prob_1, prob_2)
    # Get top 10 words by LLR score
    return sorted(llr_scores.items(), key=lambda x: x[1], reverse=True)[:10]

# split book (category) into chapters (documents)
def chapter_split(text):
    # split at every time we see the word chapters
    if re.search(r'\bmoby dick\b', text, flags=re.IGNORECASE):
        chapters = re.split(r'CHAPTER \d+', text, flags=re.IGNORECASE)
    else: 
        chapters = re.split(r'\bCHAPTER\b', text)
    # remove everything before the first chapter word, cuz they're irrelevant
    chapters = chapters[1:]
    return chapters

def process_book(filename):
    # read the book and do the splitting
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        chapters = chapter_split(text)  # Split into chapters
    return chapters

def count_tokens(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
        text = re.sub(r'[^\w\s]', '', text)  
        words = text.split()  
    return len(words)  

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <book1.txt> <book2.txt>")
        return

    filename1 = sys.argv[1]
    filename2 = sys.argv[2]

    # split book(categories) into chapters(documents)
    chapters1 = process_book(filename1)
    chapters2 = process_book(filename2)
    print("Moby Dick | # of Chapters:", len(chapters1), "| Average Tokens per Document: ", count_tokens(filename1) // len(chapters1) )
    print("War and Peace | # of Chapters:", len(chapters2), "| Average Tokens per Document: ", count_tokens(filename2) // len(chapters2))

    # probabilities of each word in each chapter
    all_chapters1 = []
    all_chapters2 = []

    # book1 doc loop
    for chapter in chapters1:
        bag = bag_of_words(chapter)
        # print(bag.most_common()[0:5])
        vocab = len(set(bag.keys()))
        probs = probabilities(bag, vocab)
        all_chapters1.append(probs)

    # book2 doc loop
    for chapter in chapters2:
        bag = bag_of_words(chapter)
         # print(bag.most_common()[0:5])
        vocab = len(set(bag.keys()))
        probs = probabilities(bag, vocab)
        all_chapters2.append(probs)

    # occurrences of a word per chapter on average
    avg_probs1 = {word: sum(chap.get(word, 0) for chap in all_chapters1) / len(all_chapters1) for word in set().union(*all_chapters1)}
    avg_probs2 = {word: sum(chap.get(word, 0) for chap in all_chapters2) / len(all_chapters2) for word in set().union(*all_chapters2)}

    # top words
    top_words1 = compare_probabilities(avg_probs1, avg_probs2)
    top_words2 = compare_probabilities(avg_probs2, avg_probs1)

    print("Top words for Book 1:")
    print(top_words1)
    print("\nTop words for Book 2:")
    print(top_words2)

if __name__ == "__main__":
    main()
