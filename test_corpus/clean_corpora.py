
with open("/storage/ice-shared/vip-vyf/embeddings_team/corpora/clean_corpora.txt", "r", encoding="utf-8") as corpus:
    lines = corpus.readlines()

    total_words = [word for line in lines for word in line.split(" ")]
    unique_words = set(total_words)
    print(f"total words: {len(total_words)}\tunique words: {len(unique_words)}\n{len(unique_words)/len(total_words) * 100}% of the corpus is unique.")
    print(f"Lines: {len(lines)} \t\t Unique lines: {len(set(lines))}")

    print("Printing first 10 unique words")
    for word in list(unique_words)[:10]:
        print(repr(word))
