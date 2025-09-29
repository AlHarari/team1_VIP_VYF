import re

def clean_up(text: str) -> str:
    text = re.sub(r"\W?\W?[A-Za-z]\W?\W?", "", text) # One fear I have is that this might be adding a lot of empty spaces.

    text_r = re.sub(r"\b[^\w\s]", r" \g<0>", text) # punctuations or signs that occur at the end of a word.
    text_l = re.sub(r"[^\w\s]\b", r"\g<0> ", text_r)
    return text_l

test_strings = [
        "Hello there! How) are you doing, my friend? Interestingly, I ate two potatoes for breakfast. ",
        "Three potatoes actually. What? They were good; I made them.",
        "See, no more matches! I say, \"No matches anymore, I say. Now here me out boy, I say there ain't no more matches; that's what I says.\"",
        "As the great Αρχιμήδης, who you might, or might not, know, once said, \"δός μοι πᾷ στῶ, καὶ τὰν γᾶν κινῶ.\"",
        "Κάθε φορά που κλαίει το μωρό, βάζω το δάχτυλό μου στα μάτια του.",
        "«Τι θα κάνουμε με το μωρό; Τι θα γίνουμε με το μωρό-γιο;» Στείλτε τον στο μαμά-γιο του."
]


for string in test_strings:
    print("INPUT SENTENCE:")
    print(string)
    print("OUTPUT SENTENCE:")
    print(clean_up(string))
    print("length: ", len(clean_up(string)))

    print("\n\n")
