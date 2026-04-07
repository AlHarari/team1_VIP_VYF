import os
import requests, json
from lxml import etree
from re import sub
from concurrent.futures import ThreadPoolExecutor, as_completed 
import time

# Change first two lines when moving into GitHub repository
output_dir = "extended_corpus"  
os.makedirs(output_dir, exist_ok=True)
NUM_WORKERS = 10 

ns = {None: 'http://chs.harvard.edu/xmlns/cts'}

def gather_greek_urns():
    url = "https://scaife.perseus.org/library/json/"
    response = requests.get(url)
    texts = json.loads(response.text)["texts"]
    base_urns = []
    for text in texts:
        if text["human_lang"].lower() == "greek":
            base_urns.append(text["urn"])
    return base_urns

def get_xml(cite):
    url = f'https://scaife.perseus.org/library/{cite}/cts-api-xml/'
    print(f"Requesting xml: {url}")
    presp = requests.get(url)
    if presp.ok:
        ptree = etree.fromstring(presp.text)
        try:
            raw = ptree.find('.//reply/passage', namespaces=ns)
        except:
            print(f"Something went wrong with extracting passage from {cite}. Here's the response text.")
            print(presp.text)
        etree.strip_elements(raw, '{*}note', with_tail=False)
        return sub('[ ]*\n[ ]*', '\n[LB]', sub('[ \t]+', ' ', ''.join(raw.itertext()).strip())) + "\n[LB]"
    else:
        return None

def fetch_works(urn):
    """
        Given CTS urn (made up of namespace, author, and work), return list of all books in work.
        Ideally, each book should be in own text file.
    """
    try:
        reff_url = f'https://scaife.perseus.org/library/{urn}/cts-api-xml/reffs/'
        good = None
        # Find the most fine-grained citation level
        for level in range(1, 5):
            xresp = requests.get(reff_url + '?level=' + str(level))
            if not xresp.ok:
                break
            good = xresp.text
        if not good:
            return []
        tree = etree.fromstring(good)
        passages = []
        previous_book_num = None
        for purn in tree.findall('.//reff/urn', namespaces=ns):
            cite = purn.text
            citation = cite.split(":")[-1]
            book_num = citation.split(".")[0] if "." in citation else citation
            text = get_xml(cite)
            if text:
                if previous_book_num == book_num:  # If we're on the same book, add to the most recent element.
                    passages[-1][1] += text
                else:                              # If previous_book_num is None or the book_nums are not the same, add a new element.
                    passages.append([book_num, text])
                previous_book_num = book_num
            time.sleep(0.05)
        return passages                            # Length should be same as however many references there are with level = 1.
    except Exception as error:
        print(error)
        print(f"Something went wrong when getting the references for {urn}.")
        return []

def main():
    urns = gather_greek_urns()
    print(f"Found {len(urns)} Greek URNs.")
    urns_length = len(urns)
    book_texts = {}

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(fetch_works, urn): urn for urn in urns}
        for i, future in enumerate(as_completed(futures), 1):
            urn = futures[future]
            result = future.result()
            for book_num, book in result:
                book_texts[(urn, book_num)] = book 
            print(f"Collected {futures[future]}")
            print(f"Completed: {round(i / urns_length * 100, 2)}%")

    for (urn, book_num), text in book_texts.items():
        safe_urn = urn.replace(".", "_")[urn.rindex(":") + 1 :]
        filename = f"{safe_urn}_Book{book_num}.txt"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    print("Done creating corpus") 

if __name__ == "__main__":
    main()