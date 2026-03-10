import requests, json, sys
from lxml import etree
from re import sub

ns = {None: 'http://chs.harvard.edu/xmlns/cts'}

## Professor Kartik tried this but the /text endpoint doesn't remove notes.
def getText(cite):
    try:
        print('here')
        url = f'https://scaife.perseus.org/library/passage/{cite}/text/'
        print(f"Requesting: {url}")  # Debugging
        presp = requests.get(url)
        print(f"Response Status: {presp.status_code}")  # Debugging
        if presp.ok:
            # print('here')
            return presp.text
        else:
            # print('here')
            print(f"API Response Error: {presp.text}")  # Print error message
            return None
    except:
        print(f'## {cite}', file=sys.stderr)
        return None

def getXML(cite):
    url = f'https://scaife.perseus.org/library/{cite}/cts-api-xml/'
    presp = requests.get(url)
    if presp.ok:
        ptree = etree.fromstring(presp.text)
        try:
            raw = ptree.find('.//reply/passage', namespaces=ns)
        except:
            print(f"Something went wrong with extracting passage from {cite}. Here's the response text.")
            print(presp.text)
        etree.strip_elements(raw, '{*}note', with_tail=False)
        return sub('[ ]*\n[ ]*', '\n', sub('[ \t]+', ' ', ''.join(raw.itertext()).strip())) + "\n"
    else:
        return None

def gather_greek_urns():
    url = "https://scaife.perseus.org/library/json/"
    response = requests.get(url)
    texts = json.loads(response.text)["texts"]
    base_urns = []
    for text in texts:
        if text["human_lang"].lower() == "greek":
            base_urns.append(text["urn"])
    return base_urns


def append_corpora_text(text_data):
    """
    Appends the given text_data string to two files:
      - corpora.txt (plain text, UTF-8)
      - corpora.bin (binary, UTF-8-encoded)
    Each call appends to the existing files instead of overwriting them.

    Note that the files corpora.txt and corpora.bin are created if they don't exist
    If they do exist, you will need to delete them as this func only appends it does
    not overwrite text
    """
    if text_data:
        with open("extended_corpora_final.txt", "a", encoding="utf-8") as txt_file, open("extended_corpora_final.bin", "ab") as bin_file:
            txt_file.write(text_data + "\n")
            bin_file.write((text_data + "\n").encode("utf-8"))


def main():
    #urns = ["urn:cts:greekLit:tlg0552.tlg013.1st1K-grc1"] stopped here last time, but I guess we'll have to repeat again
    #urns = ["urn:cts:greekLit:tlg0011.tlg007.perseus-grc2"]
    urns = gather_greek_urns()
    urns_length = len(urns)
    i = 0
    for urn in urns:
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
                continue
            tree = etree.fromstring(good)
            seq = 0
            for purn in tree.findall('.//reff/urn', namespaces=ns):
                cite = purn.text
                # loc = rec['work'] + ':' + cite.split(':')[4]
                text = getXML(cite)
                print(json.dumps({'id': cite,
                                'book': urn,
                                'seq': seq,
                                'text': text},
                                ensure_ascii=False))
                append_corpora_text(text)
                seq += 1
            i += 1
            print(f"Added {i} out of {urns_length} total texts ({round((i / urns_length) * 100, 2)}).")
        except Exception as error:
            print(error)
            print(f"Something went wrong when getting the references for {urn}.")
            print(text)
            break
        # The code below is significantly quicker, but the text is less formatted.
        # if len(valid_reffs) < 2:
        #     print(urn)
        #     text = getXML(valid_reffs[0].text)
        # else:
        #     first_section = valid_reffs[0].text.split(":")[-1]
        #     last_section = valid_reffs[-1].text.split(":")[-1]
        #     print(urn + ":" + first_section + "-" + last_section)
        #     text = getXML(urn + ":" + first_section + "-" + last_section)
        # append_corpora_text(text)
        # i += 1
        #\print(f"Added {i} out of {urns_length} total texts ({round((i / urns_length) * 100, 2)}).")

if __name__ == "__main__":
    main()