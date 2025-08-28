#!/usr/bin/env python

import os
import requests, json, fileinput, sys
from lxml import etree
from re import sub
from bs4 import BeautifulSoup

ns = {None: 'http://chs.harvard.edu/xmlns/cts'}

## I tried this but the /text endpooint doesn't remove notes.
def getText(cite):
    try:
        print('here')
        # url = f'https://scaife-dev.perseus.org/library/passage/{cite}/text/'
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
    try:
        url = f'https://scaife-dev.perseus.org/library/{cite}/cts-api-xml/'
        presp = requests.get(url)
        if presp.ok:
            ptree = etree.fromstring(sub(r'(</p>)', '\\1\n', sub(r'(</seg>)', '\\1 ', presp.text)))
            raw = ptree.find('.//reply/passage', namespaces=ns)
            etree.strip_elements(raw, '{*}note', with_tail=False)
            #print(''.join(raw.itertext()))
            # print(etree.tostring(raw))
            return sub('[ ]*\n[ ]*', '\n', sub('[ \t]+', ' ', ''.join(raw.itertext()).strip())) + '\n'
        else:
            return None
    except:
        print(f'## {cite}', file=sys.stderr)
        return None

def gather_greek_urns(repo_path):
    import os, json
    lines = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".xml") and "grc" in file:
                # Remove the ".xml" extension:
                name_no_ext = file[:-4]
                # Build the URN and also provide a 'work' field to avoid KeyError:
                urn_val = "urn:cts:greekLit:" + name_no_ext
                record = {
                    "urn": urn_val,
                    "work": name_no_ext
                }
                # Convert to JSON so it can be iterated exactly like fileinput lines:
                lines.append(json.dumps(record, ensure_ascii=False))
    return lines

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
        with open("corpora.txt", "a", encoding="utf-8") as txt_file, open("corpora.bin", "ab") as bin_file:
            txt_file.write(text_data + "\n")
            bin_file.write((text_data + "\n").encode("utf-8"))


def main():

    # Use your path to the repo here
    # It needs to have /data at the end
    path = '/Users/dahirou/canonical-greekLit/data'
    lines = gather_greek_urns(path)

    # for line in fileinput.input():
    for line in lines:
        print('running')
        rec = json.loads(line)
        if 'text' in rec:
            if rec['text'] == None:
                rec['text'] = getXML(rec['id'])
            print(json.dumps(rec, ensure_ascii=False))
            continue
        urn = rec['urn']
        reff_url = f'https://scaife-dev.perseus.org/library/{urn}/cts-api-xml/reffs/'
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
            loc = rec['work'] + ':' + cite.split(':')[4]
            text = getXML(cite)
            append_corpora_text(text)
            print(json.dumps({'id': cite,
                            'book': urn,
                            'seq': seq,
                            'loc': loc,
                            'text': text},
                            ensure_ascii=False))
            seq += 1

    print('\n\n\nFINISHED YAYAYAYA\n')

    # if len(sys.argv) > 1:  # If a URN is passed as an argument
    #     urn = sys.argv[1]
    #     print(f"Fetching text for URN: {urn}")
    #     text = getText(urn)
    #     xml_text = getXML(urn)
        
    #     print("\nPlain Text")
    #     print(text if text else "Could not retrieve plain text.")
        
    #     print("\nStripped XML Text")
    #     print(xml_text if xml_text else "Could not retrieve XML text.")
    # else:
    #     print("Please provide a URN as an argument.")

       
if __name__ == "__main__":
    main()