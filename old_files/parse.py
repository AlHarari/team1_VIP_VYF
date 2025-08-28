import os
import xml.etree.ElementTree as ET

def extract_text_from_xml(file_path):
    # Exception handling for parsing the XML file
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"Error parsing {file_path}: {e}")
        return [], []
    except Exception as e:
        print(f"Unexpected error with {file_path}: {e}")
        return [], []

    text_content = []
    footnotes = []

    # Handle namespaces if necessary
    namespace = ''
    if '}' in root.tag:
        namespace = root.tag.split('}')[0] + '}'

    # Define the set of tags considered as paratext
    paratext_tags = ['note', 'footnote', 'ref', 'fn', 'del', 'add', 'milestone', 'sp']

    def extract_text(element, parent_is_paratext=False):
        for child in element:
            # Remove namespace from tag if present
            tag = child.tag[len(namespace):] if namespace else child.tag
            is_paratext = parent_is_paratext or tag.lower() in paratext_tags

            # Decide where to append the text based on whether it's paratext
            if child.text:
                text = child.text.strip()
                if is_paratext and text:
                    footnotes.append(text)
                elif text:
                    text_content.append(text)

            # Recursively process child elements
            extract_text(child, is_paratext)

            # Handle tail text
            if child.tail:
                tail_text = child.tail.strip()
                if is_paratext and tail_text:
                    footnotes.append(tail_text)
                elif tail_text:
                    text_content.append(tail_text)

    extract_text(root)
    return text_content, footnotes

def process_files(input_directory, output_directory_text, output_directory_footnotes):
    # Ensure output directories exist
    os.makedirs(output_directory_text, exist_ok=True)
    os.makedirs(output_directory_footnotes, exist_ok=True)

    # Walk through the directory tree
    for root_dir, dirs, files in os.walk(input_directory):
        for filename in files:
            # Filter files: names start with 'tlg', end with '.xml', and are not '__cts__.xml'
            if filename.endswith(".xml") and filename != "__cts__.xml" and filename.startswith("tlg") and ('eng' not in filename):
                file_path = os.path.join(root_dir, filename)

                text, footnotes = extract_text_from_xml(file_path)
                if not text and not footnotes:
                    print(f"Skipping {file_path} due to extraction errors.")
                    continue  # Skip to the next file in the loop

                # Creating output file names
                # Preserve the directory structure in the output directories
                relative_dir = os.path.relpath(root_dir, input_directory)
                base_filename = filename[:-4]  # Removes the '.xml' extension

                # Create corresponding output directories
                output_text_dir = os.path.join(output_directory_text, relative_dir)
                output_footnotes_dir = os.path.join(output_directory_footnotes, relative_dir)
                os.makedirs(output_text_dir, exist_ok=True)
                os.makedirs(output_footnotes_dir, exist_ok=True)

                output_file_path = os.path.join(output_text_dir, f"{base_filename}_out.txt")
                output_file_path_pt = os.path.join(output_footnotes_dir, f"{base_filename}_paratext.txt")

                # Write main text content to a file
                with open(output_file_path, 'w', encoding='utf-8') as outfile:
                    outfile.write('\n'.join(text))

                # Write footnotes to a separate file
                with open(output_file_path_pt, 'w', encoding='utf-8') as outfile:
                    outfile.write('\n'.join(footnotes))

                print(f"Processed {file_path}:")
                print(f"  - Text output -> {output_file_path}")
                print(f"  - Footnotes output -> {output_file_path_pt}")

# Specify the directories
input_directory = '/home/hice1/dharden7/scratch/canonical_greeklit/canonical-greekLit/data'
output_directory_text = '/home/hice1/dharden7/scratch/canonical_greeklit/all_out/output_txt'  # Replace with your desired output directory for text
output_directory_footnotes = '/home/hice1/dharden7/scratch/canonical_greeklit/all_out/output_paratxt'  # Replace with your desired output directory for footnotes

process_files(input_directory, output_directory_text, output_directory_footnotes)

