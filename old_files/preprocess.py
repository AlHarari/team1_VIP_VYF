import os
import re

def concatenate_greek_text(directory, output_text_file, output_utf8_file):
    with open(output_text_file, 'w', encoding='utf-8') as text_outfile, open(output_utf8_file, 'wb') as utf8_outfile:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # Read the content and remove Latin characters using a regex
                        content = infile.read()
                        greek_only_content = re.sub(r'[A-Za-z]', '', content)
                        
                        # Write normal text output to the text file
                        text_outfile.write(greek_only_content)
                        text_outfile.write("\n")  # Add a newline between file contents

                        # Encode the content in UTF-8 and write it as binary to the UTF-8 file
                        utf8_outfile.write(greek_only_content.encode('utf-8', errors='ignore'))
                        utf8_outfile.write(b"\n")  # Add a newline between file contents as bytes
                except Exception as e:
                    print(f"Could not read file {file_path}: {e}")

    print(f"Successfully concatenated Greek text to both {output_text_file} and {output_utf8_file}")

# Example usage
directory_path = "/home/hice1/dharden7/scratch/canonical_greeklit/all_out"
output_text_file_path = "/home/hice1/dharden7/scratch/canonical_greeklit/post_process/corpus.txt"  # Normal text file
output_utf8_file_path = "/home/hice1/dharden7/scratch/canonical_greeklit/post_process/corpus_utf8.bin"  # Binary UTF-8 encoded file
concatenate_greek_text(directory_path, output_text_file_path, output_utf8_file_path)

