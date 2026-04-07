#!/usr/bin/env python3

import argparse
from pathlib import Path
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="/storage/ice-shared/vip-vyf/embeddings_team/corpora/consolidated_corpus",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/storage/ice1/6/7/dharden7/greekbert_proj/tokenizer_grc_wordpiece_32895",
    )
    parser.add_argument("--vocab_size", type=int, default=32895)
    parser.add_argument("--min_frequency", type=int, default=2)
    args = parser.parse_args()

    corpus_dir = Path('/storage/ice-shared/vip-vyf/embeddings_team/corpora/consolidated_corpus')
    files = sorted(str(p) for p in corpus_dir.rglob("*.txt"))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[LB]"]

    # Train WordPiece tokenizer directly from the UTF-8 text file
    tokenizer = BertWordPieceTokenizer(
        clean_text=False,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )

    start = time.perf_counter()

    tokenizer.train(
        files=files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )

    end = time.perf_counter()
    print(f'\nTRAINING FINISHED: {end-start:.2f} seconds')

    # Save vocab.txt
    tokenizer.save_model(str(out_dir))

    # Save full Hugging Face tokenizer files so you can later do:
    # BertTokenizerFast.from_pretrained(out_dir)
    hf_tokenizer = BertTokenizerFast(
        vocab_file=str(out_dir / "vocab.txt"),
        do_lower_case=False,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        additional_special_tokens=["[LB]"]
    )
    hf_tokenizer.save_pretrained(str(out_dir))


    query_tokens = ["[LB]", "##ρον"]

    print("\nToken queries:")
    for tok in query_tokens:
        raw_id = tokenizer.token_to_id(tok)
        hf_id = hf_tokenizer.convert_tokens_to_ids(tok)

        print(f"{tok}")
        print(f"  raw tokenizer id: {raw_id}")
        print(f"  hf tokenizer id:  {hf_id}")

        if raw_id is None:
            print("  -> token not in learned vocab")
        else:
            print("  -> token is in learned vocab")

        print(f"Saved tokenizer to: {out_dir}")
        print(f"Vocab size: {hf_tokenizer.vocab_size}")
        print(hf_tokenizer.tokenize("ἄνδρα μοι ἔννεπε, μοῦσα, πολύτροπον."))



if __name__ == "__main__":
    main()
    
