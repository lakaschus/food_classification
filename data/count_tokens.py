#!/usr/bin/env python3
import csv
import argparse
import os
from typing import Dict, List, Optional


def count_tokens_in_csv(
    csv_path: str,
    columns: Optional[List[str]] = None,
    delimiter: str = ",",
    tokens_per_word: float = 1.25,
) -> Dict:
    """
    Count approximate tokens in a CSV file.

    Args:
        csv_path: Path to the CSV file
        columns: List of column names to count tokens from (if None, count all columns)
        delimiter: CSV delimiter character
        tokens_per_word: Approximation factor for tokens per word

    Returns:
        Dictionary with token count statistics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    total_words = 0
    total_rows = 0
    column_word_counts = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        try:
            reader = csv.DictReader(f, delimiter=delimiter)

            # If no columns specified, use all columns
            if not columns:
                columns = reader.fieldnames

            # Initialize column counts
            for col in columns:
                column_word_counts[col] = 0

            # Process each row
            for row in reader:
                total_rows += 1

                for col in columns:
                    if col in row:
                        # Count words in the cell
                        words_in_cell = len(str(row[col]).split())
                        total_words += words_in_cell
                        column_word_counts[col] += words_in_cell

        except Exception as e:
            print(f"Error processing CSV: {e}")
            return {}

    # Calculate token estimates
    total_tokens = int(total_words * tokens_per_word)
    column_token_counts = {
        col: int(count * tokens_per_word) for col, count in column_word_counts.items()
    }

    return {
        "total_rows": total_rows,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "tokens_per_row_avg": int(total_tokens / total_rows) if total_rows > 0 else 0,
        "column_word_counts": column_word_counts,
        "column_token_counts": column_token_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Count approximate tokens in a CSV file"
    )
    parser.add_argument("csv_file", help="Path to the CSV file")
    parser.add_argument(
        "--columns",
        help="Comma-separated list of columns to count (default: all columns)",
        default=None,
    )
    parser.add_argument(
        "--delimiter", help="CSV delimiter character (default: comma)", default=","
    )
    parser.add_argument(
        "--tokens-per-word",
        help="Approximation factor for tokens per word (default: 1.25)",
        type=float,
        default=1.25,
    )

    args = parser.parse_args()

    columns = args.columns.split(",") if args.columns else None

    try:
        results = count_tokens_in_csv(
            args.csv_file,
            columns=columns,
            delimiter=args.delimiter,
            tokens_per_word=args.tokens_per_word,
        )

        if results:
            print(f"\nToken Count Results for {args.csv_file}")
            print(f"{'='*50}")
            print(f"Total rows: {results['total_rows']}")
            print(f"Total words: {results['total_words']}")
            print(f"Estimated total tokens: {results['total_tokens']}")
            print(f"Average tokens per row: {results['tokens_per_row_avg']}")

            print("\nToken counts by column:")
            for col, count in results["column_token_counts"].items():
                print(
                    f"  - {col}: {count} tokens ({results['column_word_counts'][col]} words)"
                )

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
