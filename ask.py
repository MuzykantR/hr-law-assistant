"""Command-line interface.

    uv run python ask.py "Сколько дней отпуска мне положено?"
    uv run python ask.py "..." --mode bm25 --show-context
"""

import argparse

from rag.assistant import build_assistant


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the Labour Code of the RF")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="dense")
    parser.add_argument("--show-context", action="store_true", help="print retrieved fragments")
    args = parser.parse_args()

    assistant = build_assistant(args.mode)
    answer = assistant.ask(args.question)
    if answer.query != args.question:
        print(f"(запрос исправлен: «{answer.query}»)\n")
    print(answer.text)
    if answer.sources:
        print("\nИсточники:")
        for c in answer.sources:
            print(f"  [{c.citation}] {c.title}")
            if args.show_context:
                print("    " + c.text.replace("\n", "\n    "))


if __name__ == "__main__":
    main()
