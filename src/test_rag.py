from src.retrieval.retriever_factory import RetrieverFactory
from generation.generator import Generator

def main():
    query = input("Enter query: ").strip()
    if not query:
        print("Query cannot be empty.")
        return

    retriever = RetrieverFactory.create("dense")
    generator = Generator()

    print("\nRetrieving documents...\n")
    results = retriever.search(query, k=5)
    if not results:
        print("No documents retrieved.")
        return

    print("=" * 80)
    print("RETRIEVED SOURCES")
    print("=" * 80)

    for i, result in enumerate(results, start=1):

        print(f"\n[{i}]")
        print(f"Score    : {result.get('score', 0):.4f}")
        print(f"Chunk ID : {result.get('chunk_id', 'N/A')}")
        print(f"Name     : {result.get('name', 'N/A')}")

        text = result.get("text", "")
        print("Content:")
        print(text[:200])
        if len(text) > 200:
            print("...")

    print("\n")
    print("=" * 80)
    print("GENERATED ANSWER")
    print("=" * 80)
    answer = generator.generate(
        query=query,
        contexts=results
    )
    print(answer)


if __name__ == "__main__":
    main()