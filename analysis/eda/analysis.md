# ATT&CK Corpus Analysis and Evaluation

This analysis is based on the processed ATT&CK corpus in `data/processed/attack_docs.jsonl` and `data/processed/chunks.jsonl`, plus the EDA artifacts generated in this folder.

## 1. Numeric Profile

### Corpus size

- Documents: 1,482
- Chunks: 2,518
- Average chunks per document: 1.7
- Maximum chunks for a single document: 10

### Domain distribution

- `enterprise-attack`: 1,477 documents
- `ics-attack`: 3 documents
- `mobile-attack`: 2 documents

The corpus is overwhelmingly enterprise-focused. Mobile and ICS are present only as a tiny tail, so any downstream analysis or retrieval work will mostly reflect enterprise ATT&CK content.

### ATT&CK type distribution

- `software`: 649
- `sub-technique`: 424
- `technique`: 201
- `group`: 142
- `mitigation`: 43
- `campaign`: 23

The distribution is highly skewed toward software and sub-techniques. That is expected for ATT&CK, but it matters for retrieval and evaluation because the corpus is not balanced across object types.

### Text length distribution

Document description length in characters:

- Minimum: 53
- Median: 522.5
- Mean: 771.8
- Maximum: 4,680

Chunk text length in characters:

- Minimum: 60
- Median: 752.0
- Mean: 746.8
- Maximum: 1,530

The spread is wide. A few records are very short, but many technique and sub-technique descriptions are long enough to require chunking.

## 2. Characteristics of the Data

### Long-form security text

The ATT&CK descriptions are not generic prose. They contain:

- ATT&CK IDs and technique references
- citations and URLs
- code-like tokens such as `WriteProcessMemory` and `schtasks`
- relationship-derived context such as `Used by`, `Mitigated by`, and `Sub-technique of`

This makes the corpus structurally different from normal document collections. The tokenizer sees many acronyms, symbols, and short technical fragments, so character length alone is only a rough proxy for semantic complexity.

### Chunking behavior

Most documents remain a single chunk, but the longest technique and sub-technique records split into several chunks. The documents with the highest chunk counts include:

- `T1553.003`: 10 chunks
- `T1546.004`: 9 chunks
- `T1562.002`: 8 chunks
- `T1547.001`: 8 chunks
- several others with 6 to 7 chunks

That pattern is useful: it means the chunker is only splitting where it needs to, rather than aggressively fragmenting the whole corpus.

### Relationship context richness

The chunked documents usually carry 1 to 3 relationship context lines, with:

- Minimum: 1
- Median: 2
- Mean: 2.17
- Maximum: 3

This is a strong design choice for retrieval because it makes each chunk more self-contained. A query about mitigations, parent techniques, or related groups can match a chunk even when the original description is split across multiple pieces.

## 3. Type-Level Interpretation

### Sub-techniques and techniques are the densest objects

The longest descriptions are concentrated in `sub-technique` and `technique` records:

- Sub-technique description mean: 1,372.5 characters
- Technique description mean: 1,249.4 characters
- Group description mean: 548.8 characters
- Software description mean: 324.4 characters
- Mitigation description mean: 130.9 characters

This is important because it explains why most chunking pressure comes from the ATT&CK behavior objects rather than from groups or mitigations.

### Chunk length by type

Average chunk text length by type:

- `sub-technique`: 807.3 characters
- `software`: 597.0 characters
- `technique`: 856.9 characters
- `group`: 711.8 characters
- `mitigation`: 448.3 characters
- `campaign`: 843.8 characters

The chunk sizes are reasonably consistent, which suggests the recursive chunking strategy is behaving as intended. The very long descriptions are broken down, but the resulting chunks still preserve enough context to remain meaningful.

## 4. Evaluation

### What looks good

- The corpus is large enough to support meaningful retrieval experiments.
- The dominant ATT&CK enterprise slice is internally consistent and well represented.
- The chunking strategy keeps most documents intact while splitting only long records.
- Relationship enrichment makes chunks more useful for RAG because it adds context that often appears in user questions.

### What is less ideal

- The dataset is highly imbalanced toward enterprise content, so the current corpus is not suitable for broad cross-domain ATT&CK comparisons.
- Campaigns and mitigations are relatively sparse, which may make some query classes harder to evaluate reliably.
- Character length is still a rough proxy for token cost; the corpus would benefit from a token-level EDA pass if model input limits become important.

### Practical impact for RAG

- Dense retrieval should work well for technique- and sub-technique-centric questions because those classes have the richest text and the most relationship context.
- Queries about mitigations may need careful retrieval settings because mitigation records are shorter and less numerous.
- The current chunking design is a good compromise between recall and context preservation.

## 5. Conclusion

This corpus is best described as an enterprise ATT&CK knowledge base with a strong skew toward software, sub-techniques, and techniques. The data is structurally rich, long-form, and relationship-heavy, which makes it a good fit for RAG. The current preprocessing pipeline is also sensible: it keeps most records whole, splits only long documents, and attaches enough cross-reference context to improve retrieval quality.

For the next iteration, the most valuable additions would be token-based EDA, retrieval evaluation by attack type, and a comparison of different chunk sizes or embedding models.
