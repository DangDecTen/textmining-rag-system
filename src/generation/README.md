# Stage 03/04 — Generation

Takes `list[RetrievalResult]` from the retrieval stage, builds a bounded
context window, and forces the model to answer in strict JSON
(`{"answer": ..., "found": bool}`), which is then parsed into a
`GenerationResult` and turned into a citation-bearing `Answer`. If the model
can't find the answer in context, it abstains explicitly rather than
guessing. See `src/generation/README.md` for the full design rationale and
how to add a new generator.

```bash
# Interactive CLI: pick a retriever + generator, ask a question
python run_rag.py
python run_rag.py --retriever dense --generator qwen
python run_rag.py --list          # show every registered retriever/generator
```

## Llama Generator

Groq API, [`llama-3.3-70b-versatile`](https://console.groq.com/docs/model/llama-3.3-70b-versatile).

```dotenv
# Set in .env to use this generator.
GROQ_API_KEY=<YOUR_API_KEY>
```

Limits: 131,072 context tokens, 32,768 max output tokens, ~1s latency.

## Qwen Generator

Local, [`Qwen/Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
via raw `transformers`.

```dotenv
# Optional, set in .env to enable faster model downloads.
HF_TOKEN=<YOUR_API_KEY>
```

Limits: 128k context tokens, 8k max output tokens, ~15 min latency on CPU.

## Design decisions (confirmed)

- **Model**: `Qwen/Qwen2.5-1.5B-Instruct`, raw `transformers` (no vLLM/TGI —
  consistent with using lower-level components elsewhere), greedy decoding
  (`do_sample=False`) for reproducible output on a strict-format task.
- **Abstention signal**: structured JSON (`{"answer": ..., "found": bool}`),
  not a canonical refusal phrase. Parsed by `output_parser.py`, which is
  deliberately fail-safe: any parse failure (malformed JSON, no JSON found,
  missing/null fields) is treated as `found=False`, never as a best-guess
  answer. In a cybersecurity QA tool, an incorrect abstention is a much
  cheaper mistake than a confident wrong answer — verified this fail-safe
  path explicitly in testing (see below).
- **`GenerationResult` gained a `found: bool` field** beyond what you
  originally specified — flagging this explicitly since I changed your
  schema rather than silently going around it. Rationale: the model already
  signals abstention structurally; collapsing that back into string-matching
  on `answer` downstream would throw the signal away.
- **`ContextBuilder`** dedupes by `doc_id`, sorts by score, and enforces a
  token budget (`max_context_tokens`, default 1500) using the *same*
  tokenizer the generator will use — not an approximate token count.

## Evaluation angle worth using (stage04)

AttackQA has no native "unanswerable" examples — every question was
generated from an existing document, so every question is answerable *if*
the right document is retrieved. The only realistic way to evaluate
abstention is to use **retrieval failures as a natural unanswerable set**:
for dev questions where the retriever's top-k did NOT include the
ground-truth doc_id, correct behavior is to abstain, so abstention rate
conditioned on retrieval failure is effectively a hallucination-rate metric.
For questions where retrieval succeeded, evaluate normally (EM/F1 against
the short reference `answer`). This connects stage01's retrieval eval
directly to stage04's generation eval — worth building the eval script this
way when we get there, and worth a slide in the presentation.

## Known limitation / next check

No network here to install `torch`/`transformers`, so `QwenGenerator` was
tested against hand-written stand-ins for `AutoTokenizer`/
`AutoModelForCausalLM`/`generate()`. This verified: prompt construction
(system + user messages via `apply_chat_template`), accurate token counting,
and — most importantly — all three generator outcomes end-to-end through
`ResponseBuilder`: a valid `found=true` JSON response, an explicit
`found=false` response, and a malformed/unparseable response, confirming the
fail-safe-to-abstain path actually works rather than just existing in code.
**Not tested**: real Qwen2.5-1.5B-Instruct output quality or actual JSON-format
adherence rate — that's the real open question once you run this for real.
Small instruct models don't always follow strict-JSON instructions reliably;
if you see a meaningful fraction of dev questions falling into the
fail-safe-abstain path due to malformed output (not genuine "not found"), that's
worth flagging back to me — the fix is probably a stronger worked example in
the prompt, not a bigger model.

## Registry (`registry.py`) and adding a new generator

Both `LlamaGenerator` and `QwenGenerator` implement the `Generator`
interface (`base.py`: `.generate(question, contexts) -> GenerationResult`)
and are registered with a decorator:

```python
@register_generator("llama")
class LlamaGenerator(Generator):
    ...
```

`build_generator(name, **kwargs)` instantiates a registered class by name;
`available_generators()` lists every registered name (what powers `GET /` in
the API and the Streamlit dropdown). See `src/retrieval/README.md` for the
full rationale on why this is a self-registering registry rather than a
hand-maintained `if/elif` factory — the short version: the old factory
silently drifted out of sync with the classes it was supposed to build, and
a registry makes that class of bug structurally impossible, since the class
declares its own name at the point it's defined.

**To add a new generator** (a different hosted LLM API, a different local
model):

1. New file, e.g. `src/generation/openai_generator.py`, subclassing
   `Generator` and reusing `ContextBuilder` / `output_parser.parse_structured_output`
   / `prompt.SYSTEM_PROMPT` exactly as `llama_generator.py` and
   `qwen_generator.py` do — the prompt, context-budgeting, and fail-safe JSON
   parsing are all model-agnostic; a new generator should not need to
   reimplement them.
2. Decorate the class: `@register_generator("your_name")`.
3. Add one import line to the "side-effect imports" block in
   `src/factory.py` so the decorator actually runs.
4. Add a branch in `get_generator()` in `src/factory.py` (mirroring the
   `llama`/`qwen` branches) for which config fields it needs, and add any
   new fields (model name, API key env var, etc.) to `src/config.py`.

After that, `python run_rag.py --generator your_name`, `POST /query
{"generator": "your_name", ...}`, and the Streamlit dropdown all work with
no further changes.