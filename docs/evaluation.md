# Evaluation methodology

## Public data and provenance

The benchmark downloads the **complete corpus**, then takes a seeded sample of
queries. It never filters the corpus to just documents known to be relevant.
Reports include exact query IDs, dataset SHA-256, source URL, model revisions,
configuration, individual results, and timing scope.

| Dataset | Corpus | Labels | Source |
| --- | --- | --- | --- |
| BEIR SciFact | 5,183 scientific abstracts | 300 test queries with human relevance judgments | [BEIR](https://github.com/beir-cellar/beir/wiki/Datasets-available) |
| SQuAD 1.1 development | 2,067 distinct Wikipedia passages | 10,570 human-written questions and answer spans | [Stanford SQuAD](https://rajpurkar.github.io/SQuAD-explorer/) |

Data is distributed under CC BY-SA 4.0 according to the provider dataset cards.
Downloads remain in ignored `data/`; provenance is recorded in committed reports.
Generated report excerpts retain attribution through their source URLs and IDs.
Do not interpret these public domains as representative of an unspecified
organization's confidential documents. Add domain queries, adversarial negatives,
access-control cases, and relevance labels before a domain release.

## Comparisons and metrics

Eight configurations compare fixed/recursive 650-character chunks (80-character
overlap) with vector, indexed full-text, hybrid fusion, and neural-reranked hybrid.
All use the same query sample, full corpus, embedding model, top-k, and character
budget. Metrics measure retrieved document IDs after deduplication in rank order.
Precision@k divides by k even when fewer distinct documents reach the prompt.
For queries with one relevant document, the maximum precision@10 is 0.1;
precision@1, recall@10, MRR@10, and nDCG@10 are easier to interpret together.
Unjudged retrieved sources count as nonrelevant, following the available qrels.

The first request uses a unique tenant/configuration/revision cache key and is
reported as cold; the second is warm. Model weights are loaded before timing.
Ingestion, startup, concurrent load, and model downloads are not query latency.
Bounded per-document context selection affects both recall and token counts.
Context token counts are character/4 estimates rather than billing records.
Embedding caches are keyed by full chunk content and pinned model identity;
reusing them changes setup cost, not the corpus, rankings, or query labels.

## Generation and hallucination measurement

SQuAD reference answers support exact-match and token F1 measurements. A sampled
30-query local FLAN-T5-small run generates actual model responses from retrieved
context. The independent DeBERTa verifier measures unsupported sentence rate against
1-3 sentence evidence windows (see the architecture doc for why).
Reports include generated answers, human reference spans, entailment scores, and
the fraction of answers that a strict grounding gate would retain. Reporting
coverage alongside abstention prevents presenting universal refusal as accuracy.
The small local generator is a reproducible baseline, not the production provider.

Both generator context truncation and verifier truncation limit this experiment.
NLI entailment is an automated support proxy, not a complete factuality metric.
Short answer fragments and composite sentences can be misclassified. Human
adjudication and a production-provider evaluation remain separate requirements.
The optional model-based judge in `app/rag/grounding.py` can additionally score
atomic claims with checked verbatim evidence quotes.

## Reproduce

```bash
python -m evals.public_benchmark --dataset scifact --queries 100
python -m evals.public_benchmark --dataset squad --queries 100 --generate 30
python -m evals.check_reports
```

Set migrated disposable `DATABASE_URL` and `REDIS_URL` first. The runner creates
isolated benchmark tenants and removes their corpus after each strategy. It uses
semantic local models and does not incur generation API charges. Hardware,
electricity, and database/cache costs are excluded from the zero API-cost field.

The committed retrieval gates require at least 100 queries per dataset, all eight
comparisons, recall@10 >= 0.75, nDCG@10 >= 0.45, cold p95 <= 2,000 ms, and warm
p95 <= 100 ms for recursive hybrid plus neural reranking. These are explicit local
engineering acceptance gates, not a universal definition of enterprise quality.
Generation quality requires a separate provider/domain release decision.

Measured results are in [the report directory](../evals/reports/).

The CPU deployment uses pinned TinyBERT-L2 reranking with at most 40 reranked
candidates. A diagnostic profile of the larger MiniLM model is retained in
`evals/reports/reranker_profile.json`; it is not a controlled model quality A/B.
Public retrieval reports record the final model revisions and configuration.

To evaluate your actual configured generation provider, set `OPENAI_API_KEY`
(and `OPENAI_BASE_URL` for private compatible inference), then run:

```bash
python -m evals.public_benchmark --dataset squad --queries 100 --generate 30 --generation-backend configured
```

This explicitly incurs provider charges and writes `squad-production.json`,
separate from the local baseline. Actual provider token usage is recorded per
answer; configured token prices enable generation-cost estimates. The same
independent NLI verifier and human SQuAD references are used. Citation labels
should be stripped when interpreting answer-span metrics on verbose responses.

For generative full-sentence answers, reports also measure whether a normalized
human reference span appears within the answer (`reference_span_hit`). This
avoids treating citation labels or surrounding words as an exact-match failure.
It is a lexical correctness signal, not a semantic truth verdict: a negated or
irrelevant answer can contain the reference. Read it alongside token F1, NLI
support, gate coverage, and case-level evidence.

For the default Compose deployment, `make eval-public` and `make eval-production`
use `scripts/run_evaluation.py`. It runs the evaluator on the private Compose
networks with the restricted database role, and passes credentials via a private
temporary file. No database or Redis ports need to be published. To refresh only
generation after a prompt/verifier change, run:

```bash
python scripts/run_evaluation.py --dataset squad --queries 100 --generate 30 --generation-backend configured --generation-only
```

## Generation model comparison

`evals/reports/generation_models.json` pairs Qwen3 1.7B and Qwen3 4B-Instruct on
the same 30 questions and gate. It also records an ablation of the verifier fix
(4B: gate coverage 0.33 -> 0.77, invalid citations 0.20 -> 0) and a manual
adjudication of delivered answers that missed the lexical reference match.
Refusals are counted as `abstention_rate`; `unsupported_claim_rate_nli` is over
attempted answers only. `delivered_without_reference_answer` is an upper bound
on wrong delivered answers because it is lexical.

## 200-question generation evaluation (development and held-out)

`evals/reports/generation_quality.json` summarizes three 200-question runs:

1. **Development baseline** (seed 42): previous prompt and strict gate.
2. **Development improved** (seed 42): refusal-detection fix, stronger NLI
   escalation for rejected claims, supported-sentence partial delivery, prompt
   rules. Failures on this set were inspected to tune the system.
3. **Held-out final** (seed 7, `--exclude-report` guarantees zero overlap): the
   final configuration on questions never inspected during tuning.

Each run records `accepted_by_strict_policy`, the previous policy's verdict on
the same answer, so gate changes are measured without sampling noise. Delivered
answers without a normalized reference-span match were individually adjudicated;
the reviewer was the AI assistant that built the system (not independent), and
the lists are recorded in the report for audit.

```bash
python scripts/run_evaluation.py --dataset squad --queries 200 --generate 200 \
  --generation-backend configured --generation-only \
  --seed 7 --exclude-report evals/reports/squad-generation-200.json \
  --output evals/reports/squad-generation-200-heldout.json
```

Remaining error sources, in order: model refusal despite retrieved evidence,
grounded answers to ambiguous context-dependent questions, and verifier false
negatives. A larger generator and a question-answer relevance check address the
first two; neither is solved by stricter entailment.
