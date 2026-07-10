# Release Checklist

Before pushing this repository publicly:

1. Run `git status --ignored` and confirm that raw data, logs, checkpoints,
   generated outputs, notes, PDFs, and local presentations are ignored.
2. Run a secret scan for API keys and credentials.
3. Confirm that examples are synthetic and do not contain patient-derived text.
4. Confirm that closed-source API configuration is documented but not committed.
5. Confirm that final patient records are described as requiring
   `verified_llm_cache`, `fallback=0`, and `hard_errors=0`.
6. Confirm that the target GitHub repository visibility is intentional.
