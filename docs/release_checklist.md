# Release Checklist

Before pushing this repository publicly:

1. Run `git status --ignored` and confirm that logs, checkpoints, generated
   outputs, notes, PDFs, local presentations, and credentials are ignored.
2. Run a secret scan for API keys and credentials.
3. Confirm that included `data/` files are allowed by the public dataset license.
4. Confirm that examples are synthetic and do not contain accidental private
   records.
5. Confirm that closed-source API configuration is documented but not committed.
6. Confirm that final patient records are described as requiring
   `verified_llm_cache`, `fallback=0`, and `hard_errors=0`.
7. Confirm that Git LFS is enabled for large JSONL files.
8. Confirm that the target GitHub repository visibility is intentional.
