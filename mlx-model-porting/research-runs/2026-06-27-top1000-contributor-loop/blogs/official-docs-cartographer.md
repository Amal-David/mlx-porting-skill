# Official Docs Cartographer Research Blog

## Assignment
Find MLX or Apple-supported APIs, runtime constraints, packaging rules, and validation gates that should constrain porting guidance.

## Sources sampled
- GitHub list repository contributors API (https://docs.github.com/en/rest/repos/repos#list-repository-contributors, accessed 2026-06-27)
- GitHub contributors API for ml-explore/mlx (https://api.github.com/repos/ml-explore/mlx/contributors?per_page=100&page=1, accessed 2026-06-27)
- GitHub REST API pagination (https://docs.github.com/en/rest/using-the-rest-api/using-pagination-in-the-rest-api, accessed 2026-06-27)
- GitHub REST API best practices (https://docs.github.com/en/rest/using-the-rest-api/best-practices-for-using-the-rest-api, accessed 2026-06-27)

## Candidate findings
- top1000-contributor-set-covered-available-api: Top-1000 request covered API-returned linked contributors and recorded the anonymous delta [adopted] - The expanded sweep requested ml-explore/mlx contributors pages 1-10 at 100 per page. GitHub returned 256 linked contributors across pages 1-3, or 262 author buckets with anon=true, so the top-1000 objective covered every linked contributor available through that API at retrieval time.
- top1000-link-header-collector-receipts: Contributor-scale sweeps need Link-header and rate-limit receipts [needs-validation] - GitHub pagination and API best-practice docs require following Link headers, using authenticated serial requests, and preserving rate-limit/conditional-request receipts. Hard-coded page caps are safety caps, not proof that all pages exist.

## Decision notes
- Contributor sweeps are repository-selection evidence, not API-support evidence; keep promotion tied to pinned source files and local gates.

## Open validation
- top1000-contributor-set-covered-available-api: Re-run contributor retrieval with authenticated requests and compare linked-user plus anon=true counts before claiming refreshed top-1000 coverage.
- top1000-link-header-collector-receipts: Add or run a collector that saves Link headers, ETags, Last-Modified values, retry records, and rate-limit receipts.
