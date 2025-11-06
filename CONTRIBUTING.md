# Contributing to Phase Runner

## Commit convention

All commits use the format `[foundation] 0.x <summary> ✅`. Include a succinct
summary of the change and keep commits atomic.

## Development workflow

1. Create a feature branch (e.g. `phase/foundation`).
2. Run `make bootstrap` to install toolchains.
3. Implement changes and ensure quality gates succeed:
    - `make lint`
    - `make format-check`
    - `make type-check`
    - `make test`
    - `make scan`
    - `make pii-scan`
4. Submit a pull request and request review from the code owners listed in
   [`CODEOWNERS`](CODEOWNERS).

## Code style

- Use Prettier and ESLint for JavaScript/TypeScript sources.
- Use Ruff and Mypy for Python packages.
- Prefer small, focused modules with explicit exports and type annotations.

## Pull request checklist

- [ ] Linked issue and changelog entry updated.
- [ ] Tests cover new behavior.
- [ ] Documentation and runbooks reflect operational changes.
- [ ] CI is green before requesting review.
