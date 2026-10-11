# Changelog

## Unreleased

### Documentation

- Reorganize public documentation around installation, architecture, integration,
  reproducibility, and research interpretation.
- Publish citation metadata and clarify copyright/reuse status.
- Move dated internal audits and superseded proposals into a labeled archive.
- Document the model's synthetic evaluation, data availability, and limitations.

### Developer experience

- Add cross-platform experiment reproduction and verification commands.
- Check public comparison metrics directly against published predictions.
- Add Windows CI, synthetic reproduction CI, and a CMake/CTest host build.
- Use portable editor tasks and consistent LF checkout rules.
- Fix Windows model loading affected by CRLF conversion and use Git index modes
  when checking executable permissions.

The model parameters, class semantics, and versioned payload schemas are unchanged.

## 0.2.0 — 2026-09-08

- Include the synthetic ordinal model, provenance, integrity hashes, model card,
  and four calculated inference examples in the Python package.
- Add a file-based integration CLI, operator-confirmed raw-window processing,
  and an optional authenticated HTTP service.
- Add raw-window/service-response schemas and TypeScript interfaces.
- Extend service tests, isolated wheel checks, and container smoke coverage.

This version identifies the Python package. It does not indicate a validated
real-data model, clinical deployment, or a GitHub release.
