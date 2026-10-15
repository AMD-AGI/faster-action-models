# Contributing

Thanks for your interest in `faster-action-models`.

## Issues

- Use GitHub issues for bugs, reproducibility problems, and feature requests.
  The issue templates ask for the model and configuration, hardware, ROCm/PyTorch
  versions, and the exact command you ran.
- Report security vulnerabilities privately; see [`SECURITY.md`](SECURITY.md).

## Pull requests

1. Fork the repository and create a topic branch from `main`.
2. Keep each pull request focused on one change, and describe what was run to
   validate it (unit tests, smoke evaluations, hardware).
3. Run the unit tests that cover your change, for example
   `python -m pytest tests/`.
4. Do not commit model weights, datasets, simulator assets, evaluation outputs,
   credentials, or machine-specific absolute paths.
5. Keep license headers intact. New files authored for this repository use:

   ```text
   # Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
   # SPDX-License-Identifier: Apache-2.0
   ```

   When modifying a vendored third-party file, keep its original header and add
   a `Modifications Copyright` line below it; see
   [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

The `src/` trees are frozen records of the exact code that was evaluated; see
[`src/README.md`](src/README.md) before changing them.

## Developer Certificate of Origin

Contributions are accepted under the [Apache License 2.0](LICENSE). Sign off
every commit to certify the [Developer Certificate of Origin](https://developercertificate.org/):

```bash
git commit -s -m "Describe your change"
```

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md).
