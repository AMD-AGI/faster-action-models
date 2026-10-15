# Security Policy

## Reporting a vulnerability

Please **do not** report security vulnerabilities through public GitHub issues,
discussions, or pull requests.

Report suspected vulnerabilities to the AMD Product Security Incident Response
Team (PSIRT):

- Email: [psirt@amd.com](mailto:psirt@amd.com)
- Web: <https://www.amd.com/en/resources/product-security.html>

Include as much of the following as you can:

- the affected file(s), commit, and configuration;
- steps to reproduce, or a proof of concept;
- the impact you expect an attacker could achieve.

## Scope and deployment notes

This repository contains research code. The policy servers under
`DiT4DiT/deployment/` and `src/dit4dit-*/deployment/` (and the evaluation
servers they talk to) are unauthenticated websocket services intended for a
trusted, local network. They listen on all interfaces (`host="0.0.0.0"` in
`server_policy.py`) and return Python tracebacks to the client on error. Do not
expose them to untrusted networks: restrict the port with a firewall or tunnel,
or change the host to `127.0.0.1` when client and server share a machine.

Third-party code vendored in this repository (see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)) is maintained upstream;
issues that also affect the upstream project should be reported there as well.
