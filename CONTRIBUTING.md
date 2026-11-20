# Contributing to GreenLiT

Thank you for your interest in contributing to GreenLiT! This document provides guidelines and instructions for contributing to the project.

When adding new features or making significant changes, please also update all documentation accordingly. This includes:

- Updating the README in the main repository if necessary
- Adding or updating documentation in the `docs/` directory
- Adding comments to your code

## Commit Message Format

Commits should follow the standard [Conventional Commits](https://www.conventionalcommits.org/) format.

Line-length checks are defined in [.gitlint](.gitlint).

**Summary line** (max 120 characters, ideally under 50):
```text
<type>(<scope>): [<Optional-Internal-Ticket>] <short-summary-text>
```

- `<type>`: `fix`, `feat`, `chore`, `refactor`, `style`, `test`, `docs`, `ci`
- `<scope>`: Optional, indicates the area of the codebase (e.g., `pci`, `iommu`, `docs`)
- `<Optional-Internal-Ticket>`: For internal NVIDIA contributors, please continue to include the NVBug (`nv#XXXXXXX`) or Jira (`DGX-XXXX`) ticket number. **For community contributors, please omit this and keep the summary line as short as possible.**

**Examples:**
```text
fix(pci): resolve null pointer exception during reset
feat: [DGX-8676] add support for new platform detection
style: [nv#0000000] whitespace cleanup
```

**Description** (for non-trivial changes):
```text
fix(pci): resolve null pointer exception during reset

* Detailed explanation of the change.
* Additional bullet points as needed.

Fixes #123

Changelog: fix
```

- **GitHub Issues:** If your commit resolves an open GitHub issue, please reference it at the bottom of the description using the `Fixes #123` or `Resolves #123` format. GitHub will automatically link and close the issue when merged.
- **Changelog trailer:** Append `Changelog: <type>` to enable automatic release note generation.

## Developer Certificate of Origin (DCO)

All contributions to this project must be accompanied by a Developer Certificate of Origin (DCO). The DCO is a legally binding statement that asserts that you are the creator of your contribution, and that you have the authority to submit the contribution under the open source license associated with the project.

You can read the full text of the DCO here: [https://developercertificate.org/](https://developercertificate.org/)

To indicate that you agree to the terms of the DCO, you **must sign off all your commits**.

You can do this easily by adding the `-s` or `--signoff` flag to your `git commit` command:

```bash
git commit -s -m "..."
```

This will automatically append a `Signed-off-by` trailer to your commit message, using your configured Git name and email:

```text
Signed-off-by: Your Name <your.email@example.com>
```

**Note:** The name and email in the `Signed-off-by` line must match the author of the commit. Contributions without a valid DCO sign-off cannot be accepted.

Full text of the DCO (<https://developercertificate.org/>):

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```
