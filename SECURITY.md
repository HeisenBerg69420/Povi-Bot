# Security Policy

## Supported Versions

Povi-Bot is currently under active development. Security fixes are applied only to the latest version on the default branch.

| Version | Supported |
| ------- | --------- |
| Latest default branch | Yes |
| Older commits or releases | No |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues, discussions, or pull requests.

Instead, use GitHub’s private vulnerability reporting feature:

1. Open the repository’s **Security** tab.
2. Select **Report a vulnerability**.
3. Provide enough information for the issue to be reproduced and evaluated.

Please include, when available:

- A description of the vulnerability and its potential impact
- The affected version, commit, and operating system
- Clear reproduction steps or a minimal proof of concept
- Relevant logs, screenshots, or stack traces
- Any suggested mitigation
- Whether the vulnerability may already be publicly known

Do not include real credentials, private camera recordings, personal information, signing certificates, or other sensitive data in the report. Use synthetic or redacted examples whenever possible.

## What to Expect

After receiving a report, the maintainers will attempt to:

1. Acknowledge the report.
2. Reproduce and assess the issue.
3. Determine its severity and affected versions.
4. Develop and test a fix when appropriate.
5. Coordinate disclosure with the reporter.

Response and remediation times may vary because this is a small project maintained on a best-effort basis. Please allow reasonable time for investigation before publicly disclosing a vulnerability.

## Security-Sensitive Areas

Reports are especially valuable when they concern:

- Unauthorized camera activation or camera access without visible user consent
- Camera tracks continuing after capture has been stopped
- Camera frames being persisted or transmitted unexpectedly
- Tauri command or capability escalation
- Arbitrary command execution
- Unsafe handling of local files or paths
- Loading modified, untrusted, or incompatible model files
- Dependency or supply-chain vulnerabilities with a demonstrated impact
- Exposure of credentials, certificates, signing material, or personal data
- Installer or update integrity issues

Povi-Bot is designed to keep vision processing local and opt-in. The camera must not start automatically during normal application startup, and captured frames must not be stored or sent over the network by default.

## Model and Dataset Security

Machine-learning model files and datasets should be treated as untrusted inputs unless their origin and integrity have been verified.

Model integrations should document:

- Model origin and license
- Model version
- SHA-256 checksum
- Expected input and output tensors
- Supported dimensions and data types
- Known limitations

Do not commit model weights, camera recordings, private datasets, credentials, certificates, or signing material to this repository.

Misclassifications, low model accuracy, and other expected model-quality limitations are generally not security vulnerabilities unless they can be used to violate a security or privacy boundary.

## Out of Scope

The following are normally not considered security vulnerabilities:

- General bugs without a security or privacy impact
- Expected limitations of prototype motion detection
- Model accuracy problems without a security consequence
- Issues requiring unsupported or intentionally modified builds
- Denial-of-service reports without a realistic attack scenario
- Vulnerabilities only present in outdated dependencies without a demonstrated impact on Povi-Bot
- Social engineering attacks against project maintainers

These issues may still be reported through the regular GitHub issue tracker when they do not contain sensitive information.

## Responsible Disclosure

Please:

- Give the maintainers a reasonable opportunity to investigate and fix the issue.
- Avoid accessing, modifying, or deleting data that does not belong to you.
- Avoid privacy violations, service disruption, and destructive testing.
- Test only on systems and accounts you own or are explicitly authorized to use.
- Share vulnerability details only through the private reporting channel until disclosure has been coordinated.

Thank you for helping keep Povi-Bot and its users safe.
