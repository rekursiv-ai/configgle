# Security Policy

## Why this file exists

configgle deserializes configuration into live Python objects, and doing so
imports the modules a config names. A config file is therefore closer to code
than to data: deserializing one that an attacker controls can import arbitrary
importable modules and reach their import-time side effects. Security reports
need a private path so exploit details are not published before review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- Deserialization that resolves a dotted path to an object and imports it
  (`Fig.deserialize` and `configgle.lib.codec.from_plain`), including any
  way to widen what a config can name.
- Config sources that reach the deserializer without being trusted first --
  files, environment, or command-line overrides.
- Dependency or packaging issues that affect installed users.
- Supply-chain concerns in the published wheel or its dependency set.

## Out of scope

Deserializing a config you do not trust is equivalent to running code you do
not trust; treat config files with the same care as source files. Reports that
amount to "an untrusted config can import a module" describe the documented
behavior above rather than a vulnerability.
