# Secret Handling

Never commit credentials, access tokens, passwords, private keys, local environment files, or raw
user histories. Use the operating-system credential manager, a local ignored environment file, or
GitHub Actions secrets.

An environment example may contain variable names and obvious placeholders only. Do not print
secrets in notebooks, screenshots, shell history, logs, test output, or CI artifacts.

GitHub Actions should use read-only permissions by default. Fork pull requests must not receive
repository secrets.

Use synthetic data in tests. Keep the full dataset, generated Parquet, checkpoints, and other large
or sensitive artifacts outside normal Git history.

If a secret is exposed:

1. Rotate or revoke it immediately.
2. Notify the team privately.
3. Remove it from the current tree and affected logs or artifacts.
4. Rewrite shared history only with the repository administrator.

The ignore file helps prevent mistakes but is not an access-control boundary.
