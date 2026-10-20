# Repository Protection

The main branch should use these GitHub settings:

- require a pull request but no mandatory approval;
- allow the author to merge after validation passes;
- apply the rule to administrators with no bypass;
- block force pushes and branch deletion.

Do not require a status check until #22 creates a real CI check and it succeeds at least once. Then
require only that stable check.

Enable secret scanning, push protection, dependency alerts, private vulnerability reporting, and
read-only default GitHub Actions permissions. CodeQL and automated dependency-update PRs can wait
until the initial CI pipeline is stable.

Review is optional for scientific protocol changes and shared contracts. These platform settings
keep changes visible and reversible without making team members wait on each other.
