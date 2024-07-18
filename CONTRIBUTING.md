# Contributing Guide

Thank you for your interest in contributing to our project! We use a nightly branch for ongoing development and merge into the main branch for major releases. Here are our guidelines:

## Branch Structure

- `main`: Stable branch for releases
- `nightly`: Active development branch

## Development Workflow

1. Fork the repository and clone it locally.
2. Create a new branch from `nightly` for your feature or bugfix:
   ```
   git checkout nightly
   git pull origin nightly
   git checkout -b feature/your-feature-name
   ```
3. Make your changes and commit them with clear, concise commit messages.
4. Push your changes to your fork:
   ```
   git push origin feature/your-feature-name
   ```
5. Open a pull request against the `nightly` branch.

## Nightly Branch Best Practices

- Always base your work on the latest `nightly` branch.
- Regularly sync your fork with the upstream `nightly` branch:
  ```
  git checkout nightly
  git fetch upstream
  git merge upstream/nightly
  git push origin nightly
  ```
- Keep your feature branches short-lived and focused.
- Rebase your feature branch onto `nightly` before submitting a pull request:
  ```
  git checkout feature/your-feature-name
  git rebase nightly
  ```

## Reporting Issues

- Use the issue tracker for bugs or feature requests.
- Check if the same issue already exists before creating a new one.
- Include as much information as possible in your issue report.

## Submitting Pull Requests

- Ensure your code adheres to our coding standards.
- Include unit tests for new features or bug fixes.
- Make sure all tests pass before submitting a pull request.
- Include a clear and detailed description of the changes.
- Link relevant issues in your pull request description.

## Code Review Process

- At least one core maintainer will review your pull request.
- Address any comments or requested changes promptly.
- Once approved, a maintainer will merge your pull request into the `nightly` branch.

## Release Process

- Periodically, we will merge the `nightly` branch into `main` for a new release.
- Contributors should not directly merge or push to the `main` branch.

We appreciate your contributions and look forward to seeing your pull requests!
