# Contributing to LlamaFarms

Thank you for your interest in contributing to LlamaFarms! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

This project and everyone participating in it is governed by the LlamaFarms Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to team@llamafarms.ai.

## How Can I Contribute?

### Reporting Bugs

This section guides you through submitting a bug report for LlamaFarms. Following these guidelines helps maintainers understand your report, reproduce the issue, and fix it.

Before creating bug reports, please check the existing issues to avoid duplicating.

**How Do I Submit A Good Bug Report?**

Bugs are tracked as GitHub issues. Create an issue and provide the following information:

* Use a clear and descriptive title
* Describe the exact steps to reproduce the problem
* Provide specific examples to demonstrate the steps
* Describe the behavior you observed after following the steps
* Explain which behavior you expected to see instead and why
* Include screenshots or animated GIFs if possible
* Include details about your configuration and environment

### Suggesting Enhancements

This section guides you through submitting an enhancement suggestion for LlamaFarms, including completely new features and minor improvements to existing functionality.

**How Do I Submit A Good Enhancement Suggestion?**

Enhancement suggestions are tracked as GitHub issues. Create an issue and provide the following information:

* Use a clear and descriptive title
* Provide a detailed description of the suggested enhancement
* Explain why this enhancement would be useful to most LlamaFarms users
* Include screenshots or mockups if applicable
* List some other applications where this enhancement exists, if applicable

### Pull Requests

The process described here has several goals:

* Maintain LlamaFarms's quality
* Fix problems that are important to users
* Enable a sustainable system for LlamaFarms's maintainers to review contributions

Please follow these steps to have your contribution considered by the maintainers:

1. Follow all instructions in the template
2. Follow the style guides
3. After you submit your pull request, verify that all status checks are passing

While the prerequisites above must be satisfied prior to having your pull request reviewed, the reviewer(s) may ask you to complete additional design work, tests, or other changes before your pull request can be ultimately accepted.

## Style Guides

### Git Commit Messages

* Use the present tense ("Add feature" not "Added feature")
* Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
* Limit the first line to 72 characters or less
* Reference issues and pull requests liberally after the first line
* Consider starting the commit message with an applicable emoji:
    * 🎨 `:art:` when improving the format/structure of the code
    * 🐎 `:racehorse:` when improving performance
    * 🚱 `:non-potable_water:` when plugging memory leaks
    * 📝 `:memo:` when writing docs
    * 🐧 `:penguin:` when fixing something on Linux
    * 🍎 `:apple:` when fixing something on macOS
    * 🪟 `:window:` when fixing something on Windows
    * 🧪 `:test_tube:` when adding tests
    * 🔒 `:lock:` when dealing with security

### Python Style Guide

All Python code should adhere to the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.

Additionally:

* Use 4 spaces for indentation (not tabs)
* Use docstrings for modules, classes, and functions
* Use type hints where appropriate
* Maximum line length is 88 characters (compatible with Black)
* Use snake_case for variables and function names
* Use PascalCase for class names
* Use UPPER_CASE for constants

### Documentation Style Guide

* Use Markdown for documentation
* Reference code with backticks: `code`
* Use triple backticks for code blocks with language specification
* Use relative links rather than absolute URLs when linking to other docs

## Development Environment

### Setting Up for Development

```bash
# Clone the repo
git clone https://github.com/yourusername/llamafarms.git
cd llamafarms

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dev dependencies
./install.sh --dev
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/unit/test_mlx_integration.py

# Run with coverage report
pytest --cov=llamafarms
```

## Additional Notes

### Issue and Pull Request Labels

This section lists the labels we use to help us track and manage issues and pull requests.

* `bug` - Issues that are bugs
* `enhancement` - Issues that are feature requests
* `documentation` - Issues or PRs related to documentation
* `good first issue` - Good for newcomers
* `help wanted` - Extra attention is needed
* `wontfix` - Will not be worked on

## Attribution

This Contributing Guide is adapted from the Atom Contributing Guide. 