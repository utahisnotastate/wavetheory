# Contributing to Wave Theory Chatbot

Thank you for your interest in contributing to the Wave Theory Chatbot! This document provides guidelines and information for contributors.

## 🤝 How to Contribute

### Types of Contributions

We welcome several types of contributions:

- **🐛 Bug Reports** - Help us identify and fix issues
- **✨ Feature Requests** - Suggest new functionality
- **📚 Documentation** - Improve guides, tutorials, and API docs
- **🧪 Testing** - Add tests or improve existing ones
- **🎨 UI/UX** - Enhance the user interface and experience
- **🔬 Physics** - Add new physics experiments or validation tests
- **⚡ Performance** - Optimize code and improve efficiency
- **🌐 Internationalization** - Add support for different languages

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+**
- **Git** for version control
- **Docker** (optional, for containerized development)
- **8GB+ RAM** (16GB recommended)

### Development Setup

1. **Fork the repository**
   ```bash
   # Go to https://github.com/yourusername/wave-theory-chatbot
   # Click "Fork" button
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/yourusername/wave-theory-chatbot.git
   cd wave-theory-chatbot
   ```

3. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/original-owner/wave-theory-chatbot.git
   ```

4. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

6. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

7. **Run tests to verify setup**
   ```bash
   make test
   ```

## 🔧 Development Workflow

### Branch Strategy

- **`main`** - Production-ready code
- **`develop`** - Integration branch for features
- **`feature/feature-name`** - New features
- **`bugfix/bug-name`** - Bug fixes
- **`hotfix/hotfix-name`** - Critical fixes
- **`docs/documentation-name`** - Documentation updates

### Creating a Feature Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... edit files ...

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add new physics experiment for quantum tunneling"

# Push to your fork
git push origin feature/your-feature-name
```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements
- `ci`: CI/CD changes

**Examples:**
```bash
feat(analytics): add real-time performance monitoring
fix(simulation): resolve energy conservation issue
docs(api): update PINN training documentation
test(validation): add physics conservation tests
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_simulation.py -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# Run physics validation suite
python -m src.testing.validation_suite

# Run integration tests
python -m src.testing.integration_test_suite
```

### Writing Tests

1. **Test Structure**
   ```python
   import pytest
   from src.simulation import Universe, WaveTheoryForce
   
   class TestSimulation:
       def test_energy_conservation(self):
           """Test that energy is conserved in simulation."""
           # Arrange
           force_calc = WaveTheoryForce(G=1.0, wave_freq=0.5, decay_length=10.0)
           universe = Universe(force_calc, dt=0.01)
           
           # Act
           universe.add_body(Body([0, 0, 0], [0.5, 0, 0], mass=5.0))
           history = universe.run_simulation(steps=100)
           
           # Assert
           initial_energy = history[0]['energy']['total']
           final_energy = history[-1]['energy']['total']
           assert abs(final_energy - initial_energy) < 0.01
   ```

2. **Test Categories**
   - **Unit Tests** - Test individual functions/classes
   - **Integration Tests** - Test component interactions
   - **Physics Validation** - Test physical accuracy
   - **Performance Tests** - Test speed and memory usage
   - **UI Tests** - Test user interface functionality

3. **Test Requirements**
   - Tests should be **deterministic** (same input = same output)
   - Tests should be **fast** (< 1 second for unit tests)
   - Tests should be **isolated** (no dependencies between tests)
   - Tests should have **clear assertions** with descriptive messages

## 📝 Code Style Guidelines

### Python Style

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Good
def calculate_wave_force(particle1: Dict[str, Any], 
                        particle2: Dict[str, Any],
                        physics_params: Dict[str, float]) -> np.ndarray:
    """
    Calculate Wave Theory force between two particles.
    
    Args:
        particle1: First particle data
        particle2: Second particle data
        physics_params: Physics parameters (G, wave_freq, decay_length)
    
    Returns:
        Force vector as numpy array
    """
    # Implementation here
    pass

# Bad
def calcForce(p1,p2,params):
    # Implementation here
    pass
```

### Code Formatting

```bash
# Format code with black
make format

# Check linting
make lint

# Auto-fix some issues
black src/ tests/
flake8 src/ tests/
```

### Documentation Style

1. **Docstrings** - Use Google style:
   ```python
   def train_pinn_model(model: WavePINN, 
                       data: Dict[str, np.ndarray],
                       epochs: int = 1000) -> Dict[str, Any]:
       """
       Train a Physics-Informed Neural Network.
       
       Args:
           model: PINN model to train
           data: Training data dictionary with keys 'inputs', 'targets'
           epochs: Number of training epochs
       
       Returns:
           Training history dictionary with loss values
       
       Raises:
           ValueError: If data format is invalid
       """
   ```

2. **Comments** - Explain **why**, not **what**:
   ```python
   # Good: Explains reasoning
   # Use adaptive sampling to focus on high-residual regions
   if self.config.adaptive_sampling:
       collocation_points = self._sample_adaptive_points()
   
   # Bad: States the obvious
   # Increment the counter
   counter += 1
   ```

## 🎯 Contribution Areas

### High Priority

- **🧪 Physics Experiments** - Add new pre-configured experiments
- **📊 Analytics** - Enhance monitoring and reporting
- **🎨 Visualizations** - Add new visualization modes
- **🧪 Testing** - Improve test coverage and validation
- **📚 Documentation** - Improve guides and tutorials

### Medium Priority

- **⚡ Performance** - Optimize simulation and training
- **🌐 Internationalization** - Add multi-language support
- **🔧 UI/UX** - Enhance user interface
- **🐳 DevOps** - Improve deployment and CI/CD

### Low Priority

- **🎵 Audio** - Add sound effects for simulations
- **📱 Mobile** - Mobile app development
- **🌍 Cloud** - Cloud deployment improvements

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Clear Description** - What happened vs. what you expected
2. **Steps to Reproduce** - Exact steps to reproduce the issue
3. **Environment** - OS, Python version, dependencies
4. **Error Messages** - Full error traceback
5. **Screenshots** - If applicable
6. **Minimal Example** - Smallest code that reproduces the issue

**Bug Report Template:**
```markdown
## Bug Description
Brief description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. Scroll down to '...'
4. See error

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- OS: [e.g., Windows 10, macOS 12, Ubuntu 20.04]
- Python: [e.g., 3.10.5]
- Browser: [e.g., Chrome 95, Firefox 94]
- Version: [e.g., 2.0.0]

## Additional Context
Any other context about the problem
```

## ✨ Feature Requests

When requesting features, please include:

1. **Problem Statement** - What problem does this solve?
2. **Proposed Solution** - How should it work?
3. **Alternatives Considered** - Other approaches you've thought about
4. **Use Cases** - Who would benefit from this?
5. **Implementation Ideas** - Any technical suggestions

**Feature Request Template:**
```markdown
## Feature Description
Brief description of the feature

## Problem Statement
What problem does this feature solve?

## Proposed Solution
How should this feature work?

## Use Cases
Who would benefit from this feature?

## Implementation Ideas
Any technical suggestions or considerations

## Additional Context
Any other context or screenshots
```

## 🔍 Code Review Process

### Review Checklist

**For Contributors:**
- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] No sensitive data is included
- [ ] Performance impact is considered

**For Reviewers:**
- [ ] Code is correct and efficient
- [ ] Tests are comprehensive
- [ ] Documentation is clear
- [ ] No security vulnerabilities
- [ ] Follows project conventions
- [ ] Breaking changes are documented

### Review Timeline

- **Initial Review** - Within 48 hours
- **Follow-up Reviews** - Within 24 hours
- **Final Approval** - Within 72 hours

## 🚀 Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR** - Breaking changes
- **MINOR** - New features (backward compatible)
- **PATCH** - Bug fixes (backward compatible)

### Release Checklist

- [ ] All tests passing
- [ ] Documentation updated
- [ ] Version number incremented
- [ ] CHANGELOG.md updated
- [ ] Release notes prepared
- [ ] Docker images built
- [ ] GitHub release created

## 📞 Getting Help

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Email** - your.email@example.com
- **Discord** - [Join our Discord server](https://discord.gg/your-server)

### Response Times

- **Critical Bugs** - Within 24 hours
- **Feature Requests** - Within 1 week
- **General Questions** - Within 3 days
- **Documentation Issues** - Within 5 days

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** - Contributor list
- **Release Notes** - Feature acknowledgments
- **Documentation** - Author credits
- **GitHub** - Contributor statistics

## 📄 Legal

### Contributor License Agreement

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

### Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## 🙏 Thank You

Thank you for contributing to the Wave Theory Chatbot! Your contributions help advance physics education and research through neuro-symbolic discovery.

---

**Questions?** Feel free to reach out via [GitHub Issues](https://github.com/yourusername/wave-theory-chatbot/issues) or [Discussions](https://github.com/yourusername/wave-theory-chatbot/discussions).

*Together, we're building the future of physics discovery!* 🌊✨
