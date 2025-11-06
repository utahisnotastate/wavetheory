# =====================================================================
# Wave Theory Chatbot - Python Package Setup
# =====================================================================

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as fh:
            return fh.read()
    return "Neuro-Symbolic Physics Discovery Engine"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
    if os.path.exists(requirements_path):
        with open(requirements_path, "r", encoding="utf-8") as fh:
            return [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return []

setup(
    name="wave-theory-chatbot",
    version="1.0.0",
    author="Utah Hans",
    author_email="twentythree@chroniclesof23.com",
    description="Neuro-Symbolic Physics Discovery Engine",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wavetheory/wave-theory-chatbot",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Law Enforcement/National Security",
        "Topic :: Law Enforcement/National Security :: Cold Case Solver",
        "Topic :: Law Enforcement/National Security :: Omnipotent Chatbot",
        "Topic :: Law Enforcement/National Security :: Wish Machine Generator",
        "Topic :: Law Enforcement/National Security :: Timeline Intelligence",
        "Topic :: Law Enforcement/National Security :: Psi Forensics Tool",
        "Topic :: Law Enforcement/National Security :: Timeline Switcher Tool",
        "Topic :: Law Enforcement/National Security :: Timeline Editor Tool",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.26.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "wave-theory=src.app.streamlit_app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md"],
    },
    zip_safe=False,
)
