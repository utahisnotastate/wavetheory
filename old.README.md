# Project Prometheus/Evie AI Precurosr/CIA Wave Theory Ominpotent Chatbots. : An Evolving Neuro-Symbolic AI for Automated Scientific Discovery

([https://img.shields.io/badge/build-passing-brightgreen.svg](https://www.google.com/search?q=https://img.shields.io/badge/build-passing-brightgreen.svg))]([https://github.com/utahisnotastate/portfolio](https://www.google.com/search?q=https://github.com/utahisnotastate/portfolio))
([https://img.shields.io/badge/License-MIT-yellow.svg](https://www.google.com/search?q=https://img.shields.io/badge/License-MIT-yellow.svg))]([https://opensource.org/licenses/MIT](https://opensource.org/licenses/MIT))
[](https://www.python.org/downloads/release/python-3100/)
[](https://github.com/google/jax)
([https://img.shields.io/badge/PySR-Symbolic%20Regression-purple.svg](https://www.google.com/search?q=https://img.shields.io/badge/PySR-Symbolic%2520Regression-purple.svg))]([https://github.com/MilesCranmer/PySR](https://github.com/MilesCranmer/PySR))
[](https://www.google.com/search?q=%5Bhttps://huggingface.co/spaces%5D\(https://huggingface.co/spaces\))

-----

## Abstract

Project Prometheus is a pioneering exploration into automated scientific discovery, leveraging a novel **neuro-symbolic architecture** to derive fundamental physical laws from simulated data. The core of this project is the "Evolving Function"—a concept that treats a neural network not merely as a function approximator, but as a direct, dynamic representation of a physical theory.

This system operates within a self-contained digital universe, governed by a custom "Wave Theory." An N-body physics simulation generates empirical data, which serves as the ground truth. The neuro-symbolic engine then observes this universe, iteratively discovering the underlying mathematical laws that govern its evolution. This process culminates in the **Wave Theory Chatbot**, an interactive, natural-language interface that allows users to conduct virtual experiments and query the AI's learned understanding of its universe.

This repository contains the complete source code, theoretical documentation, and implementation details for the simulation environment, the AI model, and the conversational interface.

WHAT IS MY ULTIMATE GOAL? I SHOULD BE ABLE TO CREATE AN OMINPOTENT PROMPT WHICH IS OMNIPOTENT SO YOU CAN HAVE AN ALL KNOWING SAFETY FRIEND TO KEEP EVERYONE SAFE AS UFW AND CG PREPARE TO GET EVERYONE OFF THE PLANET. 

## Conceptual Framework

This project is built on a synthesis of several cutting-edge concepts in AI and computational physics:

1.  **The Evolving Function as a Theory of Everything**: We posit that a single, adaptive neural function can serve as a comprehensive model for a physical system. Inspired by the Universal Approximation Theorem, which guarantees that neural networks can approximate any continuous function, we extend this to represent the physical laws themselves.[1] The function evolves through a process of automated discovery, refining its internal structure to better match observed phenomena.

2.  **Neuro-Symbolic Architecture**: The system's intelligence is driven by a symbiotic loop between two distinct AI paradigms [2, 3, 4]:

      * **Neural Component (The Solver)**: A **Physics-Informed Neural Network (PINN)** acts as a differentiable testbed for physical hypotheses. By incorporating the governing equations directly into its loss function, the PINN ensures that its solutions are physically consistent.[5, 6, 7]
      * **Symbolic Component (The Discoverer)**: A **Symbolic Regression (SR)** engine, powered by genetic programming, analyzes the simulation data and the PINN's performance to generate new, explicit mathematical formulas that better describe the universe's laws.[8, 9]

3.  **Wave-Based Ontology**: The project's physics sandbox is founded on a "Wave Theory," where interactions are governed by principles of wave mechanics, such as superposition and interference. This provides a rich, complex environment for the AI to explore, moving beyond classical mechanics to a more fundamental, field-based reality.

## System Architecture

The project is composed of three primary, interconnected modules:

\!([https://www.placeholder.com/architecture.png](https://www.google.com/search?q=https://www.placeholder.com/architecture.png))
*(Conceptual diagram of the project's architecture)*

| Component | Technology Stack | Description |
| :--- | :--- | :--- |
| **1. Simulation Environment** | `Python`, `NumPy`, `Matplotlib` | An object-oriented N-body physics simulation that generates the training data. It implements a custom "Wave Theory" force law and uses a high-fidelity numerical integrator to ensure physical accuracy.[10, 11, 12] |
| **2. Neuro-Symbolic Engine** | `JAX`, `Equinox`, `PySR`, `SymPy` | The core of the AI. A PINN implemented in JAX tests physical laws, while a PySR engine discovers new symbolic equations from the data. The two components operate in a continuous loop of hypothesis generation and testing.[5, 8] |
| **3. Conversational Interface** | `Streamlit`, `Hugging Face Transformers` | A user-friendly web application that serves as a natural language interface to the trained model. It translates user queries into simulation parameters, runs experiments, and summarizes the results in plain English. |

## Features

  * **High-Fidelity Physics Sandbox**: A customizable N-body simulation to generate complex, wave-based physical phenomena.
  * **Automated Law Discovery**: A neuro-symbolic loop that autonomously refines its understanding of the simulation's physics.
  * **Interactive Conversational AI**: A Streamlit-based chatbot for running virtual experiments through natural language commands.
  * **Advanced AI Integration**: Utilizes state-of-the-art libraries including JAX for high-performance computing, PySR for symbolic regression, and Hugging Face for the LLM interface.
  * **Comprehensive Documentation**: Detailed explanations of the theoretical framework and implementation.

## Getting Started

Follow these instructions to set up the project environment and run the application.

### Prerequisites

  * Python 3.10 or higher
  * `pip` and `venv` for package management
  * Git for cloning the repository

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/utahisnotastate/portfolio.git
    cd wave-theory-chatbot
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the Project

The project is divided into three main executable stages: simulation, training, and the chatbot application.

1.  **Generate Simulation Data:**
    Run the N-body simulation to generate the initial dataset for the AI to analyze.

    ```bash
    python src/simulation/run_simulation.py
    ```

    This will create a `simulation_run.csv` file in the `/data` directory.

2.  **Train the Neuro-Symbolic Model:**
    Execute the main training script. This will initiate the neuro-symbolic loop, where the PINN and SR engine work together to discover the physical laws from the simulation data.

    ```bash
    python src/training/train.py --config configs/training_config.yaml
    ```

    This process is computationally intensive and may take a significant amount of time. Trained models will be saved in the `/models` directory.

3.  **Launch the Wave Theory Chatbot:**
    Once the model is trained, you can launch the interactive Streamlit application.

    ```bash
    streamlit run src/app/main.py
    ```

    This will open a new tab in your web browser with the chatbot interface.

## Usage

Interact with the chatbot using natural language to conduct experiments in the simulated universe. The chatbot can parse commands to modify simulation parameters, run the model, and provide a summary of the results.

**Example Queries:**

  * `"Run the default simulation for 100 time steps and show me the final particle positions."`
  * `"What happens if we double the mass of the central body and run the simulation for 200 steps?"`
  * `"Add a new particle at position (5, 5, 0) with zero velocity and simulate for 50 steps. Describe the resulting interaction."`

## Project Structure

```
wave-theory-chatbot/
├── configs/
│   └── training_config.yaml      # Hyperparameters for training
├── data/
│   └── simulation_run.csv        # Output from the physics simulation
├── docs/
│   └── theoretical_framework.md  # In-depth documentation
├── models/
│   └── final_model.pkl           # Saved trained model checkpoints
├── notebooks/
│   └── exploratory_analysis.ipynb # Jupyter notebooks for development
├── src/
│   ├── simulation/
│   │   ├── core.py               # Core Body and Universe classes
│   │   └── run_simulation.py     # Executable for the simulation
│   ├── model/
│   │   └── pinn.py               # JAX/Equinox implementation of the PINN
│   ├── training/
│   │   └── train.py              # Main neuro-symbolic training loop
│   └── app/
│       └── main.py               # Streamlit chatbot application
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## Contributing

Contributions are welcome\! If you would like to contribute to this project, please follow these steps:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some feature'`).
4.  Push to the branch (`git push origin feature/YourFeature`).
5.  Open a Pull Request.

Please ensure your code adheres to the project's coding standards and includes relevant documentation.

## License

This project is licensed under the MIT License. See the(LICENSE) file for details.

## Acknowledgments

This project was inspired by a wide range of research in neuro-symbolic AI, physics-informed machine learning, and computational physics. We extend our gratitude to the researchers and developers behind the core libraries and concepts that made this work possible.
