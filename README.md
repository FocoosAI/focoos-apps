# Focoos Apps

## Overview

**Focoos Apps** is a collection of vertical applications built on top of the [Focoos AI SDK](https://github.com/FocoosAI/focoos). This repository serves as a showcase and implementation hub for real-world AI applications that leverage Focoos's computer vision capabilities.

The repository is designed to demonstrate practical implementations of AI-powered solutions for various industries and use cases, starting with smart parking and expanding to include additional applications in the future.

## Purpose

This repository exists to:

- **Demonstrate Real-World Applications**: Show how the Focoos AI SDK can be used to build practical, production-ready AI applications
- **Provide Reference Implementations**: Offer well-structured, documented code that developers can use as templates for their own projects
- **Accelerate Development**: Reduce the time-to-market for AI applications by providing pre-built solutions for common use cases
- **Showcase Best Practices**: Demonstrate proper integration patterns, error handling, and performance optimization techniques

## Applications

| Application | Description | Status | Usage Guide |
|-------------|-------------|--------|-------------|
| **Smart Parking** | AI-powered parking space occupancy detection and monitoring system | ✅ Available | [📖 Smart Parking Guide](focoos_apps/apps/smart_parking/README.md) |
| *Future Applications* | Additional AI applications will be added here | 🚧 Coming Soon | - |

## Getting Started

### Prerequisites

- Python 3.12 or higher
- Focoos AI SDK access and API key
- OpenCV and other dependencies (see `pyproject.toml`)

### Installation

```bash
# Install Focoos AI SDK
uv pip install --system 'focoos-apps @ git+https://github.com/FocoosAI/focoos-apps.git'
```

or

```bash
# Clone the repository
git clone https://github.com/FocoosAI/focoos-apps.git
cd focoos-apps

# Install dependencies
pip install -e .
```

### Using Pretrained Models

Each application in this repository can be used with our pretrained sample models. The download instructions and usage information for these models are explained in detail in each application's README file.

### Training Custom Models

For custom use cases and specific requirements, you can train your own models using the Focoos AI platform. Here's the process:

1. **Sign up to the platform**: Create an account on the [Focoos AI Platform](https://platform.focoos.ai)
2. **Load a custom dataset**: Prepare and upload your dataset with proper annotations
3. **Choose a pretrained model**: Select a base model that matches your task requirements
4. **Launch the training**: Start the fine-tuning process on your specific dataset
5. **Monitor the training**: Track progress and performance metrics in real-time
6. **Export and deploy**: Once training is complete, export your model for various runtime types
7. **Deploy custom models**: Each application's README contains instructions on how to download and deploy your custom trained models

### Quick Start

Each application has its own detailed usage guide. Start with the [Smart Parking application](focoos_apps/apps/smart_parking/README.md) to see how to:

1. Configure the application
2. Process videos or images
3. Create custom parking zones
4. Generate annotated outputs

## Architecture

The repository follows a modular architecture:

```
focoos_apps/
├── core/           # Shared components and base classes
├── apps/           # Individual applications
│   └── smart_parking/
│       ├── smart_parking.py
│       └── README.md
└── cli/            # Command-line interface
```

## License

This project is licensed under the terms specified in the [LICENSE](LICENSE) file.

## Support

For support and questions:

- 📧 Email: info@focoos.ai
- 🐛 Issues: [GitHub Issues](https://github.com/FocoosAI/focoos-apps/issues)
- 📚 Documentation: [Focoos AI Documentation](https://focoosai.github.io/focoos/)

*Built with ❤️ by the Focoos AI team*
