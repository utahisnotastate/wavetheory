"""
Testing and Validation Module
Comprehensive testing, validation, and quality assurance
"""

from .validation_suite import (
    TestResult,
    ValidationReport,
    PhysicsValidationSuite,
    ModelValidationSuite,
    IntegrationTestSuite,
    physics_validator,
    model_validator,
    integration_tester
)

__all__ = [
    'TestResult',
    'ValidationReport',
    'PhysicsValidationSuite',
    'ModelValidationSuite',
    'IntegrationTestSuite',
    'physics_validator',
    'model_validator',
    'integration_tester'
]
