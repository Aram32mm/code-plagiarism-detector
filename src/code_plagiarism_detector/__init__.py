"""Code plagiarism detection using perceptual hashing and AST fingerprinting."""

from .code_hasher import (
    CodeHasher,
    ComparisonResult,
    LanguageConfig,
    DetectorConfig,
    DEFAULT_LANGUAGES
)
from .database import CodeHashDatabase, load_code_bank

__version__ = "0.1.0"
__all__ = [
    "CodeHasher",
    "ComparisonResult",
    "LanguageConfig",
    "DetectorConfig",
    "DEFAULT_LANGUAGES",
    "CodeHashDatabase",
    "load_code_bank"
]