"""Code plagiarism detection using perceptual hashing and AST fingerprinting."""

from .code_hasher import CodeHasher, ComparisonResult
from .database import CodeHashDatabase, load_code_bank

__version__ = "0.1.0"
__all__ = ["CodeHasher", "ComparisonResult", "CodeHashDatabase", "load_code_bank"]
