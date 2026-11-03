"""All settings for the project, kept in one place so experiments are easy to change."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The nine playable races in the D&D 5e Player's Handbook.
# A race's position in this tuple is the integer ID the model sees, so changing
# the order or contents means a saved model has to be retrained.
RACES = (
    "dragonborn",
    "dwarf",
    "elf",
    "gnome",
    "half-elf",
    "half-orc",
    "halfling",
    "human",
    "tiefling",
)


@dataclass(frozen=True)
class Config:
    # Files
    names_path: Path = PROJECT_ROOT / "data" / "raw" / "names.csv"
    model_path: Path = PROJECT_ROOT / "checkpoints" / "best_model.keras"
    tokenizer_path: Path = PROJECT_ROOT / "checkpoints" / "tokenizer.json"

    # Model size. Kept small on purpose: with only a few thousand names, a larger
    # model would memorise the training names instead of learning their patterns.
    char_embedding_dim: int = 32
    race_embedding_dim: int = 16
    gru_units: int = 128
    dropout: float = 0.2

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    validation_fraction: float = 0.1
    seed: int = 42

    # Generation
    max_generated_length: int = 30
    default_temperature: float = 0.8
