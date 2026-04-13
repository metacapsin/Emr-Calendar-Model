from typing import Dict, Optional


class EntityNotFoundError(Exception):
    def __init__(self, entity: str, identifier: str, details: Optional[Dict[str, str]] = None):
        self.entity = entity
        self.identifier = identifier
        self.details = details or {}
        message = f"{entity.capitalize()} not found for '{identifier}'"
        super().__init__(message)

    def to_dict(self) -> Dict[str, str]:
        return {
            "entity": self.entity,
            "identifier": self.identifier,
            "message": str(self),
            **self.details,
        }
