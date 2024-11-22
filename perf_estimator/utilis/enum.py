from enum import Enum, EnumMeta
from typing import List, Optional


class EnumManipulator:
    """A class to manipulate Enum object."""

    def __init__(self, input_enum: EnumMeta):
        """initialize the class.

        Args:
            input_enum (EnumMeta): an Enum object
        """
        self._enum = input_enum

    @property
    def fetch_enums(self) -> EnumMeta:
        """Get the Enum object."""
        return self._enum

    def fetch_keys(self) -> List[str]:
        """Get all keys in the Enum object."""
        return self.fetch_enums._member_names_

    def fetch_enum(self, key_name: str) -> Optional[Enum]:
        """Fetch the Enum object by key name.

        Args:
            key_name (str): the key name

        Returns:
            Optional[Enum]: the Enum object or None if the key does not exist

        """
        key = None
        for _key in self.fetch_keys():
            if key_name.lower() == str(_key).lower():
                key = self.fetch_enums[_key]
                break
        return key

    def check_key(self, key_name: str) -> bool:
        """Check if the key exists in the Enum object.

        Args:
            key_name (str): the key name

        Returns:
            bool: True if the key exists, False otherwise

        """
        return False if self.fetch_enum(key_name) is None else True

    def fetch_value(self, key_name: str):
        """Fetch the value of the Enum object by key name.

        Args:
            key_name (str): the key name

        Returns:
            any: the value of the Enum object or None if the key does not exist

        """
        key = self.fetch_enum(key_name)
        value = None
        if key is not None:
            value = key.value
        return value

    def filter_by(self, keyword: str, field: str = None) -> list:
        """Fetch the key when the keyword is found in the field of the Enum object.
        If the field is None, the keyword is compared with the value of the Enum object.

        Args:
            keyword (str): the keyword to search
            field (str): the field to search. Defaults to None. if not none, the field points to the attribute of the value.

        Returns:
            list: a list of keys

        """
        keys = self.fetch_keys()
        result = []
        for key in keys:
            _value = self.fetch_enums[key].value
            if field is None:
                if keyword == str(_value):
                    result.append(key)
            else:
                _field_value = getattr(_value, field)
                if isinstance(_field_value, str):
                    if keyword.lower() in _field_value.lower():
                        result.append(key)
                elif isinstance(_field_value, list):
                    if keyword.lower() in [str(_).lower() for _ in _field_value]:
                        result.append(key)
        return result
