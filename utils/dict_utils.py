from datetime import datetime

import numpy as np

class DictUtils:
    DATETIME_FORMATS = ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]

    @classmethod
    def parse_value(cls, di: dict, path: str, default=None, split='.'):
        """
        Safely parse a value from a nested dictionary using a customizable-separated path.

        Args:
            di (dict): The dictionary to parse.
            path (str): Path string, e.g., "key1.key2.key3".
            default: Default value to return if the path does not exist.
            split (str): Delimiter used in the path string (default is '.').

        Returns:
            The value from the dictionary at the specified path or the default value.
        """
        if di is None:
            return default
        keys = path.split(split)
        current = di

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    @classmethod
    def parse_datetime_direct(cls, dt, formats=None):
        """
        Safely parse a datetime value from a datetime object or a string.
        """
        if dt is None:
            return None
        if isinstance(dt, datetime):
            return dt
        if isinstance(dt, (int, float)):
            return datetime.fromtimestamp(dt)
        if not formats:
            formats = DictUtils.DATETIME_FORMATS
        else:
            formats = list(set(formats + DictUtils.DATETIME_FORMATS))
        for fmt in formats:
            try:
                return datetime.strptime(dt, fmt)
            except (ValueError, TypeError):
                continue
        return None

    @classmethod
    def parse_datetime(cls, di: dict, path: str, default=None, formats=None, split='.'):
        """
        Safely parse a datetime value from a nested dictionary using a customizable-separated path.

        Args:
            di (dict): The dictionary to parse.
            path (str): Path string, e.g., "key1.key2.key3".
            default: Default value to return if the path does not exist or parsing fails.
            formats (list): List of datetime string formats to attempt for parsing.
            split (str): Delimiter used in the path string (default is '.').

        Returns:
            A datetime object parsed from the value, or the default value.
        """
        if di is None:
            return default
       

        value = cls.parse_value(di, path, default=None, split=split)
        if value is None:
            return default

        return DictUtils.parse_datetime_direct(value, formats=formats)

    @classmethod
    def put_value(cls, di: dict, path: str, value, split='.'):
        """
        Safely put a value into a nested dictionary, creating intermediate paths if necessary.

        Args:
            di (dict): The dictionary to modify.
            path (str): Path string where the value will be placed, e.g., "key1.key2.key3".
            value: The value to place at the specified path.
            split (str): Delimiter used in the path string (default is '.').
        """
        if di is None:
            return
        keys = path.split(split)
        current = di

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    @classmethod
    def put_datetime(cls, di: dict, path: str, value, default=None, formats=None, split='.'):
        """
        Safely put a datetime value into a nested dictionary, creating intermediate paths if necessary.

        Args:
            di (dict): The dictionary to modify.
            path (str): Path string where the datetime value will be placed, e.g., "key1.key2.key3".
            value: The datetime value to place at the specified path (datetime object, timestamp, or string).
            split (str): Delimiter used in the path string (default is '.').
        """
        if di is None or value is None:
            return default
        if isinstance(value, (int, float)):
            value = datetime.fromtimestamp(value)
        elif isinstance(value, str):
            value = DictUtils.parse_datetime(value, formats=formats)
            if value is None:
                return default
        elif not isinstance(value, datetime):
            return default

        cls.put_value(di, path, value, split=split)


    @classmethod
    def np_to_list(cls, di: dict):
        """
        Convert all numpy arrays in a dictionary to lists.

        Args:
            di (dict): The dictionary to convert.

        Returns:
            A dictionary with all numpy arrays converted to lists.
        """
        for k, v in di.items():
            if isinstance(v, np.ndarray):
                di[k] = v.tolist()
            elif isinstance(v, dict):
                di[k] = DictUtils.np_to_list(v)
        return di
