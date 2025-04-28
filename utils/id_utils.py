

from bson import Int64
import xxhash
import torch
import time

class UniqueID:
    def __init__(self, version=0.1):
        """
        Initialize with fixed structure.
        Some fields (e.g., timestamp) are auto-filled at runtime.
        """
        self.fixed_fields = {
            "version": version,
        }

        self.runtime_fields = [
            "timestamp"
        ]

        self.ordered_fields = [
            "version",
            "timestamp",
            "source_id",
            "sensor",
            "extra"  # optional user-defined
        ]

    def _generate_runtime_values(self) -> dict:
        """
        Generate dynamic fields like timestamp
        """
        return {
            "timestamp": int(time.time() * 1000)
        }

    def generate_unique_id(self, data: dict) -> torch.Tensor:
        """
        Generates a consistent, ordered, 64-bit hash-based ID
        from a combination of fixed, runtime, and user fields.
        """
        if not isinstance(data, dict):
            raise TypeError("Input must be a dictionary")

        combined = {
            **self.fixed_fields,
            **self._generate_runtime_values(),
            **data
        }

        hasher = xxhash.xxh64()
        for key in self.ordered_fields:
            hasher.update(str(combined.get(key, "")))  # empty string for missing

        hashed_int = hasher.intdigest()
        return torch.tensor(hashed_int, dtype=torch.long)

class IDUtils:
    _BASE_TRANS = {
        ' ': '_',
        '-': '_',
        '.': '_',
        '/': '_',
        ':': '_',
        '°': '',    # strip degree symbol
        '%': '',    # strip percent sign
    }
    _TRANS_TABLE = str.maketrans(_BASE_TRANS)

    @staticmethod
    def clean_str(s: str, include: list = [], exclude: list = []):
        
        if include or exclude:
            table = IDUtils._BASE_TRANS.copy()
            if include:
                for c in include:
                    table[c] = '_'
            if exclude:
                for c in exclude:
                    del table[c]
            table = str.maketrans(table)
            s = s.translate(table)
            return s.lower()
        else:
            return s.translate(IDUtils._TRANS_TABLE).lower()
    
    # @staticmethod
    # def clean_dict(di, clean_values: bool = True):
    #     if not isinstance(di, dict):
    #         if isinstance(di, list):
    #             return [IDUtils.clean_str(item) for item in di]
    #         else:
    #             return di
    #     elif isinstance(di, dict):
    #         for k, v in di.items():
    #             if clean_values:
    #                 di[IDUtils.clean_str(k)] = IDUtils.clean_str(v)
    #             else:
    #                 di[IDUtils.clean_str(k)] = v
    #         return di
    #             return {IDUtils.clean_str(k): v for k, v in di.items()}
    
    #         return {IDUtils.clean_str(k): IDUtils.clean_str(v) for k, v in di.items()}
    #     else:
    #         return {IDUtils.clean_str(k): v for k, v in di.items()}
    
    @staticmethod
    def get_id(data: dict | list | str | float):
        if not data:
            return None
        if isinstance(data, dict):
            ret = '_'.join([f"{k}_{v}" for k, v in data.items()])
        elif isinstance(data, list):
            ret = '_'.join([f"{item}" for item in data]).replace(' ', '_').lower()
        elif isinstance(data, float):
            ret = str(data)
        else:
            return None
        
        return ret.translate(IDUtils._TRANS_TABLE).lower()
    
    @staticmethod
    def get_system_unique_id(
        data,
        ret_type: str = 'int'
    ):
        """
        Generates a unique ID based on the input dictionary data using xxhash.

        Args:
            data: The input dictionary. Keys are sorted, and values are
                  converted to lowercase strings with spaces replaced by underscores.
            ret_type: The desired return type. Options:
                      'int': Returns a signed 64-bit integer (numpy.int64).
                      'tensor': Returns a PyTorch LongTensor (int64) containing the signed hash.
                      'str': Returns the unsigned 64-bit hash as a decimal string.
                      'hex': Returns the unsigned 64-bit hash as a hex string (e.g., '0x...').
                      'float': Returns the unsigned hash normalized to the range [0.0, 1.0) as a float64.

        Returns:
            The calculated ID in the specified format.

        Raises:
            TypeError: If the input 'data' is not a dictionary.
            ValueError: If 'ret_type' is not one of the allowed values.
        """
        if not isinstance(data, dict):
            raise TypeError("Input 'data' must be a dictionary")

        hasher = xxhash.xxh64()

        # Sort keys for consistent hashing regardless of dict insertion order
        for key in sorted(data.keys()):
            # Ensure value is treated as string, handle None, replace spaces, lowercase
            val = str(data.get(key, "")).replace(" ", "_").lower()
            hasher.update(val.encode('utf-8'))

        # Get the 64-bit unsigned integer hash
        hashed_uint64: int = hasher.intdigest() # Range [0, 2**64 - 1]

        if ret_type == 'int':
            # Convert unsigned 64-bit range [0, 2**64 - 1] to signed [-2**63, 2**63 - 1]
            signed_int64 = hashed_uint64 if hashed_uint64 < (1 << 63) else hashed_uint64 - (1 << 64)
            # Use numpy's int64 for explicit type
            return Int64(signed_int64)

        elif ret_type == 'tensor':
             # Return the signed 64-bit integer version as a tensor
            signed_int64 = hashed_uint64 if hashed_uint64 < (1 << 63) else hashed_uint64 - (1 << 64)
            # Ensure tensor contains the standard signed int64 value
            return torch.tensor(signed_int64, dtype=torch.long)

        elif ret_type == 'str':
            # Return the unsigned hash as a standard decimal string
            return str(hashed_uint64)

        elif ret_type == 'hex':
             # Return the unsigned hash as a hex string
            return hex(hashed_uint64)

        elif ret_type == 'float':
            # Normalize the unsigned hash to the range [0.0, 1.0)
            # Use 2.0**64 to ensure float division and get float64 precision
            normalized_float = float(hashed_uint64) / (2.0**64)
            return normalized_float

        else:
            raise ValueError("Invalid ret_type. Choose from 'int', 'tensor', 'str', 'hex', 'float'.")

    @staticmethod
    def convert_system_unique_id(
        system_unique_id,
        input_type: str,
        output_type: str
        ):
        """
        Converts a previously generated system unique ID between different formats.

        Args:
            system_unique_id: The ID generated by get_system_unique_id.
            input_type: The format of the provided 'system_unique_id'.
                        Options: 'int', 'tensor', 'str' (decimal uint64), 'hex' (hex uint64), 'float' (normalized).
            output_type: The desired output format.
                         Options: 'int', 'tensor', 'str', 'hex', 'float'.

        Returns:
            The ID converted to the 'output_type' format.

        Raises:
            ValueError: If input/output types are invalid or conversion is not supported/logical.
            TypeError: If the input ID does not match the expected 'input_type'.
        """
        valid_types = {'int', 'tensor', 'str', 'hex', 'float'}
        if input_type not in valid_types or output_type not in valid_types:
            raise ValueError(f"input_type and output_type must be one of {valid_types}")

        # --- Step 1: Parse the input ID into a canonical form (unsigned uint64) ---
        # Note: We lose the signed/unsigned distinction inherent in 'int' and 'tensor' types here,
        # always parsing back to the original unsigned hash bits for internal consistency.
        # The float type is also converted back approximately.
        try:
            parsed_uint64: int
            if input_type == 'int': # Assumes numpy.int64 representing signed int64
                if not isinstance(system_unique_id, (np.int64, int)):
                     raise TypeError(f"Expected int or numpy.int64 for input_type 'int', got {type(system_unique_id)}")
                # Convert potentially signed int64 back to uint64 bit pattern
                parsed_uint64 = int(np.uint64(system_unique_id))
            elif input_type == 'tensor': # Assumes torch.long representing signed int64
                if not isinstance(system_unique_id, torch.Tensor) or system_unique_id.numel() != 1:
                     raise TypeError(f"Expected single-element torch.Tensor for input_type 'tensor', got {type(system_unique_id)}")
                # Convert potentially signed int64 tensor back to uint64 bit pattern
                parsed_uint64 = int(np.uint64(system_unique_id.item()))
            elif input_type == 'str': # Assumes decimal string representation of uint64
                 if not isinstance(system_unique_id, str):
                      raise TypeError(f"Expected str for input_type 'str', got {type(system_unique_id)}")
                 parsed_uint64 = int(system_unique_id)
            elif input_type == 'hex': # Assumes hex string representation of uint64 ('0x...')
                 if not isinstance(system_unique_id, str):
                      raise TypeError(f"Expected str for input_type 'hex', got {type(system_unique_id)}")
                 parsed_uint64 = int(system_unique_id, 16)
            elif input_type == 'float': # Assumes normalized float [0.0, 1.0)
                 if not isinstance(system_unique_id, float):
                      raise TypeError(f"Expected float for input_type 'float', got {type(system_unique_id)}")
                 if not (0.0 <= system_unique_id < 1.0):
                      # Allow exactly 1.0 due to potential precision limits if original was 2**64-1
                      if system_unique_id != 1.0:
                           raise ValueError("Input float must be in the range [0.0, 1.0)")
                 # Convert back, acknowledging potential float precision loss
                 parsed_uint64 = int(round(system_unique_id * (2.0**64)))
                 # Clamp to uint64 max value in case of rounding issues near 1.0
                 max_uint64 = (1 << 64) - 1
                 if parsed_uint64 > max_uint64 : parsed_uint64 = max_uint64

            else: # Should be caught by initial check, but for safety
                 raise ValueError(f"Unhandled input_type: {input_type}")

            if not (0 <= parsed_uint64 < (1 << 64)):
                 raise ValueError("Parsed ID is outside the valid uint64 range after conversion.")

        except (ValueError, TypeError) as e:
            raise type(e)(f"Error parsing system_unique_id '{system_unique_id}' with input_type '{input_type}': {e}")


        # --- Step 2: Convert the parsed uint64 to the desired output format ---
        if output_type == 'int':
            signed_int64 = parsed_uint64 if parsed_uint64 < (1 << 63) else parsed_uint64 - (1 << 64)
            return Int64(signed_int64)
        elif output_type == 'tensor':
            signed_int64 = parsed_uint64 if parsed_uint64 < (1 << 63) else parsed_uint64 - (1 << 64)
            return torch.tensor(signed_int64, dtype=torch.long)
        elif output_type == 'str':
            return str(parsed_uint64)
        elif output_type == 'hex':
            return hex(parsed_uint64)
        elif output_type == 'float':
            return float(parsed_uint64) / (2.0**64)
        else: # Should be caught by initial check
            raise ValueError(f"Unhandled output_type: {output_type}")

    @staticmethod
    def get_source_id(self, source_id,  data_subtype: str = 'sensor'):
        di = {}
        di['data_type'] = 'source'
        di['data_subtype'] = data_subtype
        di['source_id'] = source_id
        return IDUtils.get_system_unique_id(di)
    
    @staticmethod
    def get_sensor_type_id(sensor_type: str, data_subtype: str = 'sensor_value', ):
        di = {}
        di['data_type'] = 'sensor_type'
        di['data_subtype'] = data_subtype
        di['sensor_type'] = sensor_type
        return IDUtils.get_system_unique_id(di)
    
    @staticmethod
    def get_provider_type_id(uid: str, data_subtype: str = 'provider'):
        di = {}
        di['data_type'] = 'provider_type'
        di['data_subtype'] = data_subtype
        di['uid'] = uid
        return IDUtils.get_system_unique_id(di)
    
    @staticmethod
    def get_sensor_id(source_id: str, sensor_type_id: str, data_subtype: str = 'source_sensor'):
        di = {}
        di['data_type'] = 'sensor_id'
        di['data_subtype'] = data_subtype
        di['source_id'] = source_id
        di['sensor_type'] = sensor_type_id

        return IDUtils.get_system_unique_id(di)