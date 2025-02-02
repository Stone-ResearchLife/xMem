def llvm_is_power_of_2(x: int) -> bool:
    """Check if a 64-bit integer is a power of 2.

    Args:
        x (int): 64-bit integer

    Returns:
        bool: True if x is a power of 2, False otherwise

    """
    return x > 0 and (x & (x - 1)) == 0


def llvm_power_of_2_floor(x: int) -> int:
    """Return the largest power of 2 less than or equal to x.

    Args:
        x (int): 64-bit integer

    Examples:
        >>> llvm_power_of_2_floor(8)
        8
        >>> llvm_power_of_2_floor(9)
        8
        >>> llvm_power_of_2_floor(0)
        0

    Returns:

    """
    return 1 << (x.bit_length() - 1)


def llvm_count_leading_zeros(n: int) -> int:
    """Count the number of leading zeros in a 64-bit integer.

    Args:
        n (int): 64-bit integer

    Examples:
        >>> llvm_count_leading_zeros(8)
        60
        >>> llvm_count_leading_zeros(0)

    Returns:

    """
    if n == 0:
        return 64  # 64-bit zero
    return 64 - n.bit_length()
