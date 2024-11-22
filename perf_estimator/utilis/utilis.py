import os
import tempfile
import datetime
from typing import List, Optional

def datetime_converter(
    time: datetime.datetime, iso8601: bool = True, format: str = "%Y%m%d-%H%M%S"
) -> str:
    """COnvert `datetime.datetime` object to iso8601 format or custom format.

    Args:
        time (datetime.datetime): the time want to be converted
        iso8601 (bool): if True, return iso8601 format. Defaults to True.
        format (str): custom format, and it is only available when iso8601 set to false. Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: the time in expected format
    """
    if iso8601:
        return time.isoformat() + "Z"
    else:
        return time.strftime(format)


def time_now(iso8601: bool = True, format: str = "%Y%m%d-%H%M%S") -> str:
    """Generate a current time in iso8601 format or custom format.

    Args:
        iso8601 (bool): if True, return iso8601 format. Defaults to True.
        format (str): custom format, and it is only available when iso8601 set to false. Defaults to "%Y%m%d-%H%M%S".

    Returns:
        str: the time in expected format

    """
    return datetime_converter(
        datetime.datetime.utcnow(), iso8601=iso8601, format=format
    )


def fetch_file_paths(directory) -> List[str]:
    """Fetch all file paths in the specified directory.

    Args:
        directory (str): the directory path

    Returns:
        List[str]: a list of file paths

    """
    file_paths = []
    # Walk through the directory and its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            # Join the root path with the file name to get the absolute path
            file_path = os.path.join(root, file)
            file_paths.append(str(file_path))

    return file_paths


def filter_files(part_file_name: str, directory: str, fuzz: bool = True) -> List[str]:
    """Fetch all file path that contains the specified part of file name.

    Args:
        part_file_name (str): the part of file name or keyword
        directory (str): the directory path
        fuzz (bool): if True, return all files that contain the part of file name. Defaults to True.

    Returns:
        List[str]: a list of file paths
    """
    file_paths = fetch_file_paths(directory)
    if fuzz:
        return [
            str(file_path)
            for file_path in file_paths
            if part_file_name in os.path.basename(str(file_path))
        ]
    else:
        return [
            str(file_path)
            for file_path in file_paths
            if part_file_name == os.path.basename(str(file_path))
        ]


def list_directories(path: str) -> Optional[List[str]]:
    """list all directories in the specified path.

    Args:
        path (str): the path to a directories

    Returns:
        Optional[List[str]]: a list of directories or None if the path does not exist

    """
    # Ensure the path exists
    if not os.path.exists(path):
        print("The specified path does not exist.")
        return None

    # List all entries in the path
    entries = os.listdir(path)
    directories = [
        entry for entry in entries if os.path.isdir(os.path.join(path, entry))
    ]

    return directories


def fetch_temp_folder() -> str:
    """Fetch a temporary folder.

    Returns:
        str: a temporary folder path

    """
    return tempfile.TemporaryDirectory().name


def format_memory(nbytes: int) -> str:
    """Format memory size in human-readable format.

    Args:
        nbytes (int): the memory size in bytes

    Returns:
        str: the memory size in human-readable format

    **The function is copied from PyTorch source code.**
    """
    KB = 1024
    MB = 1024 * KB
    GB = 1024 * MB
    if abs(nbytes) >= GB:
        return f"{nbytes * 1.0 / GB:.2f} Gb"
    elif abs(nbytes) >= MB:
        return f"{nbytes * 1.0 / MB:.2f} Mb"
    elif abs(nbytes) >= KB:
        return f"{nbytes * 1.0 / KB:.2f} Kb"
    else:
        return str(nbytes) + " b"


def get_temp_folder():
    """Create a temporary dir

    Returns:
        path of the temp dir
    """
    return tempfile.TemporaryDirectory().name


def temp_dir_with_specific_path(*args):
    """Create a temporary directory with a specific path.

    Args:
        *args: Variable length argument list to specify the subdirectory path.

    Returns:
        str: The path of the created temporary directory.
    """
    temp_dir = tempfile.gettempdir()
    flame_temp = os.path.join(temp_dir, *args)
    if not os.path.isdir(flame_temp):
        os.makedirs(flame_temp, exist_ok=True)
    return flame_temp
