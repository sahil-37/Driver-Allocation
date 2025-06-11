from datetime import datetime
from dateutil import parser


def iso_to_datetime(iso_str: str, fallback: bool = True) -> datetime:
    """
    Converts ISO-like string to datetime object. Handles both formats with and without microseconds.
    
    Args:
        iso_str: A string representing datetime in ISO format.
        fallback: If True, falls back to common formats on failure.

    Returns:
        datetime object parsed from the string.
    """
    try:
        return parser.isoparse(iso_str)  # most robust
    except Exception:
        if fallback:
            for fmt in ("%Y-%m-%d %H:%M:%S.%f %Z", "%Y-%m-%d %H:%M:%S %Z"):
                try:
                    return datetime.strptime(iso_str, fmt)
                except ValueError:
                    continue
        raise ValueError(f"Unrecognized datetime format: {iso_str}")


def hour_of_iso_date(iso_str: str) -> int:
    """
    Extracts the hour (0–23) from a given ISO-formatted string.

    Args:
        iso_str: datetime string in ISO or common datetime format.

    Returns:
        Hour of the datetime as an integer.
    """
    return iso_to_datetime(iso_str).hour


def robust_hour_of_iso_date(iso_str: str) -> int:
    """
    Robust version to extract the hour from ISO-like string.
    Handles both microsecond and non-microsecond cases, with UTC or timezone strings.

    Args:
        iso_str: datetime string.

    Returns:
        Integer hour (0–23), or -1 if parsing fails.
    """
    try:
        return hour_of_iso_date(iso_str)
    except Exception:
        return -1  # fallback, or raise/log depending on use case
