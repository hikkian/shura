def parse_duration(text):
    """Parse a duration like "1h30m", "45s", "2h 5s", "1d2h" into total seconds (int).

    Units: d (days), h (hours), m (minutes), s (seconds). Each unit may appear at most once,
    in order d, h, m, s. Spaces between parts are allowed. Case-insensitive.
    A bare integer like "90" means seconds.
    Raise ValueError for empty strings, unknown units, repeated or out-of-order units,
    and negative numbers.
    """
    raise NotImplementedError
