from collections.abc import Generator


def id_generator(start: int = -1, increment: int = 1) -> Generator[int, None, None]:
    current_id = start
    while True:
        current_id += increment
        yield current_id
