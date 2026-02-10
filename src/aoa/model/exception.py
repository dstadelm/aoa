class AoAException(Exception):
    """Base class exception for AOA."""


class NonUniqueIdException(AoAException):
    """Exception throwed when ID's are detected that aren't unique"""


class AllocationException(AoAException):
    """Exception for allocation sequence errors"""
