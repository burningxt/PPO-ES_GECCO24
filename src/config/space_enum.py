from enum import Enum

class space_operation(Enum):
    NO_SPACE = 0
    JUST_SIZES = 1
    INSTANCE_STATE = 2
    ONE_GENERATION = 3

    
class instance_ordering(Enum):
    ABSOLUTE = 0
    IMPROVEMENT = 1
    RELATIVE_IMPROVEMENT = 2
    NONE = 3



