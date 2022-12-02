import enum

class Instance():

    def __init__(self):
        self.file_name = ""
        self.type = Type.TRAINING
        self.objects = []

    def get_image_path(self):
        if self.type == Type.TRAINING:
            return f'data/images/training/{self.file_name}'
        elif self.type == Type.VALIDATION:
            return f'data/images/validation/{self.file_name}'

class Object():

    def __init__(self):
        self.class_name = ""
        self.box = Box()

class Box():

    def __init__(self):
        self.x1 = 0
        self.y1 = 0
        self.x2 = 100
        self.y2 = 100

class Type(enum.Enum):
    TRAINING = 1
    TESTING = 2
    VALIDATION = 3