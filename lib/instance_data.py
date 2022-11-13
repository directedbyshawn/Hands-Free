import enum

class Instance():

    def __init__(self):
        self.file_name = ""
        self.type = Type.TRAINING
        self.objects = []

    def get_path(self):
        if self.type == Type.TRAINING:
            return f'/data/images/training/{self.file_name}'
        elif self.type == Type.VALIDATION:
            return f'/data/images/validation/{self.file_name}'        

    def repr(self):
        for _ in range(50):
            print('-', end='')
        print()

        print(f'File name: {self.file_name}')
        self.original.show()
        print(self.preprocessed)
        print("Objects in image: ")
        for object in self.objects:
            print(f'\n{object.class_name} with box:')
            print(f'x1: {object.box.x1}, y1: {object.box.y1}')
            print(f'x2: {object.box.x2}, y2: {object.box.y2}')
        print()

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