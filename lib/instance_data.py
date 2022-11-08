class Instance():

    def __init__(self):
        self.file_name = ""
        self.original = None
        self.preprocessed = [[]]
        self.objects = []

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
            print(f'x1: {object.box.x1}, y1: {object.box.y1}\nx2: {object.box.x2}, y2: {object.box.y2}\n')

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