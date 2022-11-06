class Instance():

    def __init__(self):
        self.file_name = ""
        self.original = None
        self.preprocessed = [[]]
        self.objects = []

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