class Rect:
    def __init__(self,text):
        text=text.split()
        self.x1=int(text[0])
        self.x2=int(text[1])
        self.y1=int(text[2])
        self.y2=int(text[3])
        self.weight=eval(text[4])
class Rectangle:
    def __init__(self,x=0,y=0,width=0,height=0):
        self.x=x
        self.y=y
        self.width=width
        self.height=height