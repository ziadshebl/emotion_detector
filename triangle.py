import math
class Triangle():
    def __init__(self, p1, p2, p3):

        D1 = math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) 
        self.D1 = D1

        D2 = math.sqrt((p1[0]-p3[0])**2 + (p1[1]-p3[1])**2) 
        self.D2 = D2

        D3 = math.sqrt((p2[0]-p3[0])**2 + (p2[1]-p3[1])**2) 
        self.D3 = D3

        S = (D1 + D2 + D3)/3
        self.S = S
        
        AoT = math.sqrt(S*abs(S-D1)*abs(S-D2)*abs(S-D3))   
        self.AoT = AoT

        perimeter =  D1 + D2 + D3
        r = 2* (AoT/perimeter)  
        ICC = 2*math.pi*r
        self.ICC = ICC

        self.ICAT = math.pi*(r**2)

    def print_features(self):
        print("AoT " + str(self.AoT) + " ICC " + str(self.ICC) + " ICAT " + str(self.ICAT))