import numpy as np
class LBPFeatureExtractor:
    @staticmethod
    def get_pixel(img, center, x, y):   
        new_value = 0
        try:
            if img[x][y] >= center:
                new_value = 1       
        except:
            pass
        
        return new_value
    
    @staticmethod
    def lbp_calculated_pixel(img, x, y):
    
        center = img[x][y]
        val_ar = []
        
        # top_left
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x-1, y-1))
        
        # top
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x-1, y))
        
        # top_right
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x-1, y + 1))
        
        # right
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x, y + 1))
        
        # bottom_right
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x + 1, y + 1))
        
        # bottom
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x + 1, y))
        
        # bottom_left
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x + 1, y-1))
        
        # left
        val_ar.append(LBPFeatureExtractor.get_pixel(img, center, x, y-1))
        
        # Now, we need to convert binary
        # values to decimal
        power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    
        val = 0
        
        for i in range(len(val_ar)):
            val += val_ar[i] * power_val[i]
            
        return val
        
    def calculate_lbp_features(frame): 
        height, width = frame.shape
        frame_lbp = np.zeros((height, width),np.uint8)
   
        for i in range(0, height):
            for j in range(0, width):
                frame_lbp[i, j] = LBPFeatureExtractor.lbp_calculated_pixel(frame, i, j)
        frame_lbp = frame_lbp.flatten()        
        hist, bin_edges = np.histogram(frame_lbp, bins=np.arange(256), range=None, normed=False, weights=None, density=None)
        return hist

