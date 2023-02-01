import numpy as np
import cv2


def get_gs(window_size, sigma_s, pad_w):
    '''return gs, size: (window_size, window_size)'''
    gs = np.zeros((window_size, window_size, 3))
    for i in range(window_size):
        for j in range(i, window_size):
            result =  np.exp((-1) * ((i-pad_w)**2 + (j-pad_w)**2) * 0.5  / (sigma_s**2))
            gs[i, j, :] = result 
            gs[j, i, :] = result
    return gs


class Joint_bilateral_filter(object):
    def __init__(self, sigma_s, sigma_r):
        self.sigma_r = sigma_r
        self.sigma_s = sigma_s
        self.wndw_size = 6*sigma_s+1
        self.pad_w = 3*sigma_s
    
    def joint_bilateral_filter(self, img, guidance):
        BORDER_TYPE = cv2.BORDER_REFLECT
        padded_img = cv2.copyMakeBorder(img, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)
        padded_guidance = cv2.copyMakeBorder(guidance, self.pad_w, self.pad_w, self.pad_w, self.pad_w, BORDER_TYPE).astype(np.int32)

        gs = get_gs(self.wndw_size, self.sigma_s, self.pad_w)       
        gr_dict = np.array([np.exp(-((x**2) * 0.5 / ((self.sigma_r**2)*(255**2)))) for x in range(256)])
        output = np.zeros(img.shape)
        
        # for an image
        for i in range(img.shape[0]): 
           for j in range(img.shape[1]):
                intensity = padded_img[i:i+self.wndw_size, j:j+self.wndw_size]

                Tq = padded_guidance[i:i+self.wndw_size, j:j+self.wndw_size]
                Tp = padded_guidance[i+self.pad_w, j+self.pad_w]
                abs_value = abs(Tq-Tp)                
                if len(padded_guidance.shape) == 3:
                    gr = gr_dict[abs_value[:,:,0]] * gr_dict[abs_value[:,:,1]] * gr_dict[abs_value[:,:,2]] # rgb channel
                else:
                    gr = gr_dict[abs_value]
                    
                gr = np.stack((gr, gr, gr), axis=-1)
                
                
                output[i, j] = np.sum(np.sum(gr*gs*intensity, axis=0), axis=0) / np.sum(np.sum((gr * gs), axis=0), axis=0)
        return np.clip(output, 0, 255).astype(np.uint8)
