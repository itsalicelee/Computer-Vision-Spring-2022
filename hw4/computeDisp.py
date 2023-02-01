import numpy as np
import cv2.ximgproc as xip

def shift_image(X, dx, dy):
    X = np.roll(X, dy, axis=0)
    X = np.roll(X, dx, axis=1)
    if dy>0:
        X[:dy, :] = 0
    elif dy<0:
        X[dy:, :] = 0
    if dx>0:
        X[:, :dx] = 0
    elif dx<0:
        X[:, dx:] = 0
    return X

def computeDisp(Il, Ir, max_disp):
    h, w, ch = Il.shape
    labels = np.zeros((h, w), dtype=np.float32)
    Il = Il.astype(np.float32)
    Ir = Ir.astype(np.float32)

    # >>> Cost Computation
    # TODO: Compute matching cost
    # [Tips] Census cost = Local binary pattern -> Hamming distance
    # [Tips] Set costs of out-of-bound pixels = cost of closest valid pixel  
    # [Tips] Compute cost both "Il to Ir" and "Ir to Il" for later left-right consistency
    

    # >>> Cost Aggregation
    # TODO: Refine the cost according to nearby costs
    # [Tips] Joint bilateral filter (for the cost of each disparty)
    right_cost = np.zeros((max_disp+1, h, w), dtype=np.float32)
    left_cost = np.zeros((max_disp+1, h, w), dtype=np.float32)
    
    # right_cost = np.sum(np.square(Il-Ir))
    # left_cost = np.sum(np.square(Ir-Il))
    
    for s in range(max_disp+1):
        Ir_shifted = shift_image(Ir, s, 0)
        Il_shifted = shift_image(Il, -s, 0)
        
        right_temp = np.sum(np.square(Il-Ir_shifted), axis=2).astype(np.float32)
        left_temp = np.sum(np.square(Ir-Il_shifted), axis=2).astype(np.float32)
        # print(right_temp)
        # print(left_temp)
        right_cost[s,:,:] = xip.jointBilateralFilter(Il, right_temp, 30, 6, 6)
        left_cost[s,:,:] = xip.jointBilateralFilter(Ir, left_temp, 30 , 6, 6)  
    # print(left_cost)
    # print(left_cost.shape)
    
    # >>> Disparity Optimization
    # TODO: Determine disparity based on estimated cost.
    # [Tips] Winner-take-all
    left_winner = np.argmin(right_cost, axis=0)
    right_winner = np.argmin(left_cost, axis=0)

    # >>> Disparity Refinement
    # TODO: Do whatever to enhance the disparity map
    # [Tips] Left-right consistency check -> Hole filling -> Weighted median filtering
    for x in range(w):
        for y in range(h):
            if (x < left_winner[y,x]) or (left_winner[y,x] != right_winner[y,x-left_winner[y,x]]):
                left_winner[y,x] = -1
                
    for x in range(w):
        for y in range(h):
            if left_winner[y,x] == -1:
                l, r = 0, 0
                while x >= l and left_winner[y, x-l] == -1 :
                    l += 1
                while x+r < w and left_winner[y, x+r] == -1:
                    r += 1
                left_final = max_disp if(x-l<0) else left_winner[y, x-l]
                right_final = max_disp if (x+r > w-1) else left_winner[y, x+r]
                    
                left_winner[y,x] = np.minimum(left_final, right_final)
   
    labels = xip.weightedMedianFilter(Il.astype(np.uint8), left_winner.astype(np.uint8), 15, 1)
    return labels.astype(np.uint8)