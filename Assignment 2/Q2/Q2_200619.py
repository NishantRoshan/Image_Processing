import cv2
import numpy as np

def gauss(spatialKern, rangeKern, hWin):
    matrix = np.exp(-np.arange(256) * np.arange(256) / (2*(rangeKern**2)))
    xx=-hWin + np.arange(2 * hWin + 1)
    yy=-hWin + np.arange(2 * hWin + 1)
    x, y = np.meshgrid(xx , yy )
    spatialGS = np.exp(-(x **2 + y **2) /(2 * (spatialKern**2) ) ) #calculate spatial kernel from the gaussian function. That is the gaussianSpatial variable multiplied with e to the power of (-x^2 + y^2 / 2*sigma^2)
    return matrix,spatialGS

def padImg(img,hWin): #pad array with mirror reflections of itself.
    img=np.pad(img, ((hWin, hWin), (hWin, hWin), (0, 0)))
    return img

dMap = {}
def simi(i, j, k, flash):
    th = 10
    if i > 0 and not (i-1,j,k) in dMap:
        dMap[(i,j,k)]=(i-1,j,k)
        return abs(flash[i,j,k]-flash[i-1,j,k]) < th
    if j > 0 and not (i,j-1,k) in dMap:
        dMap[(i,j,k)]=(i,j-1,k)
        return abs(flash[i,j,k]-flash[i,j-1,k]) < th
    return False

def onePixVal(i, j, k, nflash, flash, hWin, gaussArr, gaussMat):
    i += hWin
    j += hWin

    neighbourhood=flash[i-hWin : i+hWin+1 , j-hWin : j+hWin+1, k]
    central=flash[i, j, k]
    range_gauss = gaussArr[ abs(neighbourhood - central) ]
    space_gauss = gaussMat
    res = range_gauss * space_gauss
    norm = np.sum(res)
    res = res/norm
    res = res * nflash[i-hWin : i+hWin+1 , j-hWin : j+hWin+1, k]
    return np.sum(res)

def crossBiLatFil(nflash, flash, hWin, sp_sigma, r_sigma):
    h, w, ch = flash.shape
    # hWin = int(win_size/2)

    flash = flash.astype(int)
    nflash = nflash.astype(int)

    flash_pad = padImg(flash, hWin)
    # flash_pad = cv2.cvtColor(flash, cv2.COLOR_BGR2GRAY)
    nflash_pad = padImg(nflash, hWin)

    gaussArr, gaussMat = gauss(sp_sigma, r_sigma, hWin)

    res = [onePixVal(i, j, k, nflash_pad, flash_pad, hWin, gaussArr, gaussMat) if not simi(i,j,k,flash) else -1 for i, j, k in np.ndindex(flash.shape)]
    # res = lesGo(nflash_pad, flash_pad, hWin, gaussArr, gaussMat)
    res = np.reshape(res, flash.shape)

    for i in range(h):
        for j in range(w):
            for k in range(ch):
                if res[i,j,k] == -1:
                    res[i,j,k] = res[dMap[(i,j,k)]]
                    # if i > 0:
                    #     res[i,j,k]=res[d]
                    # elif j > 0:
                    #     res[i,j,k]=res[i,j-1,k]


    return res

def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image

    spatialKern = 30
    rangeKern = 5
    no_flash = cv2.imread(image_path_a)
    flash = cv2.imread(image_path_b)


    jointBi = crossBiLatFil(no_flash, flash,10, spatialKern, rangeKern)

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    # image = cv2.imread(image_path_b)
    return jointBi

# x = solution('D:\\DATA\\Coding\\github\\EE604\\Assignment-2\\Q2\\ultimate_test\\2_a.jpg','D:\\DATA\\Coding\\github\\EE604\\Assignment-2\\Q2\\ultimate_test\\2_b.jpg')
# cv2.imwrite('n.png', x)
