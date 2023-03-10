import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pdb
import cv2
import aedat
from tqdm import tqdm
import os
from scipy.optimize import curve_fit

def func1(x, a, b, c, d):
    return a*np.square(np.cos(b*x + c)) + d

def func2(x, a, b, c, d):
    return -a*np.square(np.cos(b*x + c)) + d

def func3(x, a, b, c, d):
    return -a*np.square(np.cos(b*x - c)) + d

def func4(x, a, b, c, d):
    return a*np.square(np.cos(b*x - c)) + d

def getDarkFrames(path):
    decoder = aedat.Decoder(path)
    frames = np.zeros((IMAGER[0],IMAGER[1]))
    cnt = 0
    for packet in decoder:
        if 'frame' in packet:
            frames += packet['frame']['pixels']
            cnt += 1
    return (frames / cnt)

def getV1PixelData(frames):
    sPixels = np.zeros(((180//step)+1, IMAGER[0]//2, IMAGER[1]//2, 4))
    cnt = 0
    for frame in tqdm(frames):
        splitData = np.zeros((IMAGER[0]//2, IMAGER[1]//2, 4))
        for i in range(0, 4, 1):
            splitData[:, :, i] = frame[PIXLOC[i, 0]::2, PIXLOC[i, 1]::2]

        sPixels[cnt] = splitData
        cnt += 1
    return sPixels

def getV4PixelData(v4Data, dFrames):
    sPixels = np.zeros(((180//step)+1, IMAGER[0]//2, IMAGER[1]//2, 4))
    for dataFile in tqdm(v4Data):
        angle = int(dataFile.split('_')[4])//step
        splitData = np.zeros((IMAGER[0]//2, IMAGER[1]//2, 4))
        frames = np.zeros((IMAGER[0],IMAGER[1]))
        cnt = 0
        decoder = aedat.Decoder(os.path.join(v4Path, dataFile))
        for packet in decoder:
            if 'frame' in packet:
                frames += packet['frame']['pixels']
                cnt += 1
        frames /= cnt
        frames -= dFrames

        for i in range(0, 4, 1):
            splitData[:, :, i] = frames[PIXLOC[i, 0]::2, PIXLOC[i, 1]::2]

        sPixels[angle] = splitData
    return sPixels

def getPixelSection(left, right, top, bottom, sPixels):

    sPixels = sPixels[:, left:right, bottom:top,:]

    sPixelsMean = np.mean(np.reshape(np.swapaxes(sPixels,1,3), ((180//step)+1,4,-1)), axis=2)
    sPixelsStd = np.std(np.reshape(np.swapaxes(sPixels,1,3), ((180//step)+1,4,-1)), axis=2)

    return sPixelsMean

malus = False

IMAGER = [260, 346]
PIXLOC = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
lbls = ['I0', 'I45', 'I90', 'I135']
v4Path = '/Users/justinhaque/Google Drive/POLDVS/Apr_20_2021/MonoChrom/v4/'
v1Path = '/Users/justinhaque/Google Drive/POLDVS/Apr_10_2021/v1/ConvertedData/'
v1File = 'MalusLaw.npz'
dFramesFile = 'DarkFrames_APS.aedat4'
dsStore = '.DS_Store'
lbls = ['I0', 'I45', 'I90', 'I135']

thres = 182
version = 'v4'
step = 5
startWLen = 500
wLenStep = 20

subAngle = [[115,150], [25,55], [70,105]]
I0Angle = list(range(0,16,step)) + list(range(160, 176,step))

anglesD = np.linspace(0, 180, (180//step)+1)
anglesR = anglesD*np.pi/180
reAngleD = np.linspace(0, 180, 10000)
reAngleR = reAngleD*np.pi/180

left, right, bottom, top = 50, 92, 67, 95
lBounds = [-1e10,0.95,-100, -1e10]
uBounds = [1e10, 1.05, 100, 1e10]

# v1Data = np.load(v1Path+v1File)
allV4Data = os.listdir(v4Path)
if dFramesFile in allV4Data: allV4Data.remove(dFramesFile)
if dsStore in allV4Data: allV4Data.remove(dsStore)
allV4Data.sort()

wLens = [allV4Data[i].split('_')[2] for i in range(len(allV4Data))]
wLens.sort()
wLens = np.unique(wLens)

if version == 'v4':
    dFrames = getDarkFrames(os.path.join(v4Path, dFramesFile))
elif version == 'v1':
    dFrames = v1Data['darkFrame']

dFramesROI = dFrames[left:right,bottom:top]
roiDraw = patches.Rectangle((2*bottom, 2*left), 2*(top-bottom), 2*(right-left), linewidth=2, edgecolor='r', facecolor='none')
fig, ax = plt.subplots()
plt.imshow(dFrames, 'jet')
plt.title('Dark Frame HeatMap')
ax.add_patch(roiDraw)
plt.colorbar()
plt.figure()
plt.hist(np.reshape(dFramesROI, (dFramesROI.shape[0]*dFramesROI.shape[1],-1)), bins=50)
# plt.hist(np.reshape(dFrames, (dFrames.shape[0]*dFrames.shape[1],-1)), bins=50)
plt.xlabel('Dark Frame DN')
plt.ylabel('# of Pixels')
# plt.xlim(right=90)
# plt.xlim(left=25)
plt.title('Dark Frame Histogram')
plt.show()

WVER = np.zeros((len(wLens), 4))
# pdb.set_trace()
for wLen in wLens:
    v4Data = [allV4Data[i] for i in range(len(allV4Data)) if wLen in allV4Data[i]]
    print(wLen)

    if version == 'v4':
        sPixels = getV4PixelData(v4Data, dFrames)
    elif version == 'v1':
        sPixels = getV1PixelData(v1Data['framesAvg']-dFrames)

    sPixelsMean = getPixelSection(left, right, top, bottom, sPixels)

    if version == 'v1':
        I0Mean = np.array([np.array(I0Angle)*np.pi/180, sPixelsMean[:,2][np.array(I0Angle)//step]]).T
        I45Mean = np.array([np.arange(subAngle[0][0],subAngle[0][1]+1,step)*np.pi/180, sPixelsMean[:,0][subAngle[0][0]//step:(subAngle[0][1]//step)+1]]).T
        I135Mean = np.array([np.arange(subAngle[1][0],subAngle[1][1]+1,step)*np.pi/180, sPixelsMean[:,3][subAngle[1][0]//step:(subAngle[1][1]//step)+1]]).T
        I90Mean = np.array([np.arange(subAngle[2][0],subAngle[2][1]+1,step)*np.pi/180, sPixelsMean[:,1][subAngle[2][0]//step:(subAngle[2][1]//step)+1]]).T

    elif version == 'v4':
        I0Mean = np.array([[i*np.pi/36,sPixelsMean[i][0]] for i in range(36) if sPixelsMean[i][0] < thres])
        I45Mean = np.array([[i*np.pi/36,sPixelsMean[i][2]] for i in range(36) if sPixelsMean[i][2] < thres])
        I90Mean = np.array([[i*np.pi/36,sPixelsMean[i][3]] for i in range(36) if sPixelsMean[i][3] < thres])
        I135Mean = np.array([[i*np.pi/36,sPixelsMean[i][1]] for i in range(36) if sPixelsMean[i][1] < thres])

    popt0, pcov = curve_fit(func1, I0Mean[:,0], I0Mean[:,1], bounds=(lBounds, uBounds))
    rg0 = func1(reAngleR, *popt0)

    popt45, pcov = curve_fit(func1, I45Mean[:,0], I45Mean[:,1], bounds=(lBounds, uBounds))
    rg45 = func1(reAngleR, *popt45)

    popt90, pcov = curve_fit(func1, I90Mean[:,0], I90Mean[:,1], bounds=(lBounds, uBounds))
    rg90 = func1(reAngleR, *popt90)

    popt135, pcov = curve_fit(func1, I135Mean[:,0], I135Mean[:,1], bounds=(lBounds, uBounds))
    rg135 = func1(reAngleR, *popt135)

    if malus == True:
        plt.plot(I0Mean[:,0]*180/np.pi, I0Mean[:,1], 'x--', color='red', label=lbls[0])
        plt.plot(I45Mean[:,0]*180/np.pi, I45Mean[:,1], 'x--', color='blue', label=lbls[1])
        plt.plot(I90Mean[:,0]*180/np.pi, I90Mean[:,1], 'x--', color='orange', label=lbls[2])
        plt.plot(I135Mean[:,0]*180/np.pi, I135Mean[:,1], 'x--', color='green', label=lbls[3])

        plt.title("Intensity @ " + wLen + "nm w/ thresholding aedat" + version[1])
        plt.xlabel("Incident AoP")
        plt.ylabel('Digital Number')
        plt.legend(loc=4)
        plt.figure()

        plt.plot(reAngleD, rg0, color='red', label='Fit eRatio0: ' + str(format(np.max(rg0)/np.min(rg0), '.2f')))
        plt.plot(reAngleD, rg45, color='blue', label='Fit eRatio45: ' + str(format(np.max(rg45)/np.min(rg45), '.2f')))
        plt.plot(reAngleD, rg90, color='orange', label='Fit eRatio90: ' + str(format(np.max(rg90)/np.min(rg90), '.2f')))
        plt.plot(reAngleD, rg135, color='green', label='Fit eRatio135: ' + str(format(np.max(rg135)/np.min(rg135), '.2f')))

        plt.title("Intensity @ " + wLen + "nm using regression aedat" + version[1])
        plt.xlabel("Incident AoP")
        plt.ylabel('Interpolated Data')
        plt.legend(loc=4)
        plt.show()

    WVER[(int(wLen)-startWLen)//wLenStep][0] = np.max(rg0)/np.min(rg0)
    WVER[(int(wLen)-startWLen)//wLenStep][1] = np.max(rg45)/np.min(rg45)
    WVER[(int(wLen)-startWLen)//wLenStep][2] = np.max(rg90)/np.min(rg90)
    WVER[(int(wLen)-startWLen)//wLenStep][3] = np.max(rg135)/np.min(rg135)

x = [int(wLens[i]) for i in range(len(wLens))]
for i in range(4):
    plt.plot(x, WVER[:,i], 'x--', label=lbls[i])
plt.title('Extintion ratios over wavelengths aedat' + version[1])
plt.xlabel("Wavelength (nm)")
plt.ylabel('Extinction Ratio')
plt.legend(loc=1)
plt.show()
