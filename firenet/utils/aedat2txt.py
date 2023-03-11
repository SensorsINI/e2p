"""
 @Time    : 2/14/22 01:59
 @Author  : Haiyang Mei
 @E-mail  : mhy666@mail.dlut.edu.cn
 
 @Project : firenet-pdavis
 @File    : aedat2txt.py
 @Function:
 
"""
'''
extract events information from .aedat file and saved as .txt file
'''

import numpy as np
import os
import struct


def getDVSeventsDavis(file, ROI=np.array([]), numEvents=1e10, startEvent=0):
    print('\ngetDVSeventsDavis function called \n')
    sizeX = 346
    sizeY = 260
    x0 = 0
    y0 = 0
    x1 = sizeX
    y1 = sizeY
    if len(ROI) != 0:
        if len(ROI) == 4:
            print('Region of interest specified')
            x0 = ROI(0)
            y0 = ROI(1)
            x1 = ROI(2)
            y1 = ROI(3)
        else:
            print('Unknown ROI argument. Call function as: \n getDVSeventsDavis(file, ROI=[x0, y0, x1, y1], numEvents=nE, startEvent=sE) '
                  'to specify ROI or\n getDVSeventsDavis(file, numEvents=nE, startEvent=sE) to not specify ROI')
            return

    else:
        print('No region of interest specified, reading in entire spatial area of sensor')

    print('Reading in at most', str(numEvents))
    print('Starting reading from event', str(startEvent))

    triggerevent = int('400', 16)
    polmask = int('800', 16)
    xmask = int('003FF000', 16)
    ymask = int('7FC00000', 16)
    typemask = int('80000000', 16)
    typedvs = int('00', 16)
    xshift = 12
    yshift = 22
    polshift = 11
    x = []
    y = []
    ts = []             # Timestamps tick is 1 us
    pol = []
    numeventsread = 0

    length = 0
    aerdatafh = open(file, 'rb')
    k = 0
    p = 0
    statinfo = os.stat(file)
    if length == 0:
        length = statinfo.st_size
    print("file size", length)

    lt = aerdatafh.readline()
    while lt and str(lt)[2] == "#":
        p += len(lt)
        k += 1
        lt = aerdatafh.readline()
        continue

    aerdatafh.seek(p)
    tmp = aerdatafh.read(8)
    p += 8
    while p < length:
        ad, tm = struct.unpack_from('>II', tmp)
        ad = abs(ad)
        if (ad & typemask) == typedvs:
            xo = sizeX - 1 - float((ad & xmask) >> xshift)
            yo = float((ad & ymask) >> yshift)
            polo = 1 - float((ad & polmask) >> polshift)
            if xo >= x0 and xo < x1 and yo >= y0 and yo < y1:
                x.append(xo)
                y.append(yo)
                pol.append(polo)
                ts.append(tm)
        aerdatafh.seek(p)
        tmp = aerdatafh.read(8)
        p += 8
        numeventsread += 1

    # ts[:] = [x - ts[0] for x in ts]  # absolute time -> relative time
    x[:] = [int(a) for a in x]
    y[:] = [int(a) for a in y]

    print('Total number of events read =', numeventsread)
    print('Total number of DVS events returned =', len(ts))

    return ts, x, y, pol


if __name__ == "__main__":
    rpathraw = '/home/mhy/firenet-pdavis/data/v2e/aedat/' # folder containing .aedat files
    wpathtxt = '/home/mhy/firenet-pdavis/data/v2e/txt/' # folder to save converted files

    for root, dirs, fs in os.walk(rpathraw):
        for f in fs:
            ts, x, y, pol = getDVSeventsDavis(rpathraw + f)
            ts = np.array(ts, dtype=np.uint64).reshape(-1, 1)
            x = np.array(x, dtype=np.uint16).reshape(-1, 1)
            y = np.array(y, dtype=np.uint16).reshape(-1, 1)
            pol = np.array(pol, dtype=np.uint8).reshape(-1, 1)
            prefix = [346, 260, 0, 0]
            data = np.hstack((ts, x, y, pol))
            data = np.vstack((prefix, data))
            string = f.split(".")
            f = string[0] + ".txt"

            np.savetxt(wpathtxt + f, data, fmt='%d', delimiter='\t')
