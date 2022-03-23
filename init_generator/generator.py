from calendar import c
from re import L
from PIL import Image
import sys
import random
import struct

def binary_search(lst, value):
    l = 0
    r = len(lst) - 1

    while l <= r:
        mid = (l + r) // 2

        if lst[mid][1] < value:
            l = mid + 1
        else:
            r = mid - 1

    if l >= len(lst):
        l = len(lst) - 1
    return lst[l][0]

def deviate(deviation, coord):
    if deviation == 0:
        return coord
    else:
        while True:
            added = tuple(random.random() for _ in range(len(coord)))
            if sum(i*i for i in added) <= 1:
                return tuple(a + b * deviation for a, b in zip(coord, added))


def main():
    if len(sys.argv) != 5:
        print(f'Usage: {sys.argv[0]} <image path> <particle count> <deviation> <outfile>')
        return

    imgpath = sys.argv[1]
    pcount = int(sys.argv[2])
    deviation = float(sys.argv[3])
    outfile = sys.argv[4]

    s = 0
    choices = []

    print('Reading image...')
    with Image.open(imgpath) as im:
        maxcoord = max(im.width, im.height)
        for x in range(im.width):
            for y in range(im.height):
                coords = (x, y)
                p = im.getpixel(coords)
                if type(p) == int:
                    px = p
                else:
                    px = max(im.getpixel(coords))
                if px > 0:
                    partcoord = tuple((a - b / 2) / maxcoord for a, b in zip(coords, (im.width, im.height)))
                    choices.append((partcoord, s))
                    s += px

    print('Generating particles...')
    particles = [binary_search(choices, random.randint(0, s)) for _ in range(pcount)]
    particles = [deviate(deviation, p) for p in particles]

    print('Writing to file...')
    with open(outfile, 'wb') as f:    
        f.write(struct.pack('<i', pcount))
        for x, y in particles:
            f.write(struct.pack('<f', x))
            f.write(struct.pack('<f', y))
        for _ in particles: # velocity
            f.write(struct.pack('<f', 0))
            f.write(struct.pack('<f', 0))
        for _ in particles: # mass
            f.write(struct.pack('<f', 1))
        
if __name__ == '__main__':
    main()