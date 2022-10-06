import nbody
import random
import cv2
from time import sleep
import itertools

settings = nbody.SimulationSettings(1/60, 5e-4, 1e-2)
cam = nbody.CameraSettings(nbody.Vec2D(0, 0), 300)
codec = nbody.CodecSettings("libx264", "slow", 0)
encoder = nbody.VideoEncoder("/home/joris/test.264", 1600, 900, 30, codec)
pset = nbody.ParticleWrapperCPU(1000, 1)

for i in range(1000):
    pset.set_particle(i, nbody.Vec2D((random.random() - .5) * 2, (random.random() - .5) * 2), nbody.Vec2D(0.0, 0.0), 1.0)

dir = .1

try:
    while True:
        for zoom in itertools.chain(range(100, 700, 10), range(700, 100, -10)):
            cam.zoom = zoom
            pixels = encoder.generate_pixels(pset, cam)
            cv2.imshow('img', pixels)
            cv2.waitKey(1000//30)
except KeyboardInterrupt:
    cv2.destroyAllWindows()
