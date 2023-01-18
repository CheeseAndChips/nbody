import nbody
import random
import cv2
import numpy as np
from time import sleep, time
import itertools
import cProfile

def main():
    fps_target = 30
    width = 1600
    height = 900
    zoom = 300
    font = cv2.FONT_HERSHEY_SIMPLEX
    pixels = np.zeros((height, width), dtype=np.uint8)

    settings = nbody.SimulationSettings(1/fps_target, 2e-5, 1e-2)
    cam = nbody.CameraSettings(nbody.Vec2D(0, 0), zoom)
    codec = nbody.CodecSettings("libx264", "slow", 0)
    encoder = nbody.VideoEncoder("/home/joris/test.264", width, height, 30, codec)
    # pset = nbody.ParticleWrapperCPU(5000, 4)
    pset = nbody.ParticleWrapperGPU(100_000)
    # pset = nbody.ParticleWrapperGPU(1000)


    for i in range(pset.get_n()):
        pset.set_particle(i, nbody.Vec2D((random.random() - .5) * 2, (random.random() - .5) * 2), nbody.Vec2D(0.0, 0.0), 1.0)
        
    delta_target = 1 / fps_target
    last_fps = 0.0

    last_values = []
    values_to_keep = 10
    limit_fps = False

    try:
        for i in range(fps_target*10):
            start = time()
            with pset.exec_while_calculating(settings):
                # encoder.encode_wrapper(pset, cam)
                pset.write_to_array(pixels, cam)
                encoder.encode_image(pixels)
                # pixels.fill(0)
                cv2.putText(pixels, f'{last_fps:.2f} fps', (0, 50), font, 1, (255, 255, 255), 2)
                cv2.imshow('img', pixels)
                cv2.waitKey(1)
                if limit_fps and diff > 0:
                    diff = delta_target - (time() - start) - 1e-3
                    sleep(diff)

            last_fps = 1 / (time() - start)
    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # cProfile.run('main()')
