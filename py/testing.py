import nbody
import random
import cv2
import numpy as np
from time import sleep, time
import itertools
import cProfile

def main():
    random.seed('lol troll')

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
    last_delta = 1
    last_values = []
    values_to_keep = 10
    limit_fps = False

    try:
        total_frames = fps_target*4

        for i in range(total_frames):
            start = time()
            with pset.exec_while_calculating(settings):
                pset.write_to_array(pixels, cam)

                if len(last_values) < values_to_keep:
                    eta = '-'
                else:
                    frames_left = total_frames - i
                    time_left = round(frames_left * sum(last_values) / values_to_keep)
                    eta = f'{time_left//60}:{str(time_left%60).zfill(2)}'

                encoder.encode_image(pixels)
                cv2.putText(pixels, f'{1/last_delta:.2f} fps ETA {eta}', (0, 40), font, .6, (255, 255, 255), 1)
                cv2.imshow('img', pixels)
                cv2.waitKey(1)
                if limit_fps and diff > 0:
                    diff = delta_target - (time() - start) - 1e-3
                    sleep(diff)

            last_delta = (time() - start)
            last_values.append(last_delta)
            if len(last_values) > values_to_keep:
                last_values = last_values[1:]

    except KeyboardInterrupt:
        cv2.destroyAllWindows()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    # cProfile.run('main()')
