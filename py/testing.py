import nbody
import random

settings = nbody.SimulationSettings(1/60, 5e-4, 1e-2)
cam = nbody.CameraSettings(nbody.Vec2D(0, 0), 60*5)
codec = nbody.CodecSettings("libx264", "slow", 0)
encoder = nbody.VideoEncoder("/home/joris/test.264", 1920, 1080, 30, codec)
pset = nbody.ParticleWrapperCPU(1000, 1)

for i in range(1000):
    pset.set_particle(i, nbody.Vec2D((random.random() - .5) * 2, (random.random() - .5) * 2), nbody.Vec2D(0.0, 0.0), 1.0)

for i in range(60*5):
    pset.do_timestep(settings)
    encoder.write_wrapper(pset, cam)