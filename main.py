# General
import copy
import os
import itertools
# from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
# from IPython.display import HTML
import PIL.Image

from dm_control import suite

#@title Loading and simulating a `suite` task{vertical-output: true}

# Load the environment
random_state = np.random.RandomState(42)
env = suite.load(domain_name='walker', task_name='walk', task_kwargs={'random': random_state})

# Simulate episode with random actions
duration = 4  # Seconds
frames = []
ticks = []
rewards = []
observations = []

spec = env.action_spec()
time_step = env.reset()

while env.physics.data.time < duration:

  action = random_state.uniform(spec.minimum, spec.maximum, spec.shape)
  time_step = env.step(action)
  print(time_step)

#   camera0 = env.physics.render(camera_id=0, height=200, width=200)
#   camera1 = env.physics.render(camera_id=1, height=200, width=200)
#   frames.append(np.hstack((camera0, camera1)))
#   rewards.append(time_step.reward)
#   observations.append(copy.deepcopy(time_step.observation))
#   ticks.append(env.physics.data.time)

# html_video = display_video(frames, framerate=1./env.control_timestep())

# # Show video and plot reward and observations
# num_sensors = len(time_step.observation)

# _, ax = plt.subplots(1 + num_sensors, 1, sharex=True, figsize=(4, 8))
# ax[0].plot(ticks, rewards)
# ax[0].set_ylabel('reward')
# ax[-1].set_xlabel('time')

# for i, key in enumerate(time_step.observation):
#   data = np.asarray([observations[j][key] for j in range(len(observations))])
#   ax[i+1].plot(ticks, data, label=key)
#   ax[i+1].set_ylabel(key)

