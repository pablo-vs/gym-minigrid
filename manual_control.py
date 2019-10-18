#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym
import time
from optparse import OptionParser

import gym_minigrid

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym environment to load",
        default='MiniGrid-MultiRoom-N6-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)

    print('\n')
    print('Lattice dimensions: {}'.format(env.lattice.dim))
    print('Agent placed in room: {}'.format(env.lattice.start))
    print('\n')

    def resetEnv():
        env.reset()
        if hasattr(env, 'mission'):
            print('Mission: %s' % env.mission)

    resetEnv()

    # Create a window to render into
    renderer = env.render('human')

    def keyDownCb(keyName):
        if keyName == 'BACKSPACE':
            resetEnv()
            return

        if keyName == 'ESCAPE':
            sys.exit(0)

        action = 0

        if keyName == 'LEFT':
            action = env.actions.left
        elif keyName == 'RIGHT':
            action = env.actions.right
        elif keyName == 'UP':
            action = env.actions.forward

        elif keyName == 'SPACE':
            action = env.actions.toggle
        elif keyName == 'PAGE_UP':
            action = env.actions.pickup
        elif keyName == 'PAGE_DOWN':
            action = env.actions.drop

        elif keyName == 'RETURN':
            action = env.actions.done

        else:
            print("unknown key %s" % keyName)
            return

        obs, reward, done, info = env.step(action)

        print('\n')
        print(obs['image'][:,:,0].T)

        #print('step=%s, reward=%.2f' % (env.step_count, reward))

        ax, ay = env.agent_pos
        ri, rj = env.get_room()
        tx, ty, bx, by = env.get_view_exts()

#        print('\n')
#        print('Agent position: {}'.format(env.agent_pos))
#        print('Agent direction: {}'.format(env.dir_vec))
#        print('Agent view size: {}'.format(env.agent_view_size))
#        print('Agent room: {}'.format((ri,rj)))
#        print('View extents:\n')
#        print('  Top left X: {}'.format(tx))
#        print('  Top left Y: {}'.format(ty))
#        print('  Bottom right X: {}'.format(bx))
#        print('  Bottom right Y: {}'.format(by))

        if done:
            print('done!')
            resetEnv()

    renderer.window.setKeyDownCb(keyDownCb)

    while True:
        env.render('human')
        time.sleep(0.01)

        # If the window was closed
        if renderer.window == None:
            break

if __name__ == "__main__":
    main()
