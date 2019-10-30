import pytest

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from gym_minigrid.minigrid import OBJECT_TO_IDX, IDX_TO_OBJECT, Floor, Wall, Door
from gym_minigrid.envs.house import Lattice, House, HouseEnv, Mapper, Roomba, Agent0

# TODO
# - test_gen_obs places agent at door sometimes
# - test_get_door fails sometimes when it's told to go to the bottom door but only doors are either left or right - so maybe it's a problem in Mapper's find_path instead?

## Helpers ##
room_sizes = range(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT+1)

def composed(*decs):
    def deco(f):
        for dec in reversed(decs):
            f = dec(f)
        return f
    return deco

dim_sweep = lambda _min, _max: pytest.mark.parametrize(
        ('start', 'end', 'dim'),
        [
            ((0,0), (0,1), (m, n))\
                    for m in range(_min, _max+1)\
                    for n in range(_min, _max)\
            ]
        )

height_sweep = lambda min_h, max_h: pytest.mark.parametrize(
        ('y','h'),
        [
            (y,h) for h in range(min_h, max_h+1) for y in range(1, h+1)
            ]
        )

width_sweep = lambda min_w, max_w: pytest.mark.parametrize(
        ('x','w'),
        [
            (x,w) for w in range(min_w, max_w+1) for x in range(1, w+1)
            ]
        )

size_sweep = lambda low, high: pytest.mark.parametrize(
        ('x', 'y', 's'),
        [
            (x, y, s) for s in range(low, high+1) for x in range(1, s+1) for y in range(1, s+1)
            ]
        )

dir_sweep = pytest.mark.parametrize('agent_dir', range(4))

### HouseEnv works ##
@composed(
        size_sweep(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT),
        dir_sweep,
        )
def test_house_env_init(x, y, s, agent_dir):
    h = HouseEnv(pos_in_room=(x,y), room_w=s, room_h=s, agent_dir=agent_dir)
    print(h)

def test_get_room():
    h = HouseEnv()
    assert h.get_room() == h.lattice.start

@composed(
        size_sweep(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT),
        dir_sweep,
        )
def test_view_size(x, y, s, agent_dir, start, dim):
    house = HouseEnv(
            dim = dim, # Lattice params
            edges = None, # random
            start = start,
            end = None,
            room_w = s, # House params
            room_h = s,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = (x, y), # Agent position w/in start room
            agent_dir = agent_dir, # Agent direction in 0 to 3
            )

    assert house.agent_view_size == s+2, 'Agent view size {} but (w,h)={}'.format(house.agent_view_size, (s, s))

@composed(
        dim_sweep(Lattice.MIN_DIM, Lattice.MAX_DIM),
        size_sweep(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT),
        )
def test_no_empty_floor(x, y, s, start, end, dim):
    house = HouseEnv(
            dim = dim, # Lattice params
            edges = None, # random
            start = start,
            end = end,
            room_w = s, # House params
            room_h = s,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = (x, y), # Agent position w/in start room
            agent_dir = 0, # Agent direction in 0 to 3
            )

    # No empty tiles in the grid
    assert all([tile is not None for tile in house.grid.grid])
    assert 1 not in house.grid.encode()[:,:,0]

    # But there's one in the agent view - the agent itself
    grid, _ = house.gen_obs_grid()
    assert sum([tile is None for tile in grid.grid]) == 1
    assert OBJECT_TO_IDX['empty'] in house.gen_obs()['image'][:,:,0]

@composed(
        size_sweep(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT),
        dir_sweep,
        )
def test_gen_obs(agent_dir, s, x, y, dim=(2,2), start=(0,0)):

    house = HouseEnv(
            dim = dim, # Lattice params
            edges = None, # random
            start = start,
            end = None,
            room_w = s, # House params
            room_h = s,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = (x, y), # Agent position w/in start room
            agent_dir = agent_dir, # Agent direction in 0 to 3
            )

    _, mask = house.gen_obs_grid()

    assert mask.shape == (s+2, s+2), 'Mask is {} but should be {}'.format(mask.shape[:-1], (s+2, s+2))

    obs = house.gen_obs()
    img = obs['image']

    objects = img[:,:,0]
    _ = img[:,:,1]
    _ = img[:,:,2]

    objGrid = house.grid.decode(img)

    try:
        assert any([isinstance(tile, Door) for tile in objGrid.grid]), 'No doors!'
    except:
        # TODO
        # I've isolated the problem to this
        # The agent seems to be standing at a door
        # but when I plot the overall view of the house, that's not the case
        # Also, place_agent has a check for that:
        #   assert isinstance(self.grid.get(x,y), Floor)
        # So IDK what's going on...
        house.grid.plot(agent=(*house.agent_pos, 10+house.agent_dir))
        plt.savefig('test_figures/ad_{}_s_{}_x_{}_y_{}.png'.format(agent_dir, s, x, y))
        plt.close()

        assert np.isin(objects[0,:], [OBJECT_TO_IDX['wall'], 10, 11, 12, 13]).all()
        assert np.isin(objects[-1,:], [OBJECT_TO_IDX['wall'], 10, 11, 12, 13]).all()
        assert np.isin(objects[:,0], [OBJECT_TO_IDX['wall'], 10, 11, 12, 13]).all()
        assert np.isin(objects[:,-1], [OBJECT_TO_IDX['wall'], 10, 11, 12, 13]).all()

# Mapper works #
@composed(
    dim_sweep(2, 4),
)
def test_mapper(x, y, dim, start, end):
    
    house = HouseEnv(
            dim = dim, # Lattice params
            edges = None, # random
            start = start,
            end = end,
            room_w = House.MIN_ROOM_HEIGHT, # House params
            room_h = House.MIN_ROOM_HEIGHT,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = (x, y), # Agent position w/in start room
            agent_dir = 0, # Agent direction in 0 to 3
            )

    m = Mapper(house)
    path = m.find_path()
    room = start

    # Test path is as needed
    for next_room in path:
        room = room[0] + next_room[0], room[1] + next_room[1]

    assert room == end, 'Final room[={}] is not end room[={}]!'.format(room, end)

    # Test goal in end room
    tx = end[0] * (house.room_w+1)
    ty = end[1] * (house.room_h+1)

    end_grid = house.slice(tx, ty, house.room_w+1, house.room_h+1)

    assert any([isinstance(tile, Goal) for tile in end_grid.grid]), 'No goal in end room!'

# Roomba works #
@composed(
        size_sweep(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT),
        dir_sweep,
        )
def test_get_door(agent_dir, s, x, y, dim=(2,2), start=(0,0)):

    house = HouseEnv(
            dim = dim, # Lattice params
            edges = None, # random
            start = start,
            end = None,
            room_w = s, # House params
            room_h = s,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = (x, y), # Agent position w/in start room
            agent_dir = agent_dir, # Agent direction in 0 to 3
            )

    obs = house.gen_obs()['image']

    # drop cases with no door
    if OBJECT_TO_IDX['door'] not in obs[:,:,0]:
        return

    m = Mapper(env=house)
    big_path = m.find_path()
    next_room = big_path[0]

    r = Roomba(env=house)

    # Clock-wise rotation in usual coordinates but ccw in our coordinates where up is negative Y
    dir_vec = tuple(house.dir_vec)
    test_vecs = [
            [1, 0],
            [0, 1],
            [-1, 0],
            [0, -1],
            ]
    msg = 'Test vector {} wrong with dir vector {}'

    if dir_vec == (0,-1):
        for test in test_vecs:
            assert np.allclose(r._rotate_vec(test), test), msg.format(test, dir_vec)
    elif dir_vec == (-1,0):
        for test in test_vecs:
            assert np.allclose(r._rotate_vec(test), np.array([-test[1], test[0]])), msg.format(test, dir_vec)
    elif dir_vec == (0,1):
        for test in test_vecs:
            assert np.allclose(r._rotate_vec(test), np.array([-test[0], -test[1]]))
    elif dir_vec == (1,0):
        for test in test_vecs:
            assert np.allclose(r._rotate_vec(test), np.array([test[1], -test[0]]))

    # TODO this fails sometimes because door is empty - IDK why
    # It always happens when next_room is (0,1) = bottom
    # but the doors are either only left or right
    try:
        door = r._get_door(obs, next_room)
    except:
        raise

#@composed(
#        size_sweep(House.MIN_ROOM_HEIGHT, House.MAX_ROOM_HEIGHT),
#        dir_sweep,
#        )
def test_find_path_roomba(agent_dir=0, s=5, x=2, y=1, dim=(2,2), start=(0,0)):

    house = HouseEnv(
            dim = dim, # Lattice params
            edges = None, # random
            start = start,
            end = None,
            room_w = s, # House params
            room_h = s,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = (x, y), # Agent position w/in start room
            agent_dir = agent_dir, # Agent direction in 0 to 3
            )

    # Initiate mapper
    m = Mapper(env=house)

    # Find path across rooms
    big_path = m.find_path()

    # Initiate roomba
    r = Roomba(env=house)

    # Generate initial obs
    obs = house.gen_obs()['image']

    for next_room in big_path:
        # Find path within room
        actions = r.find_path(obs, next_room)

        for action in actions:
            obs, _, _, _ = house.step(action)
            obs = obs['image']
