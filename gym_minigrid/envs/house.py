"""
House Environment and related utilities.
"""

from gym_minigrid.minigrid import Grid, MiniGridEnv, Door, Goal
from gym_minigrid.minigrid import OBJECT_TO_IDX, IDX_TO_OBJECT
from gym_minigrid.register import register

from enum import IntEnum

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

CELL_PIXELS = 32

class HouseEnv(MiniGridEnv):
    """
    House environment for IDA
    """

#    class Actions(IntEnum):
#            # Turn left, turn right, move forward
#            left = 0
#            right = 1
#            up = 2
#            down = 3
#            done = 6

    def __init__(
            self,
            room_w = 5,
            room_h = 5,
            size = 31,
    ):
        self.room_w = room_w
        self.room_h = room_h

        super().__init__(
                grid_size=size,
                max_steps=5*5**2,
                # Set this to True for maximum speed
                see_through_walls=False,
                agent_view_size=self.room_w+2
        )

        self.actions = self.Actions

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        # TODO copied it bc maybe we want to override it?

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _gen_grid(self, width=None, height=None):
        # Gen lattice
        self.lattice = Lattice()

        # Create the grid
        self.grid = House(self.lattice, self.room_w, self.room_h,
                            width=width, height=height)

        self.width = self.grid.width
        self.height = self.grid.height

        self.agent_pos = self.place_agent()
        self.agent_dir = self._rand_int(0,4)

        # Generate the mission string
        self.mission = 'go to the end'

    def place_agent(self, **kwargs):
        """Place agent in random tile in start room."""

        # TODO the coordinate system we're using in Lattice and House is inverted
        # about the Y axis wrt the one used to render envs
        # This means that if we place the agent in the (0,0) room - bottom left -
        # it gets rendered in the top left

        ## TODO: this whole block should be a subroutine
        i, j = self.lattice.start


        xL = i * self.room_w + i
        yB = j * self.room_h + j
        xR = xL + self.room_w + 1
        yT = yB + self.room_h + 1

        ##

        x = random.choice(range(xL+2, xR-1))
        y = random.choice(range(yB+2, yT-1))

        return (x, y)

    def get_room(self):
        """
        Get lattice coordinates of room the agent is in.
        If the agent is crossing a door, return the room it is facing to.
        If crossing a horizontal door and facing up/down, return the left room.
        If crossing a vertical door and facing left/right, return the bottom room.
        """
        ax, ay = self.agent_pos
        dx, dy = self.dir_vec

        # Compute current room coords in lattice
        ri = (ax-1)/(self.room_w+1)
        ri += 1 if ri.is_integer() and dx==1 else 0

        rj = (ay-1)/(self.room_h+1)
        rj += 1 if rj.is_integer() and dy==1 else 0

        return int(ri), int(rj)

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        # Compute room coordinates in lattice
        ri, rj = self.get_room()

        # Compute the absolute coordinates of the top-left view corner
        tx = 1 + ri*(self.room_w)
        ty = 1 + rj*(self.room_h)

        # Translate
        vx = i - tx
        vy = j - ty

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set - what does this even mean?
        """

        # Compute room coordinates in lattice
        ri, rj = self.get_room()

        # Compute the absolute coordinates of the top-left and bottom-right corners
        tx = 1 + ri*(self.room_w)
        ty = 1 + rj*(self.room_h)
        bx = 1 + (ri+1)*(self.room_w)
        by = 1 + (rj+1)*(self.room_h)

        return tx, ty, bx, by

#    def step(self, action):
#        self.step_count += 1
#
#        reward = 0
#        done = False
#
#        cur_pos = self.agent_pos
#
#        # Move up
#        if action == self.actions.up:
#                fwd_pos = cur_pos + np.array((0,-1))
#                fwd_cell = self.grid.get(*fwd_pos)
#                if fwd_cell == None or fwd_cell.can_overlap():
#                        self.agent_pos = fwd_pos
#                if fwd_cell != None and fwd_cell.type == 'goal':
#                        done = True
#                        reward = self._reward()
#                if fwd_cell != None and fwd_cell.type == 'lava':
#                        done = True
#
#        # Move down
#        elif action == self.actions.down:
#                fwd_pos = cur_pos + np.array((0,1))
#                fwd_cell = self.grid.get(*fwd_pos)
#                if fwd_cell == None or fwd_cell.can_overlap():
#                        self.agent_pos = fwd_pos
#                if fwd_cell != None and fwd_cell.type == 'goal':
#                        done = True
#                        reward = self._reward()
#                if fwd_cell != None and fwd_cell.type == 'lava':
#                        done = True
#
#        # Move left
#        elif action == self.actions.left:
#                fwd_pos = cur_pos + np.array((-1,0))
#                fwd_cell = self.grid.get(*fwd_pos)
#                if fwd_cell == None or fwd_cell.can_overlap():
#                        self.agent_pos = fwd_pos
#                if fwd_cell != None and fwd_cell.type == 'goal':
#                        done = True
#                        reward = self._reward()
#                if fwd_cell != None and fwd_cell.type == 'lava':
#                        done = True
#
#        # Move right
#        elif action == self.actions.right:
#                fwd_pos = cur_pos + np.array((1,0))
#                fwd_cell = self.grid.get(*fwd_pos)
#                if fwd_cell == None or fwd_cell.can_overlap():
#                        self.agent_pos = fwd_pos
#                if fwd_cell != None and fwd_cell.type == 'goal':
#                        done = True
#                        reward = self._reward()
#                if fwd_cell != None and fwd_cell.type == 'lava':
#                        done = True
#
#        # Done action (not used by default)
#        elif action == self.actions.done:
#                pass
#
#        else:
#                assert False, "unknown action"
#
#        if self.step_count >= self.max_steps:
#                done = True
#
#        obs = self.gen_obs()
#
#        return obs, reward, done, {}


    def process_vis(self, grid, agent_pos):
        mask = np.zeros(shape=(grid.width, grid.height), dtype=np.bool)

        for i in range(0,agent_pos[0]):
                for j in range(0,agent_pos[1]):
                        mask[i, j] = True

        return mask


    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
                grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
                vis_mask = self.process_vis(grid, (self.agent_view_size, self.agent_view_size))
        else:
                vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = (self.agent_pos[0]%(self.room_w+1), self.agent_pos[1]%(self.room_h+1))
        if self.carrying:
                grid.set(*agent_pos, self.carrying)
        else:
                grid.set(*agent_pos, None)

        return grid, vis_mask


    def render(self, mode='human', close=False, highlight=True, tile_size=CELL_PIXELS):
        """
        Render the whole-grid human view
        """

        if close:
            if self.grid_render:
                self.grid_render.close()
            return

        if self.grid_render is None or self.grid_render.window is None or (self.grid_render.width != self.width * tile_size):
            from gym_minigrid.rendering import Renderer
            self.grid_render = Renderer(
                self.width * tile_size,
                self.height * tile_size,
                True if mode == 'human' else False
            )

        r = self.grid_render

        if r.window:
            r.window.setText(self.mission)

        r.beginFrame()

        # Render the whole grid
        self.grid.render(r, tile_size)

        # Draw the agent
        ratio = tile_size / CELL_PIXELS
        r.push()
        r.scale(ratio, ratio)
        r.translate(
            CELL_PIXELS * (self.agent_pos[0] + 0.5),
            CELL_PIXELS * (self.agent_pos[1] + 0.5)
        )
        r.rotate(self.agent_dir * 90)
        r.setLineColor(255, 0, 0)
        r.setColor(255, 0, 0)
        r.drawPolygon([
            (-12, 10),
            ( 12,  0),
            (-12, -10)
        ])
        r.pop()

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the absolute coordinates of the bottom-left corner
        # of the agent's view area
        tx = (self.room_w+1)*(self.agent_pos[0]//(self.room_w+1))
        ty = (self.room_h+1)*(self.agent_pos[1]//(self.room_h+1))
        top_left = (tx, ty)

        # For each cell in the visibility mask
        if highlight:
            for vis_j in range(0, self.agent_view_size):
                for vis_i in range(0, self.agent_view_size):
                    # If this cell is not visible, don't highlight it
                    if not vis_mask[vis_i, vis_j]:
                        continue

                    # Compute the world coordinates of this cell
                    abs_i = tx + vis_j
                    abs_j = ty + vis_i

                    # Highlight the cell
                    r.fillRect(
                        abs_i * tile_size,
                        abs_j * tile_size,
                        tile_size,
                        tile_size,
                        255, 255, 255, 75
                    )

        r.endFrame()

        if mode == 'rgb_array':
            return r.getArray()
        elif mode == 'pixmap':
            return r.getPixmap()
        return r

# TODO figure out why this is giving problems at runtime in ipython
register(
	id='MiniGrid-House-5x5-v0',
	entry_point='gym_minigrid.envs:HouseEnv'
)

class Agent0:
    """
    Want an agent that given a graph outputs a string of [state, action] that takes it from the start to the end.
    We need two subagents (roomba and mapper) that each performs nx.shortest_path over a single room or the graph of rooms
    """

    def __init__(self, env, mapper=None, roomba=None):
        """
        Create agent_0 instance to act on environment env.

        @param env: HouseEnv instance.
        @param mapper: Mapper instance. Created on the fly if None.
        @param roomba: Roomba instance. Created on the fly if None.
        """

        # Inherit actions and position from env
        self._env = env
        self.actions = self._env.actions
        self.pos = self._env.agent_pos

        # Assign or create subagents
        self._mapper = mapper or Mapper(self._env)
        self._roomba = roomba or Roomba(self._env)

        # Calculate high level path across rooms
        self.map = self._env.lattice
        self.room_sequence = self._mapper.find_path(self._env.lattice)

        # Save history of observations
        self._all_obs = []

    def run(self):
        """Solve environment."""

        for next_room in self.room_sequence:
            # Generate initial observation in room
            # Recall image is a numpy array of shape (n+2,m+2,3)
            # if room dimensions are (n,m) - bc it includes walls
            # obs[:,:,0] is object type - look up OBJECT_TO_IDX dict
            # obs[:,:,1] is color - look up COLOR_TO_IDX dict
            # obs[:,:,2] is state - 0: open; 1: closed; 2: locked
            obs = self._env.gen_obs()['image']
            self._all_obs += [obs]

            # Calculate path within room
            self.path = self._roomba.find_path(obs, next_room)

            # Move following that path
            for step in self.path:
                obs, reward, done, _ = self._env.step(step)
                self._all_obs += [obs]

            # Ugly hack to deal w/ the fact that the agent view doesn't always change
            # when it steps through the door. This is bc the view is defined in absolute
            # terms using semi-open intervals like
            #   [a, b) X (c, d]
            # so that if the agent enters a room from the left or below it takes one step
            # longer to see the next room. IDK if we should or can "fix" this
            # Maybe it's better that the agent view is independent of where it came from
            if self._all_obs[-1] == self._all_obs[-2]:
                obs, reward, done, _ = self._env.step(step)
                self._all_obs += [obs]

        assert done, 'Agent could not find reward'

class Mapper:

    def __init__(self, env):
        self._env = env

    def find_path(self, lattice):
        """Find shortest path in map and return sequence of actions to be passed down to Roomba."""

        # TODO check the NetworkX routine you're calling breaks ties

        shortest_path = lattice.shortest_path() #Since we specify start and end shortest_path is a list
        actions = []

        #Lets calculate the list of actions we should take
        for i, room in enumerate(shortest_path):
            if i==0:
                prev_room = room
                continue

            encoded_action = room[0]-prev_room[0], room[1]-prev_room[1]

            if encoded_action == (-1,0):
                actions.append(0)         #Move left
            elif encoded_action == (1,0):
                actions.append(1)         #Move right
            elif encoded_action == (0,1):
                actions.append(3)          #Move down
            elif encoded_action == (0,-1):
                actions.append(2)           #Move up
            else:
                raise ValueError('Room increment {} makes no sense'.format(encoded_action))

            prev_room = room

        return actions

class Roomba:
    """
    Recall that the format of rooms is a list of ((i,j),room_w,room_h, list of doors to the right and left)
    """

    def __init__(self, env):
        self._env = env

    def find_path(self, obs, next_room):
        """Find path within room."""

        objects = obs[:,:,0]
        states = obs[:,:,2]

        goal = list(zip(*np.where(objects==OBJECT_TO_IDX['goal'])))

        if goal:
            assert len(goal)==1, 'Cannot have more than one reward!'
        else:
            goal = self._get_door(obs, next_room)

        cur_pos = self._env.get_view_coords(self._env.agent_pos)

        return self._min_path(cur_pos, goal, obs)

    def _min_path(self, initial, final, obs):
        """Find shortest path between initial and final position, handling obstacles."""

        # TODO handle obstacles

        x0, y0 = initial
        x1, y1 = final
        sx, sy, _ = obs.shape

        # Start horizontally or vertically?
        if x0==0 or x0==sx-1:
            moves = abs(x1-x0), abs(y1-y0)
        elif y0==0 or y0==sy-1:
            moves = abs(y1-y0), abs(x1-x0)
        else:
            # initial case only, order doesn't matter
            moves = abs(x1-x0), abs(y1-y0)

        # Turn left or right?
        if np.sign(x1-x0)==np.sign(y1-y0):
            turn = self._env.actions.right
        else:
            turn = self._env.actions.left

        # Move in 1st dimension
        for _ in range(moves[0]):
            actions += [self._env.actions.forward]

        # Turn
        actions += [turn]

        # Move in 2nd dimension
        for _ in range(moves[1]):
            actions += [self._env.actions.forward]

        return actions

    def _get_door(self, obs, next_room):
        """Find door given wall and observation."""

        objects = obs[:,:,0]
        states = obs[:,:,2]
        doors = list(zip(*np.where(objects==OBJECT_TO_IDX['door'])))
        print(doors)
        print(next_room)

        if next_room == 0:
            door = [d for d in doors if d[0]==0] # left
        elif next_room == 1:
            door = [d for d in doors if d[1]==objects.shape[1]] # right
        elif next_room == 3:
            door = [d for d in doors if d[0]==objects.shape[0]] # down
        elif next_room == 2:
            door = [d for d in doors if d[1]==0] # up
        else:
            raise ValueError('Unknown wall position: {}'.format(next_room))

        print(door)

        return door[0]

class House(Grid):
    """
    Creates whole house environment from underlying lattice graph, adding doors, objects, rewards, obstacles, etc.
    """

    # TODO property and setters/getters to hide cumbersome checks

    # Room sizes
    MAX_ROOM_WIDTH = 8
    MAX_ROOM_HEIGHT = 8
    MIN_ROOM_WIDTH = 3
    MIN_ROOM_HEIGHT = 3

    def __init__(self, lattice, room_w=None, room_h=None, obstacles=True, doors_open=True, verbose=False, width=None, height=None):
        """
        Create house from lattice graph.

        @param lattice: Lattice instance.
        @param room_w, room_h: Positive integers. Room width and height. Random if not provided.
        @param obstacles: Boolean. If True, put an obstacle (lava) in every room.
        @param doors_open: Boolean. If True, all doors are open. Defaults to True.
        @param verbose: Boolean. If True, print stuff while building house. Mostly for debugging.
        """
        self.lattice = lattice
        self.dim = lattice.dim

        if room_w is None:
            room_w = random.choice(range(self.MIN_ROOM_WIDTH, self.MAX_ROOM_WIDTH+1))
            room_h = random.choice(range(self.MIN_ROOM_HEIGHT, self.MAX_ROOM_HEIGHT+1))

        # TODO assert h/w within boundaries

        self.room_w = room_w
        self.room_h = room_h

        #width = self.dim[0]*self.room_w+self.dim[0]+1
        #height = self.dim[1]*self.room_h+self.dim[1]+1

        # TODO fine-grained control of obstacles instead of single boolean
        # We could try to handle, from less to more random:
        # - dict with exact obstacle coords
        # - list of rooms & # of obstacles in each
        # - prob of obstacle per room
        # - whether to put obstacles or not - current implementation
        self.obstacles = obstacles
        self.doors_open = doors_open

        self.verbose = verbose

        if self.verbose:
            print('There are {}-by-{} rooms'.format(*self.dim))
            print('Each room has {}-by-{} tiles'.format(self.room_w, self.room_h))
            print('Whole house has {}-by-{} tiles'.format(width, height))

        super().__init__(width=width, height=height)

        self.shape = self.width, self.height

        self._build_outer_walls()
        self._build_inner_walls()
        self._add_reward()
        self._add_obstacles()

    def plot(self, ax=None):
        """Diagnostic plot."""

        if ax is None:
            dims = (1,2) if self.width >= self.height else (2,1)
            fig, ax = plt.subplots(1,2)

        img = self.encode()

        self.lattice.plot(ax=ax[0])
        ax[1].imshow(img[:,:,0].T, origin='lower')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        return ax

    def _build_outer_walls(self):
        """Build surrounding walls."""

        self.horz_wall(0, 0)
        self.horz_wall(0, self.height-1)
        self.vert_wall(0, 0)
        self.vert_wall(self.width-1, 0)

    def _build_inner_walls(self):
        """Build walls between rooms, doors included."""

        # Loop over rooms
        for (i,j) in self.lattice.nodes:
            if i==self.dim[0] or j==self.dim[1]: # discard rightmost and top nodes
                continue

            xL = i * self.room_w + i
            yB = j * self.room_h + j
            xR = xL + self.room_w + 1
            yT = yB + self.room_h + 1

            # Right vertical wall
            if self.verbose: print('Adding vertical wall of height {} at ({}, {})'.format(self.room_h, xR, yB))
            self.vert_wall(xR, yB, self.room_h+1)

            # Top horizontal wall
            if self.verbose: print('Adding horizontal wall of width {} at ({}, {})'.format(self.room_w, xL, yT))
            self.horz_wall(xL, yT, self.room_w+1)

            # Add doors if needed
            doors = []
            if ((i,j),(i+1,j)) in self.lattice.edges:
                coords=(xR, random.choice(range(yB+1,yT)))
                self._add_door(coords)
                doors += coords
            if ((i,j),(i,j+1)) in self.lattice.edges:
                coords=(random.choice(range(xL+1,xR)), yT)
                self._add_door(coords)
                doors += coords

            #Construct list of rooms. Format ((i,j),room_w, room_h, list_of_doors to the right and up)
            #rooms += ((i,j), room_w, room_h, doors)


    def _add_door(self, coords):
        """Add door at coords = (x,y)"""
        #TODO assert (x,y) is a wall
        self.set(*coords, Door(color='purple', is_open=self.doors_open, is_locked=False))

    def _add_reward(self):
        """Add reward in final room."""

        i, j = self.lattice.end

        xL = i * self.room_w + i
        yB = j * self.room_h + j
        xR = xL + self.room_w + 1
        yT = yB + self.room_h + 1

        if self.verbose:
            print('Adding reward in room ({},{})'.format(i,j))
            print('Choosing tile in square [{},{}] X [{},{}]'.format(xL+2, xR, yB+2, yT))

        x = random.choice(range(xL+2, xR-1))
        y = random.choice(range(yB+2, yT-1))

        self.set(x, y, Goal())

        # TODO how does this get converted to reward?

    def _add_obstacles(self):
        pass

        # TODO add obstacles
        # TODO other objects?


class Lattice:
    """
    Connected subgraph of a lattice graph.
    """

    # Max lattice size
    MIN_DIM = 2 # 2-by-2
    MAX_DIM = 5 # 5-by-5

    # Node colors
    BASE_COLOR = 'gray'
    START_COLOR = 'black'
    END_COLOR = 'green'

    def __init__(self, dim=None, edges=None, start=None, end=None):
        """
        Create graph.

        @param dim: Array with lattice dimensions e.g. a 2-by-3 grid should have dim = [2,3].
        @param edges: Array or set of edges as coordinate pairs. Random if not given.
        """
        # Create base graph containing all edges in lattice
        if dim is None:
           x = random.choice(range(self.MIN_DIM, self.MAX_DIM+1))
           y = random.choice(range(self.MIN_DIM, self.MAX_DIM+1))
           dim = [x,y]

        self.dim = dim

        # I have to permute dim here because graphs are defined as [m_rows, n_cols] but I want to do [x, y]
        self._base_graph = nx.grid_graph(dim=[self.dim[1], self.dim[0]])

        # Select subgraph either from edges variable or randomly
        if edges is not None:
            self._graph = nx.Graph(nodes=self._base_graph.nodes, edges=edges)
            assert nx.is_connected(self._graph), 'Wrong edges: the graph should be connected!'
        else:
            self._graph = get_random_connected_subgraph(self._base_graph)

        # Generate start and end if not given and make sure they're not the same node
        if start and end:
            assert start!=end, 'Start and end nodes should be different!'
        if start is None:
            start = random.choice([node for node in list(self._graph.nodes) if node!=end])
        if end is None:
            end = random.choice([node for node in list(self._graph.nodes) if node!=start])

        self.start = start
        self.end = end

        # Color graph according to start/end
        self._colors = {n: self.BASE_COLOR for n in self._graph.nodes}
        self._colors[self.start] = self.START_COLOR
        self._colors[self.end] = self.END_COLOR

        nx.set_node_attributes(self._graph, self._colors, 'color')

        # Make these visible
        self.nodes = self._graph.nodes
        self.edges = self._graph.edges

    def shortest_path(self):
        """Shortest path between start and end."""
        return nx.shortest_path(self._graph, self.start, self.end)

    def plot(self, ax=None, node_kwds={'s': 100}, edge_kwds={'color': 'gray'}):
        # TODO nodes (not edges) should be overlaid at the forefront
        if ax is None:
            fig, ax = plt.subplots()

        [ax.plot(*zip(*edge), **edge_kwds) for edge in self._graph.edges]
        [ax.scatter(*node, color=self._graph.nodes[node]['color'], **node_kwds) for node in self._graph.nodes]

        return ax

def get_random_connected_subgraph(graph):
    # TODO add sparseness parameter
    """
    Find random connected subgraph by deleting one edge at a time and stopping when disconnected.
    """

    while nx.is_connected(graph):
        edges = list(graph.edges)
        chosen = random.choice(edges)
        graph.remove_edge(*chosen)

    # Need to undo last removal because it disconnects the graph
    graph.add_edge(*chosen)

    return graph
