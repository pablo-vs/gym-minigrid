"""
House Environment and related utilities.
"""

from gym_minigrid.minigrid import Grid, MiniGridEnv, Floor, Door, Wall, Goal
from gym_minigrid.minigrid import OBJECT_TO_IDX, IDX_TO_OBJECT, IDX_TO_COLOR
from gym_minigrid.register import register

from enum import IntEnum

import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

CELL_PIXELS = 32


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

    def __init__(self, lattice, room_w=None, room_h=None, obstacles=True, doors_open=True, verbose=False):
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

        # TODO we'll consider non-square rooms in the future
        assert room_h == room_w

        if room_w is None:
            room_w = random.choice(range(self.MIN_ROOM_WIDTH, self.MAX_ROOM_WIDTH+1))
            room_h = random.choice(range(self.MIN_ROOM_HEIGHT, self.MAX_ROOM_HEIGHT+1))

        assert House.MIN_ROOM_WIDTH <= room_w <= House.MAX_ROOM_WIDTH
        self.room_w = room_w

        assert House.MIN_ROOM_HEIGHT <= room_w <= House.MAX_ROOM_HEIGHT
        self.room_h = room_h

        width = self.dim[0] * (self.room_w+self.dim[0] + 1) + 1
        height = self.dim[1] * (self.room_h+self.dim[1] + 1) + 1

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

        self._set_floor()
        self._build_outer_walls()
        self._build_inner_walls()
        self._add_reward()
        self._add_obstacles()

        # NOTE: None tiles are reserved for the agent - see self.encode
        assert all([tile is not None for tile in self.grid])

    def plot(self, ax=None, agent_plot=None):
        """Diagnostic plot."""

        if ax is None:
            dims = (1,2) if self.width >= self.height else (2,1)
            fig, ax = plt.subplots(1,2)

        img = self.encode()

        if agent_plot is not None:
            i, j, v = agent_plot
            img[i, j, 0] = v

        self.lattice.plot(ax=ax[0])
        ax[1].imshow(img[:,:,0]) #.T, origin='lower')

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])

        return ax

    @staticmethod
    def decode(array):
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 3

        grid = Grid(width, height)
        for i in range(width):
            for j in range(height):
                typeIdx, colorIdx, state = array[i, j]

                if typeIdx == OBJECT_TO_IDX['unseen'] or \
                        typeIdx == OBJECT_TO_IDX['empty']:
                    continue

                objType = IDX_TO_OBJECT[typeIdx]
                color = IDX_TO_COLOR[colorIdx]
                # State, 0: open, 1: closed, 2: locked
                is_open = state == 0
                is_locked = state == 2

                if objType == 'wall':
                    v = Wall(color)
                elif objType == 'floor':
                    v = Floor(color)
                elif objType == 'ball':
                    v = Ball(color)
                elif objType == 'key':
                    v = Key(color)
                elif objType == 'box':
                    v = Box(color)
                elif objType == 'door':
                    v = Door(color, is_open, is_locked)
                elif objType == 'goal':
                    v = Goal()
                elif objType == 'lava':
                    v = Lava()
                elif objType == 'agent':
                    v = None
                else:
                    assert False, "unknown obj type in decode '%s'" % objType

                grid.set(i, j, v)

        return grid

    def _set_floor(self):
        """Cover entire grid with floor tiles."""
        [self.set(i, j, Floor()) for i in range(self.width) for j in range(self.height)]

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
            #if i==self.dim[0] or j==self.dim[1]: # discard rightmost and top nodes
            #    continue

            xL = i * self.room_w + i
            yT = j * self.room_h + j
            xR = xL + self.room_w + 1
            yB = yT + self.room_h + 1

            # Right vertical wall
            if self.verbose: print('Adding vertical wall of height {} at ({}, {})'.format(self.room_h, xR, yB))
            self.vert_wall(xR, yT, self.room_h+1)

            # Bottom horizontal wall
            if self.verbose: print('Adding horizontal wall of width {} at ({}, {})'.format(self.room_w, xL, yT))
            self.horz_wall(xL, yB, self.room_w+1)

            # Add doors if needed
            doors = []
            if ((i,j),(i+1,j)) in self.lattice.edges:
                coords=(xR, random.choice(range(yT+1, yB)))
                self._add_door(coords)
                doors += coords
            if ((i,j),(i,j+1)) in self.lattice.edges:
                coords=(random.choice(range(xL+1, xR)), yB)
                self._add_door(coords)
                doors += coords

            #Construct list of rooms. Format ((i,j),room_w, room_h, list_of_doors to the right and up)
            #rooms += ((i,j), room_w, room_h, doors)


    def _add_door(self, coords):
        """Add door at coords = (x,y)"""
        x, y = coords
        assert isinstance(self.get(x,y), Wall), 'You can\'t put a door outside of a wall!'
        self.set(x, y, Door(color='purple', is_open=self.doors_open, is_locked=False))

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
            self._graph = nx.Graph()
            self._graph.add_nodes_from(self._base_graph.nodes)
            self._graph.add_edges_from(edges)
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

        ax.invert_yaxis()

        return ax


class HouseEnv(MiniGridEnv):
    """
    House environment for IDA
    """

    MAX_GRID_SIZE = Lattice.MAX_DIM*(max(House.MAX_ROOM_HEIGHT+1, House.MAX_ROOM_HEIGHT)+1)+1

    def __init__(
            self,
            dim = None, # Lattice params
            edges = None,
            start = None,
            end = None,
            room_w = None, # House params
            room_h = None,
            obstacles = False,
            doors_open = True,
            verbose = False,
            padding = True,
            size = None, # Total grid size - only when padding
            pos_in_room = None, # Agent position w/in start room
            agent_dir = None, # Agent direction in 0 to 3
    ):
        # Generate lattice
        # TODO these should be dynamically updated after lattice created
        self.dim = dim
        self.edges = edges
        self.start = start
        self.end = end
        self.lattice = Lattice(dim=self.dim, edges=self.edges, start=self.start, end=self.end)

        # Create the grid
        self.room_w = room_w
        self.room_h = room_h
        self.obstacles = obstacles
        self.doors_open = doors_open
        self.grid = House(
                lattice = self.lattice,
                room_w = self.room_w,
                room_h = self.room_h,
                )

        self.width = self.grid.width
        self.height = self.grid.height
        self.room_w = self.grid.room_w
        self.room_h = self.grid.room_h
        self.padding = padding

        if self.padding:
            self.size = size or HouseEnv.MAX_GRID_SIZE
        else:
            self.size = max(self.widht, self.height) # this doesn't make a lot of sense but ok

        self.pos_in_room = pos_in_room
        self._hidden_agent_dir = agent_dir

        super().__init__(
                grid_size=size,
                max_steps=5*5**2,
                # Set this to True for maximum speed
                see_through_walls=True, #False,
                agent_view_size=max(self.room_w, self.room_h)+2
        )

        self.actions = self.Actions

    def _reward(self):
        """
        Compute the reward to be given upon success
        """
        # TODO copied it bc maybe we want to override it?

        return 1 - 0.9 * (self.step_count / self.max_steps)

    def _gen_grid(self, width=None, height=None):

        widht = self.grid.width
        height = self.grid.height

        self.agent_pos = self.place_agent(self.pos_in_room)

        if self._hidden_agent_dir is None:
            self.agent_dir = self._rand_int(0,4)
        else:
            self.agent_dir = self._hidden_agent_dir

        # Generate the mission string
        self.mission = 'go to the end'

    def place_agent(self, pos_in_room=None):
        """
        Place agent in start room.

        @param pos_in_room: Pair (x,y) indicating tile in room. Defaults to random.
        """

        ## TODO: this whole block should be a subroutine
        i, j = self.lattice.start

        xL = i*(self.room_w+1)
        yT = j*(self.room_h+1)
        xR = xL + self.room_w + 1
        yB = yT + self.room_h + 1

        ##

        if pos_in_room is not None:
            dx, dy = pos_in_room
            x, y = xL+dx, yT+dy
            assert xL<x<xR, 'Room is not wide enough'
            assert yT<y<yB, 'Room is not tall enough'
        else:
            x = self._rand_int(xL+1, xR)
            y = self._rand_int(yT+1, yB)

        assert isinstance(self.grid.get(x,y), Floor)

        return (x, y)

    def get_room(self):
        """
        Get lattice coordinates of room the agent is in.
        If the agent is crossing a door, return the room it is facing to.
        If crossing a horizontal door and facing up/down, return the right room.
        If crossing a vertical door and facing left/right, return the bottom room.
        """
        ax, ay = self.agent_pos
        dx, dy = self.dir_vec

        # Compute current room coords in lattice
        ri = ax/(self.room_w+1)
        ri -= 1 if ri.is_integer() and dx==-1 else 0

        rj = ay/(self.room_h+1)
        rj -= 1 if rj.is_integer() and dy==-1 else 0

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
        tx = ri*(self.room_w+1)
        ty = rj*(self.room_h+1)

        # Translate
        vx = i - tx
        vy = j - ty

        # Rotate onto agent's reference frame
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        vx = (rx*vx + ry*vy)
        vy = -(dx*vx + dy*vy)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set - what does this even mean?
        """

        # Compute room coordinates in lattice
        ri, rj = self.get_room()

        # Compute the absolute coordinates of the top-left and bottom-right corners
        tx = ri*(self.room_w+1)
        ty = rj*(self.room_h+1)
        bx = (ri+1)*(self.room_w+1)
        by = (rj+1)*(self.room_h+1)

        return tx, ty, bx, by

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        # agent_view_size is now a tuple to account for non-square rooms
        dw, dh = self.room_w+2, self.room_h+2

        grid = self.grid.slice(topX, topY, dw, dh)

        # Plug this into grid.process_vis
        rel_pos = (self.room_w+2)//2, self.room_h+1

        #print('Agent direction: {}'.format(self.dir_vec))
        #print('Grid shape (before rotation): {}'.format((grid.width, grid.height)))

        for i in range(self.agent_dir + 1):
            #print('Rotate!')
            dw, dh = dh, dw
            grid = grid.rotate_left()

        #print('Grid shape (after rotation): {}'.format((grid.width, grid.height)))

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            #vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
            vis_mask = grid.process_vis(agent_pos=((dw//2, dh-1)))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=np.bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask, agent_dir=self.agent_dir)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

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
        tx, ty, _, _ = self.get_view_exts()
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
        self.room_sequence = self._mapper.find_path()

        # Save history of observations
        self._all_obs = []

        # Generate initial observation
        self._cur_obs = self._env.gen_obs()['image']

    def run(self):
        """Solve environment."""

        steps = []

        for next_room in self.room_sequence + ['last_room']:
            # Generate initial observation in room
            # Recall image is a numpy array of shape (n+2,m+2,3)
            # if room dimensions are (n,m) - bc it includes walls
            # obs[:,:,0] is object type - look up OBJECT_TO_IDX dict
            # obs[:,:,1] is color - look up COLOR_TO_IDX dict
            # obs[:,:,2] is state - 0: open; 1: closed; 2: locked
            #obs = self._env.gen_obs()['image']
            #self._all_obs += [obs]
            print('Next room: {}'.format(next_room))

            # Plot for debugging
            agent_position_plot = tuple(self._env.agent_pos) + (1,)
            print(agent_position_plot)
            self._env.grid.plot(agent_plot=agent_position_plot)
            plt.show()

            # Calculate path within room
            self.path = self._roomba.find_path(self._cur_obs, next_room)
            print(self.path)

            # Move following that path
            for step in self.path:
                self._cur_obs, _, _, done = self._env.step(step)
                # TODO this is ugly - let the lower function deal with the whole dict
                self._cur_obs = self._cur_obs['image']
                steps += [step]
                #obs, reward, done, _ = self._env.step(step)
                #self._all_obs += [obs]

        return steps

        #assert done, 'Agent could not find reward'


class Mapper:

    def __init__(self, env):
        self._env = env

    def find_path(self, lattice=None):
        """Find shortest path in map and return sequence of actions to be passed down to Roomba."""

        lattice = lattice or self._env.lattice

        # TODO check the NetworkX routine you're calling breaks ties - pretty sure it does

        shortest_path = lattice.shortest_path() #Since we specify start and end shortest_path is a list
        actions = []

        #Lets calculate the list of actions we should take
        for i, room in enumerate(shortest_path):
            if i==0:
                prev_room = room
                continue

            action = room[0]-prev_room[0], room[1]-prev_room[1]
            actions.append(action)
#
#            if encoded_action == (-1,0):
#                actions.append(0)         #Move left
#            elif encoded_action == (1,0):
#                actions.append(1)         #Move right
#            elif encoded_action == (0,1):
#                actions.append(3)          #Move down
#            elif encoded_action == (0,-1):
#                actions.append(2)           #Move up
#            else:
#                raise ValueError('Room increment {} makes no sense'.format(encoded_action))

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
        if next_room == 'last_room':
            print("Hello")
        goal = list(zip(*np.where(objects==OBJECT_TO_IDX['goal'])))

        if goal:
            assert len(goal)==1, 'Cannot have more than one reward!'
        else:
            goal = self._get_door(obs, next_room)

        # Current position in agent's reference frame
        cur_pos = list(zip(*np.where(objects>=OBJECT_TO_IDX['agent']))) #>= bc != directions encoded too

        if cur_pos:
            assert len(cur_pos)==1, 'There is more than one agent!'
            cur_pos = cur_pos[0]
        else:
            raise Exception('There are no agents!')
        #cur_pos = self._env.get_view_coords(*self._env.agent_pos)

        return self._min_path(cur_pos, goal, obs)

    def _min_path(self, initial, final, obs):
        """Find shortest path between initial and final position, handling obstacles."""

        # TODO handle obstacles

        print('Going from {} to {}'.format(initial, final))

        x0, y0 = initial
        x1, y1 = final
        dx, dy = x1-x0, y1-y0
        dir_vec = np.array([0,-1]) # in local coords
        sx, sy, _ = obs.shape

        cur_pos = np.array(initial)
        actions = []

        # If at door, take one step first to enter room
        at_door = obs[x0-1, y0, 0] == obs[x0+1, y0, 0] == OBJECT_TO_IDX['wall']
        if at_door:
            print('At door, taking one step forward')
            actions.append(self._env.actions.forward)

        # True if one of the coordinates is already equal to goal
        aligned = lambda cur_pos: (cur_pos[0]==x1) or (cur_pos[1]==y1)

        # Assumes already aligned
        def face_goal(cur_pos, dir_vec, final):
            # Vector from current position pointing towards goal
            dr = np.array(final) - cur_pos

            if np.dot(dr, dir_vec) != 0: # pointing backwards
                return [self._env.actions.left, self._env.actions.left], -dir_vec

            # Counterclockwise pi/2 rotation
            rot = np.array([[0,-1], [1,0]])

            """This was checking for np.dot > 0 but if it's non-zero we've already returned so it's not doing anything
            we need to check for which of the two perpendicular directions we are facing. We can do it from the det
            of the dr and dir vectors stacked (so the "cross-prod"), checked all cases and this should work. (If we stack
            them differently then we have to reverse the signs)"""

            dir_matrix = np.vstack((dr, dir_vec))
            det = np.linalg.det(dir_matrix)

            if det > 0:
                # turn right
                return self._env.actions.right, np.dot(-rot, dir_vec)
            else:
                # turn left
                return self._env.actions.left, np.dot(rot, dir_vec)

        def move_forward(actions, cur_pos, final, keep_cond, stop_cond=False):
            while keep_cond(cur_pos, final):
                actions.append(self._env.actions.forward)
                cur_pos += dir_vec
                print('Currently at: {}\tGoing to: {}'.format(tuple(cur_pos), tuple(final)))

                if stop_cond(cur_pos):
                    raise Exception('Loop went bad at {}'.format(cur_pos))

            return actions

        stop_cond = lambda cur_pos: (abs(cur_pos[0])>self._env.grid.width) or\
                (abs(cur_pos[1])>self._env.grid.height)

        if aligned(cur_pos): # move in only one dimension

            print('I am aligned!')

            # First check if facing goal...
            dx, dy = dir_vec
            facing_goal = ((x1-x0)/dx > 0) if dx!=0 else ((y1-y0)/dy>0)

            # ... and turn as needed if not
            if not facing_goal:
                print('But I am not facing goal')
                turn, dir_vec = face_goal(cur_pos, dir_vec, final)
                print('Turning {}'.format({0: 'left', 1: 'right'}[turn]))
                actions.append(turn)

            print('Moving forward...')
            # Then just move forward
            actions = move_forward(
                    actions,
                    cur_pos,
                    final,
                    keep_cond = lambda cur_pos, final: not np.array_equal(cur_pos, final),
                    stop_cond = stop_cond,
                    )

        else: # otherwise move in two dimensions

            print('I am not aligned')

            # Need to turn 180 deg if y1>y0
            if dy>0:
                print('Turning 180 degs')
                actions.append(self._env.actions.left)
                actions.append(self._env.actions.left)

            # Move until aligned
            print('Moving forward...')
            actions = move_forward(
                    actions,
                    cur_pos,
                    final,
                    keep_cond = lambda cur_pos, final: not aligned(cur_pos),
                    stop_cond = stop_cond,
                    )

            # Then turn as needed
            turn, dir_vec = face_goal(cur_pos, dir_vec, final)
            print('Turning {}'.format({0: 'left', 1: 'right'}[turn]))
            actions.append(turn)

            # Then keep moving forward
            print('Moving forward...')
            actions = move_forward(
                    actions,
                    cur_pos,
                    final,
                    keep_cond = lambda cur_pos, final: not np.array_equal(cur_pos, final),
                    stop_cond = stop_cond,
                    )

        return actions

    def _get_door(self, obs, next_room):
        """Find door given wall and observation."""

        next_room = tuple(map(int, self._rotate_vec(next_room)))

        objects = obs[:,:,0]
        states = obs[:,:,2]
        doors = list(zip(*np.where(objects==OBJECT_TO_IDX['door'])))

        """This had opposite coord system. 0,0 is top right so we need to transform door coordinates."""
        if next_room == (-1,0):
            door = [d for d in doors if d[0]==0] # left
        elif next_room == (1,0):
            door = [d for d in doors if d[0]==objects.shape[0]-1] # right
        elif next_room == (0,1):
            door = [d for d in doors if d[1]==objects.shape[1]-1] # down
        elif next_room == (0,-1):
            door = [d for d in doors if d[1]==0] # up
        else:
            raise ValueError('Unknown wall position: {}'.format(next_room))

        # TODO this fails sometimes because door is empty - IDK why
        # It always happens when next_room is (0,1) = bottom
        # but the doors are either only left or right
        try:

            return door[0]
        except IndexError:
            # Where am I going?
            going = {
                    (0,1): 'bottom',
                    (0,-1): 'top',
                    (1,0): 'right',
                    (-1,0): 'left',
                    }[next_room]

            # Where are the doors?
            wheres = []

            for door in doors:
                if door[0] == 0:
                    where = 'left'
                elif door[0] == objects.shape[0]-1:
                    where = 'right'
                elif door[1] == 0:
                    where = 'top'
                elif door[1] == objects.shape[1]-1:
                    where = 'bottom'

                wheres += [where]

                if where==going:
                    break
            else:
                raise Exception('Going to the {} but the available doors are: {}'.format(going, wheres))

    def _rotate_vec(self, vec):
        """Rotate vector in house's reference frame to agent's reference frame (vision pointing up)"""

        # Angle wrt up
        theta = -(self._env.agent_dir+1)*np.pi/2

        # Rotation
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        # New vector
        vec = np.dot(rot, np.array(vec))

        return vec


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


house = HouseEnv()
plt.show()
a0 = Agent0(house)
result = a0.run()
print("Done")
