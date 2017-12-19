from gym_minigrid.minigrid import *
from gym_minigrid.register import register

class Room:
    def __init__(
        self,
        top,
        size,
        color,
        objects
    ):
        self.top = top
        self.size = size

        # Color of the room
        self.color = color

        # List of objects contained
        self.objects = objects

class FourRoomQAEnv(MiniGridEnv):
    """
    Environment to experiment with embodied question answering
    https://arxiv.org/abs/1711.11543
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        left = 0
        right = 1
        forward = 2
        toggle = 3
        say = 4

    def __init__(self, size=16):
        assert size >= 10
        super(FourRoomQAEnv, self).__init__(gridSize=size, maxSteps=8*size)

        # Action enumeration for this environment
        self.actions = MiniGridEnv.Actions

        # TODO: dictionary action_space, to include answer sentence?
        # Actions are discrete integer values
        self.action_space = spaces.Discrete(len(self.actions))

        # TODO: dictionary observation_space, to include question?

    def _genGrid(self, width, height):
        grid = Grid(width, height)

        # Horizontal and vertical split indices
        vSplitIdx = self._randInt(5, width-4)
        hSplitIdx = self._randInt(5, height-4)

        # Create the four rooms
        self.rooms = []
        self.rooms.append(Room(
            (0, 0),
            (vSplitIdx, hSplitIdx),
            'red',
            []
        ))
        self.rooms.append(Room(
            (vSplitIdx, 0),
            (width - vSplitIdx, hSplitIdx),
            'purple',
            []
        ))
        self.rooms.append(Room(
            (0, hSplitIdx),
            (vSplitIdx, height - hSplitIdx),
            'blue',
            []
        ))
        self.rooms.append(Room(
            (vSplitIdx, hSplitIdx),
            (width - vSplitIdx, height - hSplitIdx),
            'yellow',
            []
        ))

        # Place the room walls
        for room in self.rooms:
            x, y = room.top
            w, h = room.size

            # Horizontal walls
            for i in range(w):
                grid.set(x + i, y, Wall(room.color))
                grid.set(x + i, y + h - 1, Wall(room.color))

            # Vertical walls
            for j in range(h):
                grid.set(x, y + j, Wall(room.color))
                grid.set(x + w - 1, y + j, Wall(room.color))

        # Place wall openings connecting the rooms
        hIdx = self._randInt(1, hSplitIdx-1)
        grid.set(vSplitIdx, hIdx, None)
        grid.set(vSplitIdx-1, hIdx, None)
        hIdx = self._randInt(hSplitIdx+1, height-1)
        grid.set(vSplitIdx, hIdx, None)
        grid.set(vSplitIdx-1, hIdx, None)







        # TODO: pick a random room to be the subject of the question
        # TODO: identify unique objects


        # TODO:
        # Generate a question and answer
        self.question = ''

        # Question examples:
        # - What color is the X?
        # - What color is the X in the ROOM?
        # - What room is the X located in?
        # - What color is the X in the blue room?
        # - How many rooms contain chairs?
        # - How many keys are there in the yellow room?
        # - How many <OBJs> in the <ROOM>?

        #self.answer





        return grid


    def _reset(self):
        obs = MiniGridEnv._reset(self)

        obs = {
            'image': obs,
            'question': self.question
        }

        return obs

    def _step(self, action):
        obs, reward, done, info = MiniGridEnv._step(self, action)

        obs = {
            'image': obs,
            'question': self.question
        }

        return obs, reward, done, info







register(
    id='MiniGrid-FourRoomQA-v0',
    entry_point='gym_minigrid.envs:FourRoomQAEnv'
)
