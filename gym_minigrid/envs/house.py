from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from inputs import House, Lattice
import random

class HouseEnv(MiniGridEnv):
	"""
	House environment for IDA
	"""

	class Actions(IntEnum):
		# Turn left, turn right, move forward
		left = 0
		right = 1
		up = 2
		down = 3
		done = 6

	def __init__(
		self,
		room_w = 5,
		room_h = 5
	):
		self.room_w = room_w
		self.room_h = room_h

		super().__init__(
			grid_size=0,
			max_steps=5*5**2,
			# Set this to True for maximum speed
			see_through_walls=False,
			agent_view_size=self.room_w+1
		)

		self.actions = self.Actions

	def _gen_grid(self, width, height):
		# Gen lattice
		self.lattice = Lattice()
		
		# Create the grid
		self.grid = House(self.lattice, self.room_w, self.room_h)

		self.width = self.grid.width
		self.height = self.grid.height

		self.agent_pos = self.place_agent()
		self.agent_dir = 0;

		# Generate the mission string
		self.mission = 'go to the end'

	def place_agent(self, **kwargs):
		(ry,rx) = self.lattice.start
		return np.array(random.choice([(i,j) for i in range((1+self.room_w)*rx+1,(1+self.room_w)*(rx+1)-2) for j in range((1+self.room_h)*(ry)+1, (1+self.room_h)*(ry+1)-2)]))


	def get_view_coords(self, i, j):
		"""
		Translate and rotate absolute grid coordinates (i, j) into the
		agent's partially observable view (sub-grid). Note that the resulting
		coordinates may be negative or outside of the agent's view size.
		"""

		ax, ay = self.agent_pos

		# Compute the absolute coordinates of the top-left view corner
		tx = (self.room_w+1)*(ax%(self.room_w+1))
		ty = (self.room_h+1)*(ay%(self.room_h+1))

		vx = i - tx
		vy = j - ty

		return vx, vy

	def get_view_exts(self):

		ax, ay = self.agent_pos

		tx = (self.room_w+1)*(ax%(self.room_w+1))
		ty = (self.room_h+1)*(ay%(self.room_h+1))
		bx = tx + self.room_w+2
		by = ty + self.room_h+2
		return (tx, ty, bx, by)

	def step(self, action):
		self.step_count += 1

		reward = 0
		done = False

		cur_pos = self.agent_pos

		# Move up
		if action == self.actions.up:
			fwd_pos = cur_pos + np.array((0,-1))
			fwd_cell = self.grid.get(*fwd_pos)
			if fwd_cell == None or fwd_cell.can_overlap():
				self.agent_pos = fwd_pos
			if fwd_cell != None and fwd_cell.type == 'goal':
				done = True
				reward = self._reward()
			if fwd_cell != None and fwd_cell.type == 'lava':
				done = True

		# Move down
		elif action == self.actions.down:
			fwd_pos = cur_pos + np.array((0,1))
			fwd_cell = self.grid.get(*fwd_pos)
			if fwd_cell == None or fwd_cell.can_overlap():
				self.agent_pos = fwd_pos
			if fwd_cell != None and fwd_cell.type == 'goal':
				done = True
				reward = self._reward()
			if fwd_cell != None and fwd_cell.type == 'lava':
				done = True

		# Move left
		elif action == self.actions.left:
			fwd_pos = cur_pos + np.array((-1,0))
			fwd_cell = self.grid.get(*fwd_pos)
			if fwd_cell == None or fwd_cell.can_overlap():
				self.agent_pos = fwd_pos
			if fwd_cell != None and fwd_cell.type == 'goal':
				done = True
				reward = self._reward()
			if fwd_cell != None and fwd_cell.type == 'lava':
				done = True

		# Move right
		elif action == self.actions.right:
			fwd_pos = cur_pos + np.array((1,0))
			fwd_cell = self.grid.get(*fwd_pos)
			if fwd_cell == None or fwd_cell.can_overlap():
				self.agent_pos = fwd_pos
			if fwd_cell != None and fwd_cell.type == 'goal':
				done = True
				reward = self._reward()
			if fwd_cell != None and fwd_cell.type == 'lava':
				done = True

		# Done action (not used by default)
		elif action == self.actions.done:
			pass

		else:
			assert False, "unknown action"

		if self.step_count >= self.max_steps:
			done = True

		obs = self.gen_obs()

		return obs, reward, done, {}


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


register(
	id='MiniGrid-House-5x5-v0',
	entry_point='gym_minigrid.envs:HouseEnv'
)
