# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.

from sys import platform
import os
from goexplore_py.randselectors import *
from goexplore_py.goexplore import *
#import goexplore_py.montezuma_env as montezuma_env
from goexplore_py.montezuma_env import MyMontezuma
import goexplore_py.pitfall_env as pitfall_env
from goexplore_py.nchain_env import MyNChain
import cProfile
from goexplore_py.policies import *
from tensorflow import summary, ConfigProto, Session, name_scope
from goexplore_py.myUtil import makeHistProto
from itertools import product as itproduct

from diverseExplorer import PPOExplorer_v3 as PPOExplorer, MlshExplorer

VERSION = 1

THRESH_TRUE = 20_000_000_000
THRESH_COMPUTE = 1_000_000
MAX_FRAMES = None
MAX_FRAMES_COMPUTE = None
MAX_ITERATIONS = None
MAX_TIME = 12 * 60 * 60
MAX_LEVEL = None

PROFILER = None

LOG_DIR = None

test_dict = {'log_path': ["log/gridsearch"], 'explorer':['mlsh'], 'game':['montezuma'], 'actors':[1],
			 'nexp':[100, 1000, 2000], 'batch_size':[100], 'resolution': [16],
		'lr': [1.0e-03], 'lr_decay':[ 1],
		'cliprange':[0.1], 'cl_decay': [ 1],
		'n_tr_epochs':[2],
		'mbatch': [4],
		'gamma':[0.99], 'lam':[0.95],
		'nsubs' : [8],
		'timedialation': [20],
		'master_lr': [0.01,  0.04, 0.001],
		'lr_decay_master': [1],
		'master_cl': [0.1],
		'cl_decay_master' :[1],
		'warmup': [ 10, 20, 40],
		'train': [ 20, 40, 80]}
TERM_CONDITION = True
NSAMPLES = 4









def _run(resolution=16, score_objects=True, mean_repeat=20,
		 explorer='repeated',
		 seen_weight=0.0, seen_power=1.0,
		 chosen_weight=0.0, chosen_power=1.0,
		 action_weight=0.0, action_power=1.0,
		 horiz_weight=0.3, vert_weight=0.1,
		 low_score_weight=0.5, high_score_weight=10.0,
		 explore_steps=100, ignore_death=1,
		 x_repeat=2, show=False,
		 seed_path=None, base_path='./results/', clear_old_checkpoints=True,
		 game="montezuma",
		 chosen_since_new_weight=0, chosen_since_new_power=1,
		 warn_delete=True, low_level_weight=0.1,
		 objects_from_pixels=True, objects_remember_rooms=True,
		 only_keys=True, optimize_score=True, use_real_pos=True,
		 target_shape=(6, 6), max_pix_value=255,
		 prob_override=0.0,
		 reset_pool=False, pool_class='py',
		 start_method='fork',
		 path_postfix='',
		 n_cpus=None,
		 save_prob_pictures=False,
		 save_item_pictures=False,
		 keep_prob_pictures=False,
		 keep_item_pictures=False,
		 batch_size=100,
		 reset_cell_on_update=False,
		 actors=1,
		 nexp = None,
		 lr=1.0e-03, lr_decay=0.99999,
		 cliprange=0.1, cl_decay=0.99999,
		 n_tr_epochs=2,
		 mbatch=4,
		 gamma=0.99, lam=0.95,
		 log_path="log",
		 nsubs = 8,
		 timedialation = 20,
		 master_lr = 0.01,
		 lr_decay_master =0.99999,
		 master_cl = 0.1,
		 cl_decay_master =0.99999,
		 warmup= 20,
		 train = 40,
		 retrain_N = None


		 ):

	if game == "robot":
		explorer = RepeatedRandomExplorerRobot()
	elif explorer == "ppo":
		ncpu = multiprocessing.cpu_count()
		if sys.platform == 'darwin': ncpu //= 2
		config = ConfigProto(allow_soft_placement=True,
							 intra_op_parallelism_threads=ncpu,
							 inter_op_parallelism_threads=ncpu)
		config.gpu_options.allow_growth = True  # pylint: disable=E1101
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		sess = Session(config=config).__enter__()
		if nexp is None:
			nexp = explore_steps
		explorer = PPOExplorer(actors=actors, nexp=nexp, lr=lr, lr_decay=lr_decay,
							   cliprange=cliprange, cl_decay=cl_decay, n_tr_epochs=n_tr_epochs,
							   nminibatches=mbatch, gamma=gamma, lam=lam)
		# if game == 'nchain':
		# 	explorer.init_model(env="NChain-v0", policy=MlpPolicy)
		# else:
		# 	explorer.init_model(env="MontezumaRevengeDeterministic-v4", policy=CnnPolicy)
	elif explorer == 'mlsh':
		ncpu = multiprocessing.cpu_count()
		if sys.platform == 'darwin': ncpu //= 2
		config = ConfigProto(allow_soft_placement=True,
							 intra_op_parallelism_threads=ncpu,
							 inter_op_parallelism_threads=ncpu)
		config.gpu_options.allow_growth = True  # pylint: disable=E1101
		os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
		sess = Session(config=config).__enter__()
		if nexp is None:
			nexp = explore_steps
		explorer = MlshExplorer(nsubs=nsubs, timedialation=timedialation, warmup_T=nexp*warmup, train_T=nexp*train,
								actors=actors, nexp=nexp//timedialation, lr_mas=master_lr, lr_sub=lr, lr_decay=lr_decay_master,
								lr_decay_sub=lr_decay, cl_decay=cl_decay_master, cl_decay_sub=cl_decay, n_tr_epochs=n_tr_epochs,
								nminibatches=mbatch, gamma=gamma, lam=lam, cliprange_mas=master_cl, cliprange_sub=cliprange, retrain_N=retrain_N)




	elif explorer == 'repeated':
		explorer = RepeatedRandomExplorer(mean_repeat)
	else:
		explorer = RandomExplorer()

	if game == "montezuma":
		game_class = MyMontezuma
		game_class.TARGET_SHAPE = target_shape
		game_class.MAX_PIX_VALUE = max_pix_value
		game_args = dict(
			score_objects=score_objects, x_repeat=x_repeat,
			objects_from_pixels=objects_from_pixels,
			objects_remember_rooms=objects_remember_rooms,
			only_keys=only_keys,
			unprocessed_state=True
		)
		grid_resolution = (
			GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
			GridDimension('x', resolution), GridDimension('y', resolution)
		)
	elif game == "pitfall":
		game_class = pitfall_env.MyPitfall
		game_class.TARGET_SHAPE = target_shape
		game_class.MAX_PIX_VALUE = max_pix_value
		game_args = dict(score_objects=score_objects, x_repeat=x_repeat)
		grid_resolution = (
			GridDimension('level', 1), GridDimension('score', 1), GridDimension('room', 1),
			GridDimension('x', resolution), GridDimension('y', resolution)
		)
	elif game == "nchain":
		game_class = MyNChain
		game_class.TARGET_SHAPE = target_shape
		game_class.MAX_PIX_VALUE = max_pix_value
		game_args = dict(N=10000)
		grid_resolution = (GridDimension('state', 1),)
	else:
		raise NotImplementedError("Unknown game: " + game)


	if game != "nchain":
		selector = WeightedSelector(game_class,
									seen=Weight(seen_weight, seen_power),
									chosen=Weight(chosen_weight, chosen_power),
									action=Weight(action_weight, action_power),
									room_cells=Weight(0.0),
									dir_weights=DirWeights(horiz_weight, vert_weight, low_score_weight, high_score_weight),
									chosen_since_new_weight=Weight(chosen_since_new_weight, chosen_since_new_power),
									low_level_weight=low_level_weight
									)
	else:
		selector = NChainSelector(game_class,
								  seen=Weight(seen_weight, seen_power),
								  chosen=Weight(chosen_weight, chosen_power),
								  action=Weight(action_weight, action_power),
								  room_cells=Weight(0.0),
								  dir_weights=DirWeights(horiz_weight, vert_weight, low_score_weight, high_score_weight),
								  chosen_since_new_weight=Weight(chosen_since_new_weight, chosen_since_new_power),
								  low_level_weight=low_level_weight, with_domain=use_real_pos
								  )


	pool_cls = multiprocessing.get_context(start_method).Pool
	if pool_class == 'torch':
		pool_cls = torch.multiprocessing.Pool
	elif pool_class == 'loky':
		pool_cls = LPool

	expl = Explore(
		explorer,
		selector,
		(game_class, game_args),
		grid_resolution,
		explore_steps=explore_steps,
		ignore_death=ignore_death,
		optimize_score=optimize_score,
		use_real_pos=use_real_pos,
		prob_override=prob_override,
		reset_pool=reset_pool,
		pool_class=pool_cls,
		n_cpus=n_cpus,
		batch_size=batch_size,
		reset_cell_on_update=reset_cell_on_update
	)

	if seed_path is not None:
		expl.grid = pickle.load(lzma.open(seed_path, 'rb'))
		print(random.sample(list(expl.grid.keys()), 10))
		print('Number at level > 0: ', len([e for e in expl.grid.keys() if e.level > 0]))

	n_digits = 12

	old = 0
	old_compute = 0

	with tqdm(desc='Time (seconds)', smoothing=0, total=MAX_TIME) as t_time, tqdm(desc='Iterations', total=MAX_ITERATIONS) as t_iter, tqdm(desc='Compute steps', total=MAX_FRAMES_COMPUTE) as t_compute, tqdm(desc='Game step', total=MAX_FRAMES) as t:
		start_time = time.time()
		last_time = np.round(start_time)
		# TODO: make this more generic for each level switch
		seen_level_1 = False
		n_iters = 0
		prev_checkpoint = None

		def should_continue():
			if MAX_TIME is not None and time.time() - start_time >= MAX_TIME:
				return False
			if MAX_FRAMES is not None and expl.frames_true + old >= MAX_FRAMES:
				return False
			if MAX_FRAMES_COMPUTE is not None and expl.frames_compute + old_compute>= MAX_FRAMES_COMPUTE:
				return False
			if MAX_ITERATIONS is not None and n_iters >= MAX_ITERATIONS:
				return False
			if MAX_LEVEL is not None and len(Counter(e.level for e in expl.grid).keys()) > MAX_LEVEL:
				return False
			if TERM_CONDITION and False:
				return False
			return True

		logDir = f'{log_path}/{game}_{explorer.__repr__()}/res_{resolution}_explStep_{explore_steps}'f'_cellbatch_{batch_size}'
		if explorer.__repr__() == 'ppo':
			logDir = f'{logDir}_actors_{actors}_exp_{nexp}_lr_{lr}_lrDec_{lr_decay}_cl_{cliprange}_clDec_{cl_decay}' \
				f'_mbatch_{mbatch}_trainEpochs_{n_tr_epochs}_gamma_{gamma}_lam_{lam}'
		if  explorer.__repr__() == 'mlsh':
			logDir = f'{logDir}_subs_{nsubs}_timeadialation_{timedialation}_warmUp_{warmup}_jointrain_{train}_exp_{nexp}' \
				f'_lrMas_{master_lr}_lrDecMas_{lr_decay_master}_clMas_{master_cl}' \
				f'_clDecMas_{cl_decay_master}_lrSub_{lr}_lrDecSub_{lr_decay}_clSub_{cliprange}_clDecSub_{cl_decay}' \
				f'_retrain_{retrain_N}' \
				f'_mbatch_{mbatch}_trainEpochs_{n_tr_epochs}_gamma_{gamma}_lam_{lam}'
		logDir = f'{logDir}_{time.time()}'
		global LOG_DIR
		LOG_DIR= logDir
		summaryWriter = summary.FileWriter(logdir=logDir, flush_secs=20)
		keys_found = []

		while should_continue():
			# Run one iteration
			old += expl.frames_true
			old_compute += expl.frames_compute

			expl.run_cycle()


			t.update(expl.frames_true )#- old)
			t_compute.update(expl.frames_compute )#- old_compute)
			t_iter.update(1)
			cur_time = np.round(time.time())
			t_time.update(int(cur_time - last_time))
			last_time = cur_time
			n_iters += 1


			entry = [summary.Summary.Value(tag='Rooms_Found', simple_value=len(get_env().rooms))]
			entry.append(summary.Summary.Value(tag='Cells', simple_value=len(expl.grid)))
			entry.append(summary.Summary.Value(tag='Top_score', simple_value=max(e.score for e in expl.grid.values())))
			if game != "nchain":
				dist = Counter(e.score for e in expl.real_grid)
				for key in dist.keys():
					if key not in keys_found:
						keys_found.append(key)
				hist = makeHistProto(dist, bins=30, keys=keys_found)
				leveldist = Counter(e.level for e in expl.real_grid)
				histlvl = makeHistProto(leveldist, bins=5)
				entry.append(summary.Summary.Value(tag="Key_dist", histo=hist))
				entry.append(summary.Summary.Value(tag="Level_dist", histo=histlvl))
			entry.append(summary.Summary.Value(tag="Avg traj-len", simple_value=(expl.frames_compute/batch_size)/explore_steps))

			entry.extend(expl.summary)
			summaryWriter.add_summary(summary=summary.Summary(value=entry), global_step=expl.frames_compute + old_compute)
			expl.summary = []

			# In some circumstances (see comments), save a checkpoint and some pictures
			if ((not seen_level_1 and expl.seen_level_1) or  # We have solved level 1
					old == 0 or  # It is the first iteration
					old // THRESH_TRUE != expl.frames_true // THRESH_TRUE or  # We just passed the THRESH_TRUE threshold
					old_compute // THRESH_COMPUTE != expl.frames_compute // THRESH_COMPUTE or  # We just passed the THRESH_COMPUTE threshold
					not should_continue()):  # This is the last iteration

				# Quick bookkeeping, printing update
				seen_level_1 = expl.seen_level_1
				filename = f'{base_path}/{expl.frames_true:0{n_digits}}_{expl.frames_compute:0{n_digits}}'

				tqdm.write(f'Cells at levels: {dict(Counter(e.level for e in expl.real_grid))}')
				tqdm.write(f'Cells at objects: {dict(Counter(e.score for e in expl.real_grid))}')
				tqdm.write(f'Max score: {max(e.score for e in expl.grid.values())}')
				tqdm.write(f'Compute cells: {len(expl.grid)}')

				# Save pictures
				if show or save_item_pictures or save_prob_pictures:
					# Show normal grid
					if show or save_item_pictures:
						get_env().render_with_known(
							list(expl.real_grid), resolution,
							show=False, filename=filename + '.png',
							get_val=lambda x: 1,
							combine_val=lambda x, y: x + y
						)

					if not use_real_pos:
						object_combinations = sorted(set(e.real_cell.score for e in expl.grid.values() if e.real_cell is not None))
						for obj in object_combinations:
							grid_at_obj = [e.real_cell for e in expl.grid.values() if e.real_cell is not None and e.real_cell.score == obj]
							get_env().render_with_known(
								grid_at_obj, resolution,
								show=False, filename=filename + f'_object_{obj}.png',
								get_val=lambda x: 1,
								combine_val=lambda x, y: x + y
							)

					# Show probability grid
					if (use_real_pos and show) or save_prob_pictures:
						expl.selector.set_ranges(list(expl.grid.keys()))
						possible_scores = sorted(set(e.score for e in expl.grid))
						total = np.sum(
							[expl.selector.get_weight(x, expl.grid[x], possible_scores, expl.grid) for x in expl.grid])
						get_env().render_with_known(
							list(expl.grid.keys()), resolution,
							show=False, filename=filename + '_prob.PNG',
							combine_val=lambda x, y: x + y,
							get_val=lambda x: expl.selector.get_weight(x, expl.grid[x], possible_scores,
																	   expl.grid) / total,
						)
					if prev_checkpoint and clear_old_checkpoints:
						if not keep_item_pictures:
							try:
								os.remove(prev_checkpoint + '.png')
							except FileNotFoundError:
								# If it doesn't exists, we don't need to remove it.
								pass
						if use_real_pos and not keep_prob_pictures:
							try:
								os.remove(prev_checkpoint + '_prob.PNG')
							except FileNotFoundError:
								# If it doesn't exists, we don't need to remove it.
								pass

				with open(filename + ".csv", 'w') as f:
					f.write(str(len(expl.grid)))
					f.write(", ")
					f.write(str(max([a.score for a in expl.grid.values()])))
					f.write("\n")

				# Save checkpoints
				grid_copy = {}
				for k, v in expl.grid.items():
					grid_copy[k] = v
				# TODO: is 7z still necessary now that there are other ways to reduce space?
				pickle.dump(grid_copy, lzma.open(filename + '.7z', 'wb', preset=0))

				# Clean up previous checkpoint.
				if prev_checkpoint and clear_old_checkpoints:
					os.remove(prev_checkpoint + '.7z')
				prev_checkpoint = filename

				# A much smaller file that should be sufficient for view folder, but not for restoring
				# the demonstrations. Should make view folder much faster.
				grid_set = {}
				for k, v in expl.grid.items():
					grid_set[k] = v.score
				pickle.dump(grid_set, lzma.open(filename + '_set.7z', 'wb', preset=0))
				pickle.dump(expl.real_grid, lzma.open(filename + '_set_real.7z', 'wb', preset=0))

				if PROFILER:
					print("ITERATION:", n_iters)
					PROFILER.disable()
					PROFILER.dump_stats(filename + '.stats')
					# PROFILER.print_stats()
					PROFILER.enable()
				# Save a bit of memory by freeing our copies.
				grid_copy = None
				grid_set = None
		# TODO Insert model save here
		#print(expl.explorer.__repr__())
		if expl.explorer.__repr__() == 'ppo':
			sess.__exit__(None, None, None)
			tf.reset_default_graph()
		else:
			print('did not clear graph')


def run(base_path, **kwargs):
	cur_id = 0
	if os.path.exists(base_path):
		current = glob.glob(base_path + '/*')
		for c in current:
			try:
				idx, _ = c.split('/')[-1].split('_')
				idx = int(idx)
				if idx >= cur_id:
					cur_id = idx + 1
			except Exception:
				pass

	base_path = f'{base_path}/{cur_id:04d}_{uuid.uuid4().hex}/'
	os.makedirs(base_path, exist_ok=True)
	info = copy.copy(kwargs)
	info['version'] = VERSION
	info['code_hash'] = get_code_hash()
	print('Code hash:', info['code_hash'])
	json.dump(info, open(base_path + '/kwargs.json', 'w'))

	print('Experiment running in', base_path)

	_run(base_path=base_path, **kwargs)
	if LOG_DIR is not None:
		json.dump(info, open(LOG_DIR + '/kwargs.json', 'w'))


if __name__ == '__main__':
	if platform == "darwin":
		# Circumvents the following issue on Mac OS:
		# https://github.com/opencv/opencv/issues/5150
		cv2.setNumThreads(0)
	parser = argparse.ArgumentParser()

	parser.add_argument('--resolution', '--res', type=float, default=16, help='Length of the side of a grid cell.')
	parser.add_argument('--explorer', '--expl', type=str, default='mlsh',
						help='The explorer to use when searching for solution')
	parser.add_argument('--use_scores', dest='use_objects', action='store_false', help='Use scores in the cell description. Otherwise objects will be used.')
	parser.add_argument('--repeat_action', '--ra', type=int, default=20, help='The average number of times that actions will be repeated in the exploration phase.')
	parser.add_argument('--explore_steps', type=int, default=100, help='Maximum number of steps in the explore phase.')
	parser.add_argument('--ignore_death', type=int, default=1, help='Number of steps immediately before death to ignore.')
	parser.add_argument('--base_path', '-p', type=str, default='./results/', help='Folder in which to store results')
	parser.add_argument('--path_postfix', '--pf', type=str, default='', help='String appended to the base path.')
	parser.add_argument('--seed_path', type=str, default=None, help='Path from which to load existing results.')
	parser.add_argument('--x_repeat', type=int, default=2, help='How much to duplicate pixels along the x direction. 2 is closer to how the games were meant to be played, but 1 is the original emulator resolution. NOTE: affects the behavior of GoExplore.')
	parser.add_argument('--log_path', type=str, default='log', help='Default = log')

	parser.add_argument('--seen_weight', '--sw', type=float, default=0.0, help='The weight of the "seen" attribute in cell selection.')
	parser.add_argument('--seen_power', '--sp', type=float, default=0.5, help='The power of the "seen" attribute in cell selection.')
	parser.add_argument('--chosen_weight', '--cw', type=float, default=0.0, help='The weight of the "chosen" attribute in cell selection.')
	parser.add_argument('--chosen_power', '--cp', type=float, default=0.5, help='The power of the "chosen" attribute in cell selection.')
	parser.add_argument('--chosen_since_new_weight', '--csnw', type=float, default=0.0, help='The weight of the "chosen since new" attribute in cell selection.')
	parser.add_argument('--chosen_since_new_power', '--csnp', type=float, default=1.0, help='The power of the "chosen since new" attribute in cell selection.')
	parser.add_argument('--action_weight', '--aw', type=float, default=0.0, help='The weight of the "action" attribute in cell selection.')
	parser.add_argument('--action_power', '--ap', type=float, default=0.5, help='The power of the "action" attribute in cell selection.')
	parser.add_argument('--horiz_weight', '--hw', type=float, default=0.2, help='Weight of not having one of the two possible horizontal neighbors.')
	parser.add_argument('--vert_weight', '--vw', type=float, default=0.0, help='Weight of not having one of the two possible vertical neighbors.')
	parser.add_argument('--low_score_weight', type=float, default=0.0, help='Weight of not having a neighbor with a lower score/object number.')
	parser.add_argument('--high_score_weight', type=float, default=0.5, help='Weight of not having a neighbor with a higher score/object number.')

	parser.add_argument('--end_on_death', dest='end_on_death', action='store_true', help='End episode on death.')

	parser.add_argument('--low_level_weight', type=float, default=0.1, help='Weight of cells in levels lower than the current max. If this is non-zero, lower levels will keep getting optimized, potentially leading to better solutions overall. Setting this to greater than 1 is possible but nonsensical since it means putting a larger weight on low levels than higher levels.')

	parser.add_argument('--max_game_steps', type=int, default=None, help='Maximum number of GAME frames.')
	parser.add_argument('--max_compute_steps', '--mcs', type=int, default=None, help='Maximum number of COMPUTE frames.')
	parser.add_argument('--max_iterations', type=int, default=None, help='Maximum number of iterations.')
	parser.add_argument('--max_hours', '--mh', type=float, default=12, help='Maximum number of hours to run this for.')
	parser.add_argument('--checkpoint_game', type=int, default=20_000_000_000_000, help='Save a checkpoint every this many GAME frames (note: recommmended to ignore, since this grows very fast at the end).')
	parser.add_argument('--checkpoint_compute', type=int, default=1_000_000, help='Save a checkpoint every this many COMPUTE frames.')
	parser.add_argument('--max_level', type=int, default=None, help='Set max number of levels to complete. Complete means seen the level after')

	parser.add_argument('--pictures', dest='save_pictures', action='store_true', help='Save pictures of the pyramid every checkpoint (uses more space).')
	parser.add_argument('--prob_pictures', '--pp', dest='save_prob_pictures', action='store_true',
						help='Save pictures of showing probabilities.')
	parser.add_argument('--item_pictures', '--ip', dest='save_item_pictures', action='store_true',
						help='Save pictures of showing items collected.')
	parser.add_argument('--keep_checkpoints', dest='clear_old_checkpoints', action='store_false', help='Keep all checkpoints in large format. This isn\'t necessary for view folder to work. Uses a lot of space.')
	parser.add_argument('--keep_prob_pictures', '--kpp', dest='keep_prob_pictures', action='store_true',
						help='Keep old pictures showing probabilities.')
	parser.add_argument('--keep_item_pictures', '--kip', dest='keep_item_pictures', action='store_true',
						help='Keep old pictures showing items collected.')
	parser.add_argument('--no_warn_delete', dest='warn_delete', action='store_false', help='Do not warn before deleting the existing directory, if any.')
	parser.add_argument('--game', '-g', type=str, default="montezuma", help='Determines the game to which apply goexplore.')

	parser.add_argument('--objects_from_ram', dest='objects_from_pixels', action='store_false', help='Get the objects from RAM instead of pixels.')
	parser.add_argument('--all_objects', dest='only_keys', action='store_false', help='Use all objects in the state instead of just the keys')
	parser.add_argument('--remember_rooms', dest='remember_rooms', action='store_true', help='Remember which room the objects picked up came from. Makes it easier to solve the game (because the state encodes the location of the remaining keys anymore), but takes more time/memory space, which in practice makes it worse quite often. Using this is better if running with --no_optimize_score')
	parser.add_argument('--no_optimize_score', dest='optimize_score', action='store_false', help='Don\'t optimize for score (only speed). Will use fewer "game frames" and come up with faster trajectories with lower scores. If not combined with --remember_rooms and --objects_from_ram is not enabled, things should run much slower.')
	parser.add_argument('--prob_override', type=float, default=0.0, help='Probability that the newly found cells will randomly replace the current cell.')

	parser.add_argument('--resize_x', '--rx', type=int, default=11, help='What to resize the pixels to in the x direction for use as a state.')
	parser.add_argument('--resize_y', '--ry', type=int, default=8, help='What to resize the pixels to in the y direction for use as a state.')
	parser.add_argument('--state_is_pixels', '--pix', dest='state_is_pixels', action='store_true', help='If this is on, the state will be resized pixels, not human prior.')
	parser.add_argument('--max_pix_value', '--mpv', type=int, default=8, help='The range of pixel values when resizing will be rescaled to from 0 to this value. Lower means fewer possible states in states_is_pixels.')
	parser.add_argument('--n_cpus', type=int, default=None, help='Number of worker threads to spawn')
	parser.add_argument('--batch_size', type=int, default=100, help='Number of worker threads to spawn')

	parser.add_argument('--pool_class', type=str, default='loky', help='The multiprocessing pool class (py or torch or loky).')
	parser.add_argument('--start_method', type=str, default='fork', help='The process start method.')
	parser.add_argument('--reset_pool', dest='reset_pool', action='store_true', help='The pool should be reset every 100 iterations.')
	parser.add_argument('--reset_cell_on_update', '--rcou', dest='reset_cell_on_update', action='store_true',
						help='Reset the times-chosen and times-chosen-since when a cell is updated.')
	parser.add_argument('--profile', dest='profile', action='store_true',
						help='Whether or not to enable a profiler.')
	parser.add_argument("--run_test", dest='test_run', action='store_true')

	parser.set_defaults(save_pictures=False, use_objects=True, clear_old_checkpoints=True,
						warn_delete=True, objects_from_pixels=True, only_keys=True, remember_rooms=False,
						optimize_score=True, state_is_pixels=False, reset_pool=False, end_on_death=False, test_run=False)

	args = parser.parse_args()

	if args.start_method == 'fork' and args.pool_class == 'torch':
		raise Exception('Fork start method not supported by torch.multiprocessing.')
	if args.start_method != 'fork' and args.pool_class == 'loky':
		raise Exception('Loky only supports the fork start method.')

	THRESH_TRUE = args.checkpoint_game
	THRESH_COMPUTE = args.checkpoint_compute
	MAX_FRAMES = args.max_game_steps
	MAX_FRAMES_COMPUTE = args.max_compute_steps
	MAX_TIME = args.max_hours * 3600
	MAX_ITERATIONS = args.max_iterations
	MAX_LEVEL = args.max_level
	TERM_CONDITION = args.test_run

	if args.profile:
		PROFILER = cProfile.Profile()
		PROFILER.enable()
	try:
		if not args.test_run:
			run(resolution=args.resolution, score_objects=args.use_objects, explorer=args.explorer,
				mean_repeat=args.repeat_action, explore_steps=args.explore_steps, ignore_death=args.ignore_death,
				base_path=args.base_path, seed_path=args.seed_path, x_repeat=args.x_repeat, seen_weight=args.seen_weight,
				seen_power=args.seen_power, chosen_weight=args.chosen_weight, chosen_power=args.chosen_power,
				action_weight=args.action_weight, action_power=args.action_power, horiz_weight=args.horiz_weight,
				vert_weight=args.vert_weight, low_score_weight=args.low_score_weight, high_score_weight=args.high_score_weight,
				show=args.save_pictures, clear_old_checkpoints=args.clear_old_checkpoints, warn_delete=args.warn_delete,
				chosen_since_new_weight=args.chosen_since_new_weight, chosen_since_new_power=args.chosen_since_new_power,
				game=args.game, low_level_weight=args.low_level_weight, objects_from_pixels=args.objects_from_pixels,
				only_keys=args.only_keys, objects_remember_rooms=args.remember_rooms, optimize_score=args.optimize_score,
				use_real_pos=not args.state_is_pixels, target_shape=(args.resize_x, args.resize_y),
				max_pix_value=args.max_pix_value,
				prob_override=args.prob_override,
				reset_pool=args.reset_pool,
				pool_class=args.pool_class,
				start_method=args.start_method,
				path_postfix=args.path_postfix,
				n_cpus=args.n_cpus,
				save_prob_pictures=args.save_prob_pictures,
				save_item_pictures=args.save_item_pictures,
				keep_prob_pictures=args.keep_prob_pictures,
				keep_item_pictures=args.keep_item_pictures,
				batch_size=args.batch_size,
				reset_cell_on_update=args.reset_cell_on_update,
				log_path=args.log_path)
		else:
			keys, values = zip(*test_dict.items())
			for _ in range(NSAMPLES):
				for v in itproduct(*values):
					run( base_path='./results/', **dict(zip(keys, v))) #Run experiment with permutation of values from test_dict
		if PROFILER is not None:
			PROFILER.disable()
	finally:
		if PROFILER is not None:
			PROFILER.print_stats()
