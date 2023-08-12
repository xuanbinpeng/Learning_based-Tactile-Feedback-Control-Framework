import time
import os
import torch
import numpy as np
import h5py
from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise

collect_actions = False
collect_obs = True
action_replay = False
obs_replay = True
ar_dir = './actions/straight/action.npy'
ac_dir = './actions/straight/action.npy'
or_dir = './obs/test.npy'
oc_dir = './obs/test.npy'
tactile_data_dir = './data/straight'


def demo_rl_env(env, num_iter, action_ids, show_gui, show_tactile, render, print_info=False):
    """
    Control loop for demonstrating an RL environment.
    Use show_gui and show_tactile flags for visualising and controlling the env.
    Use render for more direct info on what the agent will see.
    """
    record = False
    if record:
        import imageio

        render_frames = []
        log_id = env._pb.startStateLogging(
            loggingType=env._pb.STATE_LOGGING_VIDEO_MP4, fileName=os.path.join("example_videos", "gui.mp4")
        )

    # collection loop #TODO
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]), sigma=0.1 * np.ones(env.action_space.shape[-1])) 
    policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])

    #policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[512, 256, 128])
    model = TQC("MultiInputPolicy", env,learning_rate=1e-3, batch_size=256,
               tau=0.001, gamma=0.95, action_noise=action_noise, buffer_size=int(1e6), train_freq=(5, 'step'),
               learning_starts=10000, policy_kwargs=policy_kwargs, device=device,tensorboard_log="../tensorboard_logs/",verbose=1)
    model = TQC.load("/home/nathan/tactile_gym/model/tactile_vae.zip", env)

    action_replays = np.load(ar_dir)
    observation_replays = np.load('/home/nathan/tactile_gym/examples/obs_replay.npy')
    for i in range(num_iter):
        r_sum = 0
        o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        step = 0
        f = h5py.File(tactile_data_dir+str(i)+'.h5', 'w')
        action_list = []
        obs_list = []
        while not d:
            
            if show_gui:
                a = []
                for action_id in action_ids:
                    a.append(env._pb.readUserDebugParameter(action_id))
            else:
                action,_ = model.predict(observation=o)
            #print(o[32],o[33],o[37],step)
            action,_ = model.predict(observation=o)
            # step the environment
            #print(a)
        
            action_list.append(a)
            obs_list.append(o)
            if action_replay :
                o, r, d, info = env.step(action_replays[step])
                print(action_replays[step])
                #print(o)
            elif obs_replay:
                #print(action)
                _, r, d, info = env.step(action)
                o = observation_replays[step]
                print(o[33])
            elif collect_actions:
                
                o, r, d, info = env.step(a)
            else:
                o, r, d, info = env.step(action)
            #if step == 249:
            #    d = True
            #print(o)
            
            f.create_dataset('data_'+str(step), data=env.current_img)
            if print_info:
                print("")
                print("Step: ", step)
                print("Act:  ", a)
                print("Obs:  ")
                for key, value in o.items():
                    if value is None:
                        print("  ", key, ":", value)
                    else:
                        print("  ", key, ":", value.shape)
                print("Rew:  ", r)
                print("Done: ", d)

            # render visual + tactile observation
            if render:
                render_img = env.render()
                if record:
                    render_frames.append(render_img)

            r_sum += r
            step += 1

            q_key = ord("q")
            r_key = ord("r")
            keys = env._pb.getKeyboardEvents()
            if q_key in keys and keys[q_key] & env._pb.KEY_WAS_TRIGGERED:
                exit()
            elif r_key in keys and keys[r_key] & env._pb.KEY_WAS_TRIGGERED:
                d = True
                if collect_actions:
                    np.save(ac_dir,action_list)
                if collect_obs:
                    np.save(oc_dir,obs_list)


        print("Total Reward: ", r_sum)

    if record:
        env._pb.stopStateLogging(log_id)
        imageio.mimwrite(os.path.join("example_videos", "render.mp4"), np.stack(render_frames), fps=12)

    env.close()
