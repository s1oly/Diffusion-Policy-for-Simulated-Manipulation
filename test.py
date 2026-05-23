import robosuite as suite 


def main():
    env = suite.make(
        env_name = "NutAssemblySquare",
        robots = "Panda",
        has_renderer = True, 
        has_offscreen_renderer = False, 
        use_camera_obs = False
    )

    obs = env.reset()

    while True:
        action = env.action_spec[0]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
    # env.close()

if __name__ == '__main__':
    main()