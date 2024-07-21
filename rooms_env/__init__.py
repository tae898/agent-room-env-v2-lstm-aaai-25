from gymnasium.envs.registration import register


register(
    id="RoomsEnv",
    entry_point="rooms_env.envs:RoomsEnv",
)
