from gym.envs.registration import register



register(
    id='Circle-v0',
    entry_point='driftgym.envs:CircleEnv',
)


register(
    id='Drift-v0',
    entry_point='driftgym.envs:FastCircleEnv',
)


register(
    id='Straight-v0',
    entry_point='driftgym.envs:StraightEnv',
)