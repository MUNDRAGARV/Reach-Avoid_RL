from gym.envs.registration import register

register(
    id="dubins_car-v1",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarOneEnv"
)

register(
    id="dubins_car_pe-v0",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarPEEnv"
)

register(
    id="dubins_car_novel-v1",
    entry_point="gym_reachability.gym_reachability.envs:DubinsCarPEEnv_novel"
)

