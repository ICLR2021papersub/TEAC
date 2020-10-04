from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
import spinup.algos.teac.core as core
from spinup.utils.logx import EpochLogger
from torch.distributions.uniform import Uniform


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TEAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        # def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32).to(self.device) for k, v in batch.items()}


def teac(env_fn, load, env_name, exp_name,
         actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
         max_divergence=0.005, alpha=0.2, beta=0.01, const_dual=False,
         polyak=0.995, lr=1e-3, lr_alpha=1e-3, lr_beta=1e-4, batch_size=100, start_steps=10000,
         update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000,
         logger_kwargs=dict(), save_freq=1):
    """
    Trust-Entropy Actor-Critic (TEAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        max_divergence (float): Maximum divergence for KL constraint. (varies from 
            different tasks. For now, we found that the best value for tasks are:
                ===============            =======  =================================
                task                       value    action_space
                ===============            =======  =================================
                HalfCheetah-v3             0.5      6
                Humanoid-v3                0.001    17
                Swimmer-v3                 0.005    2
                Ant-v3                     0.05     8
                Hopper-v3                  0.1      3
                Walker2d-v3                0.1      6
                ===============            =======  =================================
            for a fair comparision with baselines, we set max_divergence=0.005 for all tasks.
            )

        alpha (float): Dual variable w.r.t entropy constraint. (Only work when 
        const_dual=True)

        beta (float): Dual variable w.r.t KL-divergence constraint. (Only work when 
        const_dual=True)

        const_dual (bool): if const_dual=True, we use constant dual variables 
                            else we use gradient descent to change the value automatically.

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        lr_alpha (float): Learning rate (used for alpha).

        lr_beta (float): Learning rate (used for beta).

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    # device = 'cpu'
    print(device)
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space,
                      env.action_space, **ac_kwargs).to(device)
    ac_targ = deepcopy(ac).to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(
        obs_dim=obs_dim, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module)
                       for module in [ac.pi, ac.q1, ac.q2])
    logger.log(
        '\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)

    def heuristic_target_entropy(action_space):
        heuristic_target_entropy = -torch.prod(torch.Tensor(action_space.shape)).to(
            device).item()
        return heuristic_target_entropy

    _max_divergence = max_divergence
    if const_dual == True:
        _alpha = alpha
        _beta = beta
    else:
        _log_alpha = torch.zeros(
            1, requires_grad=True, device=device)
        _alpha = _log_alpha.exp()

        _target_entropy = heuristic_target_entropy(env.action_space)
        _log_beta = torch.zeros(1, requires_grad=True, device=device)
        _beta = _log_beta.exp()

    def compute_loss_q(data, alpha, beta):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        o, a, r, o2, d = o.to(device), a.to(device), r.to(
            device), o2.to(device), d.to(device)
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy and *old* policy
            a2, logp_a2 = ac.pi(o2)
            _, logp_a2_old = ac.pi_old(o2)
            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            # See Sec.3.2
            backup = r + gamma * (1 - d) * (q_pi_targ -
                                            (alpha+beta) * logp_a2 + alpha * logp_a2_old)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                      Q2Vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing TEAC pi loss
    def compute_loss_pi(data, alpha, beta):
        o = data['obs'].to(device)
        # Target actions come from *current* policy and *old* policy
        pi, logp_pi = ac.pi(o)
        _, logp_pi_old = ac.pi_old(o)
        # Q-values
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        # See Sec.3.3
        loss_pi = ((alpha+beta)*logp_pi -
                   (alpha*logp_pi_old.detach() + q_pi)).mean()
        loss_pi = ((alpha+beta)*logp_pi -
                   q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().cpu().numpy())

        return loss_pi, pi_info

    # See Sec.3.4
    def compute_loss_alpha(data, alpha, beta, log_alpha, max_divergence):
        o = data['obs'].to(device)

        pi, logp_pi = ac.pi(o)
        pi_old, logp_pi_old = ac.pi_old(o)
        pi_kl = logp_pi - logp_pi_old
        loss_alpha = -1.0 * (
            log_alpha * (logp_pi - logp_pi_old - max_divergence).detach()).mean()
        alpha_info = dict(alpha=alpha.detach().cpu().numpy(),
                          pi_kl=pi_kl.detach().cpu().numpy()
                          )

        return loss_alpha, alpha_info

    # See Sec.3.4
    def compute_loss_beta(data, alpha, beta, log_beta, target_entropy):
        o = data['obs'].to(device)
        pi, logp_pi = ac.pi(o)

        loss_beta = -1.0 * (
            log_beta * (logp_pi + target_entropy).detach()).mean()

        beta_info = dict(beta=beta.detach().cpu().numpy(),
                         logp_pi=logp_pi.detach().cpu().numpy()
                         )

        return loss_beta, beta_info

    # Set up optimizers for policy, q-function, alpha and beta
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    if not const_dual:
        alpha_optimizer = Adam([_log_alpha], lr=lr_alpha)
        beta_optimizer = Adam([_log_beta], lr=lr_beta)
    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, alpha, beta,
               log_alpha, max_divergence,
               log_beta, target_entropy):
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data, alpha, beta)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data, alpha, beta)
        # IMPORTANT: the copy opt should be done here
        with torch.no_grad():
            for old_param, new_param in zip(ac.pi_old.parameters(), ac.pi.parameters()):
                old_param.data.copy_(new_param.data)
        loss_pi.backward(retain_graph=True)
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)
        if not const_dual:
            logger.store(alpha=alpha.item())
            logger.store(beta=beta.item())

        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

        if not const_dual:
            alpha_optimizer.zero_grad()
            loss_alpha, alpha_info = compute_loss_alpha(data, alpha, beta,
                                                        log_alpha, max_divergence)
            loss_alpha.backward(retain_graph=True)
            alpha_optimizer.step()
            alpha = log_alpha.exp()

            beta_optimizer.zero_grad()
            loss_beta, beta_info = compute_loss_beta(data, alpha, beta,
                                                     log_beta, target_entropy)
            loss_beta.backward(retain_graph=True)
            beta_optimizer.step()
            beta = log_beta.exp()

        return alpha, log_alpha, beta, log_beta

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32).to(device),
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time
                t_a, logp = get_action(o, True)
                o, r, d, _ = test_env.step(t_a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy.
        if t > start_steps:
            a, _ = get_action(o)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)
        # replay_buffer.store(o, a, r, o2, d)
        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                _alpha, _log_alpha, _beta, _log_beta = update(data=batch, alpha=_alpha, beta=_beta,
                                                              log_alpha=_log_alpha, max_divergence=_max_divergence,
                                                              log_beta=_log_beta, target_entropy=_target_entropy)
        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossAlpha', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('logp_pi', average_only=True)
            # logger.log_tabular('_log_alpha', average_only=True)
            # logger.log_tabular('_log_beta', average_only=True)
            # logger.log_tabular('KL_divergence', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            if not const_dual:
                logger.log_tabular('alpha', average_only=True)
                logger.log_tabular('beta', average_only=True)
                logger.log_tabular('pi_kl', average_only=True)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=750)
    parser.add_argument('--exp_name', type=str, default='teac')
    parser.add_argument('--md', type=float, default=0.005)
    parser.add_argument('--lr_alpha', type=float, default=0.0001)
    parser.add_argument('--lr_beta', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, args.env)

    torch.set_num_threads(1)

    teac(lambda: gym.make(args.env), args.load, args.env, args.exp_name,
         actor_critic=core.MLPActorCritic,
         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l),
         gamma=args.gamma, max_divergence=args.md, alpha=0.2, beta=0.01, const_dual=False,
         seed=args.seed, epochs=args.epochs,
         lr_alpha=args.lr_alpha, lr_beta=args.lr_beta, batch_size=args.batch_size,
         logger_kwargs=logger_kwargs)
