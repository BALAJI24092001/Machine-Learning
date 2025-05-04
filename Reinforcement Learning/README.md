# REINFORCEMENT LEARNING

The core idea of machine learning comes from the desire to make a computer learn. Deceving from the sections of Supervised Learning and Un-supervised learning where the machine learns from data which has the infomation on what kind of pattern gives what kind of results, or having to segregate the data based on it's simillarities, thereby learning or identifying a pattern in the data respectively defines the given learning paradigms. Unlike these two learning paradigms, Reinforcement Learning doesn't rely on data to learn. Essentially in Supervised learning, we're trying to minimize an entropy function, which is defined on the prediction of a particular object as opposed to it's original state and try to optimize the disorderliness in predition of the model by minimizing this entroypy. In Unsupervised learning, given a data without any labels, we consider the whole system of unlabelled data and make an entropy function to label the data, such that simillar the observation, more likely it belongs to the same label and hence minimizing this entropy results in desirable or more approprioate clustering of the data.

In Reinforcement Learning, we have 4 core components through which we try to learn.[^Prof_Balaraman_RL][^David_Silver_RL_Lec]

1. Policy: It gives the agent's way of behaving at a given time, ensentially a mapping from perceived states to actions.
2. Reward: This defines the goal of the problem, providing feedback to the agent about the goodness or badness of its actions. The agent's objective is to maximize the total reward received over time.
3. Value Fucntion: This **estimates** the "goodness" of being in a particular state, or the "goodness" of taking a specific action in a specific state, in terms of the expected future reward.
4. Model

Reinforcement learning relies heavily on the concept of state—as input to the policy and value function, and as both input to and output from the model. Informally, we can think of the state as a signal conveying to the agent some sense of “how the environment is” at a particular time.

<!--
TODO:
    1. Learn Markov Chains and Markov Decision Processes from Szepesvári
    2. Re-read the first chapter of Sutton and Barto
-->

**Some History on RL**

The early history of reinforcement learning has **two main threads** that were pursued independently for a long time before coming together, along with a **third, less distinct thread**. According to the sources, these three threads are:[^Sutton_Barto_RL_Book][^Szepesvári_RL_Book][^Dimitri_RL_Book]

- The thread concerning **learning by trial and error**, which originated in the psychology of animal learning. This thread is traced through early work in artificial intelligence and its revival in the early 1980s.
- The thread concerning the problem of **optimal control** and its solution using value functions and dynamic programming. Initially, this thread did not primarily involve learning.
- A **third, less distinct thread concerning temporal-difference methods**, such as the one used in the tic-tac-toe example mentioned earlier in the book. This thread is considered smaller but has played a particularly important role.

All three of these threads came together in the late 1980s to form the modern field of reinforcement learning as it is presented in the book. The source notes that it is most familiar with and has the most to say about the trial-and-error learning thread in its brief history.

## Reinforcement Learning from Human Feedback(RHFL)

Refer to Nathan Lambert RHFL book.[^Nathan_RLHF]

## References

[^Sutton_Barto_RL_Book]: Richard S. Sutton and Andrew G. Barto. 2018. **Reinforcement Learning: An Introduction**. A Bradford Book, Cambridge, MA, USA.

[^Szepesvári_RL_Book]: Szepesvári, C. (2010). **Algorithms for Reinforcement Learning. In Synthesis Lectures on Artificial Intelligence and Machine Learning**. Springer International Publishing.

[^Dimitri_RL_Book]: Dimitri P. Bertsekas and Steven E. Shreve. 1978. **Stochastic Optimal Control: The Discrete-Time Case**. Academic Press, Inc., USA.

[^Nathan_RLHF]: Lambert, N. (2024). **Reinforcement Learning from Human Feedback**. Online: https://rlhfbook.com/

[^Prof_Balaraman_RL]: Prof. Balaraman Ravindran, [Introduction to Reinforcement Learning](https://www.youtube.com/playlist?list=PLEAYkSg4uSQ0Hkv_1LHlJtC_wqwVu6RQX), IITM Lecture Series.

[^David_Silver_RL_Lec]: David Silver's [Reinforcement Learning Course](https://www.youtube.com/playlist?list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ), DeepMind.
