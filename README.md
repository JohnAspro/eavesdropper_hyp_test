# **Evasive Hypothesis Testing**

### Summary of the publication

Hypothesis testing in general is the study of problems where, given a set of data we try to find 
a hypothesis that corresponds optimally to the given data. In this particular example we assume 
that we are granted access to 3 sensors with binary output. We can access one of them at every time
step of the process, each sensor follows a bernoulli distribution with fixed parameters that were
initiated at the start of the simulation. Its assumed that either none or one of the 3 sensors might 
be behaving abnormal, meaning that it's a lot less likely to output '1' or p < 0.5 for that particular 
sensor. The problem set we repressented has a fixed horizon. The goal of the above problem is to find as 
quicly as possible the abnormal sensor, which has been researched thourougly and Deep-RL has given 
impressive results. Now we will introduse an evasive or eavesdropper into the problem, which means 
that someone knows and follows your actions (what sensor you see at each time step) but with different 
probalities for his observation. We call legitimate the agent which is trying to maximize his belief 
for some hypothesis and minimize the belief for the eavesdropper. The output that legitimate receives
is noted by y and the eavesdropper is noted z. We use two 4-dimentional belief vectors for the four
hypothesis and the two agents, they are initialized with the prior probabilities, in our case 1/4 since 
we have 3 sensors and 4 hypothesis as stated above. After every timestep a feedback from some sensor we 
choosed was given and the belief vector is updated by bayesian update rule. We feed the above vector to 
some Deep-RL algorithms and compare the results with the well known Chernoff test. As a reward we used 
r = b * AEP - a * LEP , where b and a are some constant weights to 
Adversary Error probability = (1 - maximum of the adversary belief vector) and 
Legitimate error probabily = (1 - maximum of the legitimate belief vector) respectivly. 
In the training part this reward was fed back at every timestep and in the testing part it was given only 
at the end of the episode.
