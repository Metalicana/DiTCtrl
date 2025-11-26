## Expectation of a Random Variable.
First of all, a random variable x can be anything.
I wanna say, random variable is wind speed today.

$E_x[f(x)] = \int{xf(x)dx}$

This formula is interesting. $f(x)$ is probability density function.
$\int{f(x)dx}$ is gonna be 1. Sum of the probability of all possible values of x

For each probability, multiplying with the value itself gives the expected value.

For example, expected value of a dice throw is the value $1*1/6 + 2*1/6 + 3*1/6 4*1/6 5*1/6 + 6*1/6 = 3.5$ 

## Chain Rule Of Probability
$p(x, y) = p(x | y)p(y)$
Let's say x is wind speed, and y is rain amount in mm.
then $p(x, y)$ is probability of windspeed x and rain y mm.
Which is equal to, probability of x given y has happened times probability of y happening.
Let's say x = 120mph and y = 5mm
p(120, 5) = p(120|5) x p(5)

### Bayes' Theorem
$$p(x | y) = \frac{p(y | x)p(x)}{p(y)}$$

This is simply symmetry of the chain rule. But for intuition, the p(y) here is all the possibilities where y has happened, and p(y|x)p(x) is all the possibilities where x and y happened. 

### K-L Divergence 

The distance between two probability distributions.
Surprise = $1/p(x)$ means, less likely an event more surprising it is.
If the surprising events happens 5 times, it is 5 times surprising!
$I(x) = log\frac{1}{p(x)}$
Log because assume events are independent, then probabilities multiply.

What is the average surprise or Expected value:
$$H(P) = P1*log(1/P1) + P2*log(1/P2) + ... Pn*log(1/Pn)$$
This is not expected value exactly, rather it's the average surprise! 
Because we are considering $log(1/p)$ as the surprise.

Now, Let's say there are two distributions, P and Q, for the same thing.
Then new formula $$H(P, Q) = P1*log(1/Q1) + P2*log(1/Q2) + ... + Pn*log(1/Qn)$$
Think about it, let's say you wanna use the distribution Q for probabilistic modeling. So if Q1 happens, surprise is 1/Q1. But if you multiply it with Q1, but this surprise's probability is not Q1 in reality. It is P1.

If you consider log as 2 based, we can measure how many bits it takes to encode the information. Oh god I finally understand this.

H(P) gonna be number of bits needed to encode real probability. H(P, Q) is gonna be number of bits needed to encode fake probability. So the difference is the KL Divergence.

$$
D_{KL}(P||Q) = H(P, Q) - H(P)
$$
$$
D_{KL}(P||Q) = \sum P(x)*log(1/Q(x)) - \sum P(x) * log(1/P(x))

$$

$$
D_{KL}(P||Q) =  \sum P(x)*log(\frac{P(x)}{Q(x)})
$$
if instead of entirely doing it, we let the probability itself sample x, so the multiplication term can be ignored. Basically, we do sampling. Also, if Q(x) is bigger than P(x), then the log ratio is gonna be negative. We want KL-divergence to always be non-negative. Also, think about it, it can be solved by square and square rooting together, root can be put outside.
$$
D_{KL}(P||Q) =  \sum P(x)*\frac{1}{2}log(\frac{P(x)}{Q(x)})^2
$$
Important intuition here, squaring means, despite having low variance, it is highly biased. Meaning, if two distributions are quite close to each other, then estimation is good.

To make it low bias and low variance, we add a term that is actually 0.
$$
D^{'}_{KL}(P||Q) = \frac{1}{N} \sum{[log \frac{P(x)}{Q(x)} + \lambda (r(x) - E_{x~P}[r(x)])]}
$$