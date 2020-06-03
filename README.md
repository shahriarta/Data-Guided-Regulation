# Data-Guided Regulation (DGR)


In this repository, we provide the code for the simulations of the following manuscript: <br> <br>


### [Online Regulation of Unstable LTI Systems from a Single Trajectory](https://128.84.21.199/abs/2006.00125)

Shahriar Talebi, Siavash Alemzadeh, Newsha Rahimi, Mehran Mesbahi <br> <br>


---

## Abstract

<div align="justify"> Recently, data-driven methods for control of dynamic systems have received considerable attention in system theory and machine learning as they provide a mechanism for feedback synthesis from the observed time-series data. However learning, say through direct policy updates, often requires assumptions such as knowing <em> a priori </em> that the initial policy (gain) is stabilizing, e.g., when the open-loop system is stable. In this paper, we examine online regulation of (possibly unstable) partially unknown linear systems with no <em> a priori </em> assumptions on the initial controller. First, we introduce and characterize the notion of "regularizability" for linear systems that gauges the capacity of a system to be regulated in finite-time in contrast to its asymptotic behaviour (commonly characterized by stabilizability/controllability). Next, having access only to the input matrix, we propose the Data-Guided Regulation synthesis that (as its name suggests) regulates the underlying states while also generating informative data that can subsequently be used for data-driven stabilization or system identification. The analysis is also related in spirit, to the spectrum and the "instability number" of the underlying linear system, a novel geometric property studied in this work. We further elucidate our results by considering special structures for system parameters as well as boosting the performance of the algorithm via a rank-one matrix update using the discrete nature of data collection in the problem setup. Finally, we demonstrate the utility of the proposed approach via an example involving direct (online) regulation of the X-29 aircraft. </div>

![Image of X-29](http://depts.washington.edu/uwrainlab/wordpress/wp-content/uploads/2020/06/x-29-axis.png)


