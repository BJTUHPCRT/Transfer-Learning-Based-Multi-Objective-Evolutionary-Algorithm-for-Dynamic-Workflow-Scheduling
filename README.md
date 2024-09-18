# Transfer-Learning-Based-Multi-Objective-Evolutionary-Algorithm-for-Dynamic-Workflow-Scheduling
The core code of the article "Transfer Learning Based Multi-Objective Evolutionary Algorithm for Dynamic Workflow Scheduling in the Cloud"
# DOI
https://doi.org/10.1109/TCC.2024.3450858
# Abstract
Managing scientific applications in the Cloud poses many challenges in terms of workflow scheduling, especially in handling multi-objective workflow scheduling under quality of service (QoS) constraints. However, most studies address the workflow scheduling problem on the premise of the unchanged environment, without considering the high dynamics of the Cloud. In this paper, we model the constrained workflow scheduling in a dynamic Cloud environment as a dynamic multi-objective optimization problem with preferences, and propose a transfer learning based multi-objective evolutionary algorithm (TL-MOEA) to tackle the workflow scheduling problem of dynamic nature. Specifically, an elite-led transfer learning strategy is proposed to explore effective parameter adaptation for the MOEA by transferring helpful knowledge from elite solutions in the past environment to accelerate the optimization process. In addition, a multi-space diversity learning strategy is developed to maintain the diversity of the population. To satisfy various QoS constraints of workflow scheduling, a preference-based selection strategy is further designed to enable promising solutions for each iteration. Extensive experiments on five well-known scientific workflows demonstrate that TL-MOEA can achieve highly competitive performance compared to several state-of-art algorithms, and can obtain triple win solutions with optimization objectives of minimizing makespan, cost and energy consumption for dynamic workflow scheduling with user-defined constraints.
# Environment
Workflowsim framework (https://github.com/WorkflowSim/WorkflowSim-1.0), JMetal framework, DL4J framework, Windows 10, Java 13, Eclipse IDE for Java Developers - 2022-03, Apache Maven 3.8.6.
# Running
TL4DMOEAplanningAlgorithmExample.java
# Dataset
5 types of scientific workflows
# Citation
H. Xie, D. Ding, L. Zhao and K. Kang, "Transfer Learning Based Multi-Objective Evolutionary Algorithm for Dynamic Workflow Scheduling in the Cloud," in IEEE Transactions on Cloud Computing, doi: 10.1109/TCC.2024.3450858.
# Note
Due to GitHub's file size limitations, we have only uploaded the core code for TL-MOEA. You may encounter challenges running the code in its entirety. If you need access to the complete code, feel free to reach out to us without hesitation.
# Contact
fhzm1995@163.com or 20112033@bjtu.edu.cn
