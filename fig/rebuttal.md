A good article with theoretical contributions and good empirical results on synthetic and real data.
Official Reviewby Reviewer Ke2e25 Mar 2025, 01:46 (modified: 04 Apr 2025, 22:16)Program Chairs, Area Chairs, Reviewers Submitted, Reviewer Ke2e, AuthorsRevisions
Q1 Summary And Contributions:

The authors propose a new method to detect the root causes of observed outliers in the data. This method is an extension of the method called Integrated Gradient (IG) of [Nguyen et al. 2024].
Q2-1 Originality-Novelty: 3: Good: The paper makes non-trivial advances over the current state-of-the-art.
Q2-2 Correctness-Technical Quality: 3: Good: The paper appears to be technically sound, but I have not carefully checked the details.
Q2-3 Extent To Which Claims Are Supported By Evidence: 3: Good: the main claims are supported by convincing evidence (in the form of adequate experimental evaluation, proofs, (pseudo-)code, references, assumptions).
Q2-4 Reproducibility: 3: Good: key resources (e.g. proofs, code, data) are available and key details (e.g. proofs, experimental setup) are sufficiently well-described for competent researchers to confidently reproduce the main results.
Q2-5 Clarity Of Writing: 3: Good: The paper is well organized but the presentation could be improved.
Q3 Main Strengths:

The paper is well written and very informative. It contains theoretical contributions and good empirical results.
Q4 Main Weakness:

Some parts could perhaps be explained a little more. I'm thinking in particular of section 3.2.3, which is a little obscure for users unfamiliar with this type of denoising method.
Q5 Detailed Comments To The Authors:

Here are my main remarks and questions regarding the different sections of the paper.

Literature review :

I think that the method CAM [1] could be cited has is is proven that it can identity the true DAG in this case, given by Equation (2) and corresponding to a FCM with additive noise.

Model description :

Equation 2 is presented as a very general causal model, but is should be written that is a restriction to functional model with additive noise. The more general form is X_j = f_j(Pa_j, Z_j).

In the case under consideration, it seems that the additive noise must be Gaussian, in order to obtain a consistent likelihood estimate with MSE when training the regression neural networks in section 3.2.4.

In section 2.3, I find this sentence not very clear : "In this way, the noise variables z play the role of selecting deterministic mechanisms."

What is clearly the different between the IG score defined by equation 10 and the target outlier score defined by equation 12 ? It looks exactly the same, isn't it ?

I do not see the connection between Equation (11) and Equations (1) and (5). Normally, it should be the same score-based outlier measure, isn't it ? But maybe I missed something.

Section 3.2.3 seems at the core of the method and is not really clear to me, in particular the following sentence : "The gradient path sampling is done by “denoising” the outlier observation [...] where each path has k time steps."

I think it could be better explained in a new version of the paper.

The transitions between section 3.2.3 and 3.2.4 could also be improved, to better see why one needs to learn the impacts function "f_i".

I have two more general questions :

    Can this method be extended to the general FCM with non additive noise: X_i = f_i(X_pa(i), Z_i)) ?

    I see that the loss used to fit the model to the data is the MSE. I think that this loss is not very general has it corresponds to the likelihood for Gaussian distribution. Why not using a more general likelihood term that could be learned for example with adversarial learning ?

Theory :

It appears that his paper brings new theoretical results It shows that the score-based integrated gradient satisfies different axioms for outlier attribution problem.

Experiments :

The results seem very good in comparison with existing methods for this problem.

How are set the parameters of the networks to build the FCM ?

Minor comments :

    When defining Integrated Gradient method, it is written the "attribution value of node i", and after in equation 10, j is used.

    In section 3.2.4 a parameter phi_i is used but is is the same notation as equation 7. For better clarity, I think it could changed.

References :

[1] Bühlmann, P., Peters, J., & Ernest, J. (2014). CAM: Causal additive models, high-dimensional order search and penalized regression.
Q6 Overall Score: 6: Weak Accept: Technically solid paper, with no major concerns with respect to provided evidence, resources, reproducibility, and ethical considerations.
Q7 Justification For Your Score:

This paper is based on the use of some existing techniques and deals with a problem that seems to have a relatively restricted field of application. However, it makes some interesting theoretical contributions to the problem, and moreover the proposed method shows good empirical results on synthetic and real data. I think the method could be explained a little more clearly, especially in section 3.2.3.
Q8 Confidence In Your Score: 3: Somewhat confident, but there's a chance I missed some aspects. I did not carefully check some of the details, e.g. novelty, proof of a theorem, experimental design, or statistical validity of conclusions. I am somewhat familiar with the topic but may not know all related work.
Q9 Complying With Reviewing Instructions: Yes
Add:
Review of "Score-based Integrated Gradient for Root Cause Explanations of Outliers"
Official Reviewby Reviewer tYPK22 Mar 2025, 13:17 (modified: 04 Apr 2025, 22:16)Program Chairs, Area Chairs, Reviewers Submitted, Reviewer tYPK, AuthorsRevisions
Q1 Summary And Contributions:

The paper proposes SIREN to detect the root cause of outliers based on the assumption that data are generated from an additive noise SCM. They propose to use an integrated gradient of the log-likelihood as an outlier measure. Under the model the integrand can be estimated via score matching and the derivative of an estimated regression function. The method is evaluated against its contemporaries on an simulated random graph dataset and a cloud service latency dataset.
Q2-1 Originality-Novelty: 2: Fair: The paper contributes some new ideas.
Q2-2 Correctness-Technical Quality: 3: Good: The paper appears to be technically sound, but I have not carefully checked the details.
Q2-3 Extent To Which Claims Are Supported By Evidence: 2: Fair: the main claims are somewhat supported by evidence (but the experimental evaluation may be weak, or does not match entirely with the claims, important baselines may be missing, proofs contain important ideas but lack rigor, algorithmic details are only discussed superficially, references are imprecise, assumptions are not sufficiently motivated or explicated, etc.).
Q2-4 Reproducibility: 1: Poor: key details (e.g. proof sketches, experimental setup) are incomplete/unclear, or key resources (e.g. proofs, code, data) are unavailable
Q2-5 Clarity Of Writing: 2: Fair: The paper is somewhat clear, but some important details are missing or unclear.
Q3 Main Strengths:

    Score matching is a non-parametric technique that avoids parametric assumptions about the noise distribution, compared to the closely related previous work (Nguyen et al., 2024).
    Experimental results show that SIREN generally outperforms other methods.
    Writing and derivations are detailed and clear.

Q4 Main Weakness:

    Some contributions are incremental and/or already known/trivial.
    Outliers being defined as outlying instances of additive noise seems less general than previous works.
    Usage of raw negative log-likelihood as a outlier measure largely unjustified compared to other methods.
    Experimental design and results are not particularly compelling, and no experimental details (e.g., what architecture did you use for the mean estimators?) is available.

Q5 Detailed Comments To The Authors:
Contributions.

I think the authors should clarify their contributions: detailed points below.

    Contribution (1): The "conditional" score matching refers to applying existing score matching techniques to the residuals and is not inherently conditional nor novel. The gradient paths section (3.2.3) was unclear: are you using the score estimate to generate normal samples, i.e., denoising diffusion models, or are you somehow using the SDE to estimate the path integral in (10)? The former is not novel and seems unnecessary to include as a contribution. The latter is very interesting but it is unclear how you can achieve this.

    Contribution (2): This decomposition is straight forward to obtain from the additive structure and the chain rule, and a very similar decomposition has already been leveraged in the context of causal discovery, see Equation (8) of Montagna et al., 2023 (there, the score is w.r.t. 

instead of

    ). I think the authors derivation here is strictly speaking novel but is not suitable to qualify as a main contribution.

    Contribution (3): I was confused at the connection to Shapley values, do you mean the connections to the typical axioms? Regarding this, it seems that Axioms 1-3 are derived fully based on the integrated gradient form rather than the proposed score function. Can you clarify your contribution here in view of the existing literature (i.e., Sundararajan and Najmi 2020)?

Definition of Outliers

It seems more reasonable to assume that outliers are caused by drastic changes in mechanisms, which is the perspective of Budhathoki et al., 2022 and to a lesser extent Nguyen et al., 2024 (through the perturbation of a linear coefficient prior). This would indicate a failure of causal invariance, which to me seems to be a better interpretation of outlying points rather than an outlier in the unobserved additive noise.
Justification of Outlier Measure

I understand using the negative log-likelihood as an outlier measure on an intuitive level. But compared to the previous works, which make information theoretic arguments for example, why is this a reasonable choice? Maybe there is some relation to the classical notion of influence functions?
Experimental Details

I think this section needs an Appendix detailing e.g., training hyperparameters, how the weights of the synthetic MLPs are generated, how the graphs are generated, etc.. It was also unclear to me whether the 3-layer MLPs referred to the synthetic SCM or the regression function.
Final Suggestions

I think it is neat that the integrated gradients and score function naturally tie together when using the negative log-likelihood as an outlier measure. To me this is a compelling contribution of this work compared to the previous shapley or parametric methods. I think the authors would do well if they focused on this point and conducted a more detailed study of the negative log-likelihood as an outlier measure, and maybe moved away from additive noise (you can still autodiff to obtain the score w.r.t. the noise from an estimated neural non-additive SCM!)
Q6 Overall Score: 3: Reject: For instance, a paper with technical flaws, limited novelty, weak experimental evaluation, inadequate reproducibility, incompletely addressed ethical considerations.
Q7 Justification For Your Score:

For me the contribution of this paper is either very limited or not well-communicated enough to me (W1). The strengths I listed I think describes more the potential of the study rather than the current execution. It is promising that the simulations work well, but for me some steps in the direction of the suggestions I listed would be required for a higher score (although, I think it is probably outside of the scope of the review period).
Q8 Confidence In Your Score: 3: Somewhat confident, but there's a chance I missed some aspects. I did not carefully check some of the details, e.g. novelty, proof of a theorem, experimental design, or statistical validity of conclusions. I am somewhat familiar with the topic but may not know all related work.
Q9 Complying With Reviewing Instructions: Yes
Add:
Convincing advancement using counterfactual root cause analysis
Official Reviewby Reviewer Ujz419 Mar 2025, 10:05 (modified: 04 Apr 2025, 22:16)Program Chairs, Area Chairs, Reviewers Submitted, Reviewer Ujz4, AuthorsRevisions
Q1 Summary And Contributions:

This paper introduces a method for root causes analysis of outliers in a system using the causal structured and relationships between variables. By utilizing integrated gradients along paths from outliers to normal data, the authors improve on a similar technique for attributing root causes to (causal) upstream nodes. The method has been evaluated in artificial data.
Q2-1 Originality-Novelty: 3: Good: The paper makes non-trivial advances over the current state-of-the-art.
Q2-2 Correctness-Technical Quality: 3: Good: The paper appears to be technically sound, but I have not carefully checked the details.
Q2-3 Extent To Which Claims Are Supported By Evidence: 3: Good: the main claims are supported by convincing evidence (in the form of adequate experimental evaluation, proofs, (pseudo-)code, references, assumptions).
Q2-4 Reproducibility: 2: Fair: key resources (e.g. proofs, code, data) are unavailable but key details (e.g. proof sketches, experimental setup) are sufficiently well-described for an expert to confidently reproduce the main results.
Q2-5 Clarity Of Writing: 4: Excellent: The paper is well-organized and clearly written.
Q3 Main Strengths:

    Novel idea using integration of score-based methods
    Fair discussion of drawbacks of related methods
    Convincing experimental results (although limited to synthetic data)
    Generalization of existing methods

Q4 Main Weakness:

    While the synthetic experiments are convincing, there is a lack of real-world data experiments
    Implementation details are unclear, e.g., how sensitive it is to different hyperparameters
    While the assumptions are implicit based on the general framework, they are not explicitly mentioned
    Proof details are limited

Q5 Detailed Comments To The Authors:

The work is a great and fair extension of the work by Budhathoki et al. Overall, the paper is easy to follow. My only concern is the lack of real-world evaluation and baseline comparison. Some other remarks/questions:

    A clear list of assumptions are missing, although they are implicitly clear from the setup. The paper could be improved by more explicitly listing them.
    You mention that one needs to assume a distribution when reconstructing the noise in related work. However, one could also take the empirical distribution (i.e., the residuals) which do not need to assume any specific noise distribution. This slightly weakens your point regarding arbitrary noise distributions.
    Eq. 12 and the equation used in proof Axiom 1 seem slightly different without explanation.
    It is unclear how well your approach performs when core assumptions are violated (e.g. wrong graphs or hidden confounders with different strengths). Some discussions on this could be insightful.

Q6 Overall Score: 6: Weak Accept: Technically solid paper, with no major concerns with respect to provided evidence, resources, reproducibility, and ethical considerations.
Q7 Justification For Your Score:

The work is a non-trivial extension of existing work with well theoretical foundation. The only concern here is the lack of real-world evaluation and clarification of assumptions. Nevertheless, the experiments in the given artificial data is convincing. Some of the proofs could be more detailed.
Q8 Confidence In Your Score: 4: Quite confident. I tried to check the important points carefully. It is unlikely, though conceivable, that I missed some aspects that could otherwise have impacted my evaluation. I am familiar with the research topic and most of the related work.
Q9 Complying With Reviewing Instructions: Yes
Add:
Review for Score-based Integrated Gradient for Root Cause Explanations of Outliers
Official Reviewby Reviewer GBpw19 Mar 2025, 08:31 (modified: 04 Apr 2025, 22:16)Program Chairs, Area Chairs, Reviewers Submitted, Reviewer GBpw, AuthorsRevisions
Q1 Summary And Contributions:

The paper studies the problem of root cause explanation of outliers in causal inference and anomaly detection. Traditional approaches, such as heuristic-based methods and counterfactual reasoning, struggle with uncertainty and high-dimensional dependencies. The authors propose SIREN which estimates score functions of data likelihood to compute attributions using integrated gradients. It operates directly on score functions, aligns with Shapley value principles by satisfying key axioms, and works for nonlinear causal models with unknown noise sources. Experiments are conducted to show the outperformance compared to existing methods synthetic graphs and real-world cloud service latency data.
Q2-1 Originality-Novelty: 3: Good: The paper makes non-trivial advances over the current state-of-the-art.
Q2-2 Correctness-Technical Quality: 3: Good: The paper appears to be technically sound, but I have not carefully checked the details.
Q2-3 Extent To Which Claims Are Supported By Evidence: 3: Good: the main claims are supported by convincing evidence (in the form of adequate experimental evaluation, proofs, (pseudo-)code, references, assumptions).
Q2-4 Reproducibility: 3: Good: key resources (e.g. proofs, code, data) are available and key details (e.g. proofs, experimental setup) are sufficiently well-described for competent researchers to confidently reproduce the main results.
Q2-5 Clarity Of Writing: 3: Good: The paper is well organized but the presentation could be improved.
Q3 Main Strengths:

    Instead of relying on explicit likelihood functions, the proposed method estimates score functions, which improves computational efficiency and robustness in high-dimensional settings.
    The proposed method works with nonlinear causal models and unknown noise distributions, unlike methods assuming linear Gaussian structures.
    The method satisfies three out of four Shapley axioms and introduces a novel asymmetry axiom derived from causal structures.
    Experiemtns show higher accuracy in identifying multiple root causes compared to baselines.

Q4 Main Weakness:

    The accuracy of the score-based estimation is crucial, but the paper does not discuss how errors in score estimation might propagate to attributions.
    The paper does not formally define the task, i.e. what is root cause analysis for outliers. It took an average reader some time to approximately figure out what the paper is trying to do, especially with the extra layer of graphical model context.
    While the paper claims efficiency, it does not provide runtime comparisons against baselines.

Q5 Detailed Comments To The Authors:

    The proposed method relies on a new notation of outlier measure (defn 1), which is different from the one introduced before (eq 1). What is the reason behind to introduce a new measure? how to compare with eq 1? what is the trade-off of using the new measure?
    How robust is SIREN to incorrect or misspecified causal structures?
    Would using other density estimation techniques further improve attribution accuracy?

Q6 Overall Score: 6: Weak Accept: Technically solid paper, with no major concerns with respect to provided evidence, resources, reproducibility, and ethical considerations.
Q7 Justification For Your Score:

The paper studies an interesting question and provides an appealing method by combining many advanced techniques. I am not very familiar with this literature.
Q8 Confidence In Your Score: 1: Not confident. My evaluation is an educated guess, or the topic is outside my area of expertise.
Q9 Complying With Reviewing Instructions: Yes
Q10 Ethical Concerns:

No concern.
