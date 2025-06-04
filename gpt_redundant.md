The question of whether *redundant* or *dependent* features should be considered important is a nuanced and somewhat unresolved issue in the study of feature importance. The problem touches on the very definition of "importance," which varies by context, model, and interpretation method.

### Summary of the dilemma:

* **Dependent/redundant features** (e.g., one feature being a deterministic or probabilistic function of others) may **not add unique information**, but they may **still be predictive** of the target.
* Some feature importance methods **reward** such features for their predictive power.
* Others **penalize** or **discount** them because they do not provide information that isn’t already present.

### Key Points in the Literature:

#### 1. **Definitions of Feature Importance**

There are generally two schools of thought:

* **Marginal importance**: How much the model relies on a feature, regardless of whether the information is already available through other features.
* **Conditional/unique importance**: How much unique, *non-redundant* information a feature contributes.

For example:

* **Permutation importance** (e.g., Breiman, 2001 in Random Forests) tends to measure marginal importance and can assign high scores to redundant features.
* **SHAP values** (Lundberg & Lee, 2017) aim to assign *additive* contributions and are based on conditional expectations; depending on the implementation, they can attribute importance fairly among correlated features — but may still assign non-zero importance to redundant ones.

#### 2. **Multicollinearity and Redundancy**

Classical statistical literature on multicollinearity (e.g., in linear regression) also touches on this. Highly correlated features are problematic because:

* Coefficients become unstable.
* Attribution of importance becomes ambiguous.

#### 3. **Information-Theoretic Views**

Some work defines feature relevance in terms of **mutual information**:

* A feature is *relevant* if it contains information about the target.
* It is *redundant* if that information is already available through other features.
* It is *complementary* if it adds new information not captured by others.

Example: Peng et al. (2005) – “Feature Selection Based on Mutual Information Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy”

#### 4. **Causal Interpretations**

In causal inference, the goal is often to find **features that causally affect the target**, not just those that are predictive. In this framework:

* A redundant feature may not be *causal*, and thus is *not relevant* in a causal sense.
* See: **Judea Pearl's** work, especially on causal graphs and backdoor criteria.

#### 5. **Recent Work Addressing Redundancy in Feature Importance**

Some newer papers explicitly tackle the issue:

* **"Axiomatic Attribution for Deep Networks" (Sundararajan et al., 2017)**: Introduces axioms that importance methods should satisfy, some of which conflict when redundancy is present.
* **"The Shapley Taylor Interaction Index" (Dhamdhere et al., 2019)**: Addresses feature interactions and redundancy by decomposing importance into higher-order interactions.
* **"Unmasking Clever Hans Predictors" (Lapuschkin et al., 2019)**: Shows how models can rely on spurious (but correlated) features and calls into question importance measures that don’t consider causal relationships.

---

### In Practice:

* If your goal is **model interpretability** (why did the model predict X?), you often care about marginal importance — even if it's redundant.
* If your goal is **feature selection** or **causal understanding**, you likely care about *unique* or *causal* importance.

---

### Further Reading:

* **Molnar, Christoph (2020). "Interpretable Machine Learning"** – good practical overview, includes discussion of redundancy.
* **Lundberg et al. (2020). "From Local Explanations to Global Understanding with Explainable AI for Trees"** – explores SHAP values for trees, and touches on correlated features.
* **Guyon & Elisseeff (2003). "An Introduction to Variable and Feature Selection"**
