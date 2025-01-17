.. currentmodule:: chi

.. _Dataset_1 : https://github.com/DavAug/chi/blob/main/docs/source/getting_started/code/dataset_1.csv
.. _Pints: https://github.com/pints-team/pints
.. _ArViz: https://python.arviz.org/en/stable/
.. _Issue: https://github.com/DavAug/chi/issues

**********************************
Fitting mechanistic models to data
**********************************

In the previous tutorial, :doc:`mechanistic_model`, we have seen how we can
implement and simulate treatment response models in Chi. For example, we can simulate
the time course of drug concentration levels by 1. implementing a model in SBML;
2. setting the route of administration and the dosing regimen; 3. simulating the
drug concentration for a given set of model parameters.

.. literalinclude:: code/3_fitting_models_1.py
    :lines: 112-146

.. raw:: html
   :file: images/3_fitting_models_1.html

This functionality to simulate treatment responses is pretty cool in its own
right! For example, it can help us to study nonlinearities in the treatment response
dynamics and to optimise dosing regimens to target a desired treatment response.

However, at this point, the simulated treatment responses have little to do with
real treatment responses. To describe *real* treatment
responses, i.e. treatment responses that we may observe in clinical practice, we need to somehow connect
our model to reality.

The most common approach to relate models to real treatment responses is to
compare the model predictions to measurements. Below, we have prepared an example
dataset with drug concentration measurements. These drug concentrations were
measured after repeatedly adminstering 2 mg of a drug every 24 hours.
You can download the dataset from the following link: Dataset_1_.

.. csv-table:: Drug concentration measurements
   :file: code/dataset_1.csv
   :widths: 4, 12, 12, 12, 12, 12, 12, 12, 12
   :header-rows: 1

The dataset contains one column that identifies measured individuals (``ID``),
two columns that specify measurement times and dose administration times
(``Time`` and ``Time unit``), three columns that specify measured values
(``Observable``, ``Value``, ``Observable unit``), and three columns that specify
dose administrations (``Dose``, ``Duration``, ``Dose unit``).

Downloading the file and saving it in the same directory as the Python script,
we can visualise the measurements by executing the below script

.. literalinclude:: code/3_fitting_models_1.py
    :lines: 176-195

.. raw:: html
   :file: images/3_fitting_models_2.html

The figure shows the drug concentration measurements as blue scatter points.
The treatment response dynamics indicated by the measurements
is similar to the simulated treatment response in the previous code block.
But looking more closely at the magnitude of the values,
it appears that the measured values are much smaller
than the simulated ones. We can therefore conclude that, at this point, our
model does not provide an accurate description of the measured treatment
response.

To find a better description of the treatment response, we have two options:
1. we can try to find parameter values that improve the proximity of the
model output to the measurements; or 2. we can define a new mechanistic model and see
whether this new model is able to describe the measurements better. This tutorial
will be about the former and will detail how we can find better model parameters
for a given model structure.

Estimating model parameters from data: Background
*************************************************

Before we can try to find better parameter values that describe the observed
treatment response, we first need to agree on what we mean by
"*better*" for the relationship between the mechanistic model output and the measurements.
An intuitive notion of "better" is "closer", quantified by the distance
between the measurements and the model output,
i.e. the difference between the measured values and the
simulated values. Then the model parameters that best
describe the measurements would be those parameter values that make the mechanistic
model output perfectly match the measurements, resulting in distances of 0 ng/mL
between the model output and the measurements at all measured time points.
However, as outlined in Sections 1.3 and 1.4 of the
:doc:`quick_overview`, measurements of treatment responses are imprecise and
noisy, and will therefore not perfectly represent the treatment response dynamics.
Consequently, if we were to match the model outputs to measurements perfectly,
we would end up with an inaccurate description of the treatment response
that is corrupted by measurement noise.

One way to overcome this limitation is to change our notion of "better" and incorporate the measurement
process into our computational model of the treatment response. This makes explicit
that we do not expect the mechanistic model output to match
the measurements perfectly. In Chi, the measurement process can be captured
using :class:`chi.ErrorModel` s. Error models promote the single value output
of mechanistic model simulations to a distribution of
values. This distribution characterises a
range of values around the mechanistic model output where measurements may be
expected.
We can use this measurement distribution in two ways: 1. for simulation; and 2. for
parameter estimation. For simulation, the distribution
can be used to sample measurement values and
imitate the measurement process of real treatment responses, see
Section 1.3 in the :doc:`quick_overview` for an example. For parameter estimation,
the distribution can be used to quantify the likelihood with which the observed
measurements would have been generated by our model,
see Section 1.4 in the :doc:`quick_overview`.

Formally, we denote the measurement distribution by :math:`p(y | \psi, t, r)`,
where :math:`y` denotes the measurement value, :math:`\psi` denotes the model parameters,
:math:`t` denotes the time point of the measurement, and :math:`r` denotes the adminstered
dosing regimen. With this measurement distribution in place, we can quantify the
likelihood with which any given set of measurements would have been generated by the
model. For example, the likelihood of a measurement :math:`y_1` at time :math:`t_1`
under dosing regimen :math:`r^*` is defined by the value of the probability density
of the measurement distribution evaluated at the measurement,
:math:`p(y_1 | \psi, t_1, r^*)`. Note that this
likelihood depends on the choice of model parameters, :math:`\psi`. The model
parameters with the maximum likelihood are
the parameter values that "best" describe the measurements.

.. note::
    The measurement distribution, :math:`p(y | \psi, t, r)`, is defined
    by the mechanistic model output and the error model. To illustrate this, let
    use denote the simulated drug concentration values of the
    the 1-compartment PK model by :math:`c(\psi, t, r)`. By definition of
    the model, the drug concentration values, :math:`c`, are
    a function of the model parameters, :math:`\psi = (a_0, k_a, k_e, v)`, the time,
    :math:`t`, and the dosing regimen, :math:`r`.

    1. If we choose a :class:`chi.GaussianErrorModel` to describe the difference
    between the model output and the measurements, we assume that measurements
    are distributed according to a Normal distribution around the model output

    .. math::
        p(y | \psi, t, r) = \frac{1}{\sqrt{2\pi \sigma ^2}}\mathrm{e}^{-\big(y - c(\psi, t, r)\big)^2 / 2\sigma ^2},

    where :math:`\sigma` defines the width of the distribution. For ease of notation,
    we extend the definition of the model parameters to include :math:`\sigma`,
    :math:`\psi = (a_0, k_a, k_e, v, \sigma)`.

    We can see that the mechanistic model output
    defines the mean or Expectation Value of the measurement distribution.

    2. If we choose a :class:`chi.LogNormalErrorModel` to describe the difference
    between the model output and the measurements, we assume that measurements
    are distributed according to a lognormal distribution around the model output

    .. math::
        p(y | \psi, t, r) = \frac{1}{\sqrt{2\pi \sigma ^2}}\frac{1}{y}\mathrm{e}^{-\big(\log y - \log c(\psi, t, r) + \sigma / 2\big)^2 / 2\sigma ^2}.

    One can show that also for this distribution the mechanistic model output defines the mean
    or Expectation Value of the measurement distribution.

    The main difference between the two distributions is the shape. The
    :class:`chi.GaussianErrorModel` is symmetrically distributed around the
    model output, while :class:`chi.LogNormalErrorModel` is scewed in such a
    way that measurements can never assume negative values. To visualise these
    differences, we recommend simulating many measurements with different
    error models, similar to Section 1.3 in :doc:`quick_overview`. But instead
    of choosing different times, sample all measurements at the same time. You
    can then histogram the samples, using for example ``go.Histogram``, as used
    in Section 1.4.2 in :doc:`quick_overview`, to visualise the shape of
    the probability density.


Assuming independence of measurements, the likelihood for a dataset with :math:`n` measurements,
:math:`\mathcal{D}=((t_1, y_1), (t_2, y_2), \ldots, (t_n, y_n), r)`, is given
by the product of the individual likelihoods

.. math::
    p(\mathcal{D}| \psi ) = \prod _{j=1}^n p(y_n | \psi, t_n, r),

where *independence* refers to the assumption that measurements are
independently and identically distributed according to our model of the
measurement process (this does not have to be the case, and is especially unlikely
to be the case when our model fails to describe the measurement process accurately).

With this likelihood in place, Chi makes it straightforward to run numerical optimisation algorithms in order
to derive the maximum likelihood estimates that best describe the measurements,
see Section 1.4.1 in the :doc:`quick_overview`.

Alternatively, Chi also supports Bayesian inference of the model parameters:
Bayesian inference is conceptually different from maximum likelihood estimation
as it does not seek to find a single set of model parameters that "best"
describes the observed measurements. Instead, Bayesian inference acknowledges
the fact that noisy measurements leave uncertainty about which model parameters
best describe the data, and therefore focuses instead on deriving
a distribution of parameter values which
are all consistent with the observed measurements, see Section 1.4.2 in :doc:`quick_overview`
for a more detailed discussion. This
distribution is derived from the likelihood using Bayes' rule

.. math::
    p(\psi| \mathcal{D} ) = \frac{p(\mathcal{D}| \psi )\, p(\psi)}{p(\mathcal{D} )},

where :math:`p(\psi)` denotes the prior distribution of the model parameters.
The prior distribution quantifies the baseline distribution of parameter values
before the parameter estimation, making it possible to inform the inference results
with otherwise hard-to-quantify knowledge of feasible or likely parameter values prior to the
inference.

Defining the log-posterior
**************************

In Chi, we can derive posterior distributions from measurements
using Markov chain
Monte Carlo (MCMC) algorithms implemented in the open source Python package
Pints_.
In Sections 1.4.1 and 1.4.2 in the :doc:`quick_overview`,
we showed in some detail how we can define (log-)posterior distributions,
:class:`chi.LogPosterior`, for this purpose. Here, we want to show
how we can use the :class:`chi.ProblemModellingController` to simplify and automate
this process as much as possible.

The tricky bit when implementing log-posteriors for treatment response models
is often that log-posteriors do not only
depend on the treatment response measurements, :math:`((t_1, y_1), (t_2, y_2), \ldots, (t_n, y_n))`,
but also on the administered dosing regimen, :math:`r`.
This can make it tedious to define log-posteriors,
especially when parameters across measurements of multiple
individuals with different dosing regimens are inferred, as dosing regimens of the
model will have to be manually set to the dosing regimens administered during the measurement
process. A mismatch between the dosing regimens would render the inference results invalid.

To eliminate this source of error, we have implemented the :class:`chi.ProblemModellingController`.
The :class:`chi.ProblemModellingController` facilitates the construction of
log-posteriors and reduces the workflow to a simple 4-step approach:

- 1. Definition of the mechanistic model
- 2. Definition of the error model
- 3. Definition of the measurements
- 4. Definition of the prior distribution

In the below code block, we illustrate this workflow for the above drug
concentration dataset, Dataset_1_.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 651-688

The first four blocks in the code define the individual components of the
log-posterior: the mechanistic model, the error model, the data, and the prior
distribution. Note that the administration of the dosing regimen is set
before passing the mechanistic model to the :class:`ProblemModellingController`.

The prior distribution defines marginal distributions for the parameters, and
is implemented using Pints_. In Bayesian inference, we can use the prior
distribution to bias the inference
results towards feasible areas of the parameter space. In this
case, we have little prior knowledge about feasible parameter ranges, so we
choose relatively uninformative prior
distributions (below will be an illustration of the prior distribution).
Note that we seem to be specifying marginal prior distributions for only
4 of the 6 model parameters. This is because we fix the values of the initial drug
amounts to ``0`` prior to the inference
in the lines below, reflecting our knowledge that the subject had no prior
exposure to the drug before starting the trial. This reduces the number of model
parameters from 6 to 4. The fixing of model parmaters
is optional, but can sometimes improve the inference results when some model
parameters are already well understood.

For the remaining 4 parameters, only positive values make biological sense, so
we choose prior distributions that focus on positive values. For
two model parameters, the volume of distribution and the scale parameter,
negative or zero values are particularly bad as they will break the simulation
(1. a volume of zero causes a division by zero error; and 2. the lognormal distribution
is only defined for positive sigma). We therefore use ``pints.LogNormalLogPrior``
to constrain those parameters to strictly positive values.

In the final block of the code, we define the log-posterior. In the first line,
we specify the mechanistic model and the error model. In the next line, we set
the dataset. Note that we need to use the ``output_observable_dict`` to map the
output variable of the model, ``global.drug_concentration``, to the Observable name
in the dataset, ``Drug concentration``. Other specifications are not required, and
dosing regimens are automatically set, when the dosing regimen related columns,
``Dose`` and ``Duration``, are present in the dataset. In the following line, we
fix the initial drug amounts to ``0`` using :meth:`ProblemModellingController.fix_parameters`.
You can use this method to fix any parameters of the model.
In the last two lines, we set the log-prior and implement the log-posterior using
the :class:`ProblemModellingController.get_log_posterior` method.

Inferring the posterior distribution
************************************

With this :class:`chi.LogPosterior` in place, we can infer the posterior
distribution using any MCMC algorithm of our choice. Recall that inference, in this context,
means the reconstruction of the posterior distribution from the
:class:`chi.LogPosterior`, which defines the log-pdf of the posterior distribution
up to an unknown constant shift

.. math::
    \log p(\psi| \mathcal{D} ) = \log p(\mathcal{D}| \psi ) + \log p(\psi) + \text{constant}.

By comparison to above Bayes rule, you will find that :math:`\text{constant} = -\log p(\mathcal{D})`,
so the constant shift may seem not so 'unkown' after all. However, *unknown* here
does not mean that we cannot write down an expression for it, it
refers to the fact that for most treatment response models
:math:`p(\mathcal{D})` is (practically) impossible to evaluate, as
evaluating :math:`p(\mathcal{D})` requires the numerical integration of the
likelihood-prior product over the full parameter space,
:math:`p(\mathcal{D}) = \int \mathrm{d} \psi \, p(\mathcal{D}, \psi ) = \int \mathrm{d} \psi \, p(\mathcal{D}| \psi )\, p(\psi)`.
This renders the value of the constant shift for all intents and purposes unknown.

The unknown shift makes it very difficult to make statements about the absolute probability
of parameter values from the :class:`chi.LogPosterior` alone. However, the uknown shift does allow for relative comparisons of
probabilities as the shift is the same for all parameter values -- a fact exploited by MCMC algorithms to circumvent the limitation
of the partially known log-posterior. MCMC algorithms use
the relative comparison of parameter probabilities to generate random samples from the
posterior distribution, opening a gateway to reconstruct the distribution. The
more random samples are generated, the closer the histogram over the samples will
approximate the original posterior distribution. In fact, one can show that the histogram
will converge to the posterior distribution as the number of samples approaches
infinity. This makes it possible for MCMC algorithms
to reconstruct any posterior distribution from a :class:`chi.LogPosterior`.

To illustrate this, let us run an MCMC algorithm to infer the above
defined posterior distribution of the 1-compartment PK
model.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 692-698

In the code block, we use an MCMC algorithm implemented in Pints_, called
``pints.HaarioBardenetACMC``. For technical reasons that we will discuss below,
we run the algorithm three times for 20,000 iterations.
Note that we use the :class:`chi.SamplingController` to set the number of
runs, the number of iterations, and to run the sampling algorithm. The
:class:`chi.SamplingController`  can also do other things, such as running the
chains in parallel, but we will not go into this additional functionality in this
tutorial, and refer instead to the API reference.

Executing the above code block will spawn a response of the below form.
The left most column indicates the current
iteration of the MCMC algorithm. The other columns show the total number of
log-posterior evaluations and the fraction of accepted MCMC proposals of each
chain. The target acceptance ratio depends on the details of the MCMC algorithm.
For the ``pints.HaarioBardenetACMC`` the target is ``0.23``.

.. code-block:: bash

    Using Haario-Bardenet adaptive covariance MCMC
    Generating 3 chains.
    Running in sequential mode.
    Iter. Eval. Accept.   Accept.   Accept.   Time m:s
    0     3      0         0         0          0:00.0
    1     6      0         0.5       0.5        0:00.0
    2     9      0.333     0.667     0.333      0:00.0
    3     12     0.25      0.75      0.5        0:00.0
    20    63     0.714     0.571     0.476      0:00.0
    40    123    0.756     0.61      0.561      0:00.0
    60    183    0.738     0.475     0.59       0:00.0
    80    243    0.716     0.407     0.642      0:00.1
    100   303    0.693     0.406     0.653      0:00.1
    120   363    0.736     0.421     0.686      0:00.1
    .
    .
    .

When the three runs of the algorithm terminate, the generated samples are returned in
form of an ``xarray.Dataset``.
We can visualise the samples using the code
block documented at the end of this section (we move the code block to the end
to avoid disrupting the flow of the tutorial with less relevant code snippets).

.. raw:: html
   :file: images/3_fitting_models_3.html

The left column of the
figure shows the histogram over the samples across the three runs
in orange, as well as the probability density of the prior distribution in black.
The first row shows the
samples of the absorption rate, the second row shows the samples of the elimination
rate, the third row shows the samples of the volume of distribution, and the
fourth row shows the samples of the scale parameter of the error model, sigma.
The right column of the figure shows the same
samples, this time visualised over the iterations at which they were drawn.
The samples from the different runs are illustrated in different
colours: run 1 (green); run 2 (red); run 3 (blue).

The posterior distribution
^^^^^^^^^^^^^^^^^^^^^^^^^^

The orange distribution is the result of the inference -- the posterior distribution.
It contains all parameter values that are consistent with the drug concentration
measurements and our prior knowledge, assigning each set of parameter values
with a probability of being the data-generating parameter values. Notably, the
figure shows that our prior knowledge and Dataset_1_ are insufficient to
conclude on a single set of parameter values (see orange distribution).
Instead, the measurements only
allow us to refine our understanding of feasible parameter values. For example,
we can see in the second row of the figure that the marginal posterior distribution
substantially differs from the marginal prior distribution. This is because the
drug concentration measurements contain important information about the elimination rate, rendering
rates above 1.5 1/day or below 0.25 1/day as extremely unlikely for the
model of the treatment response. This is in stark contrast to the relatively wide
range of model parameters that we deemed feasible prior to the inference
(see black line). However, the measurements are not conclusive enough
to reduce the distribution of feasible elimination rates to a single value. Similarly,
for the volume of distribution (row 3) and the error scale parameter
(row 4), the measurements lead to substantial updates relative to the
prior distribution.
In comparison, the measurements appear less informative about the absorption rate
(see row 1), given that the marginal posterior distribution of
the absorption rate is almost identical to its prior distribution.
We will have a closer look at an intuitive understanding of why the measurements
contain little information about
the absorption rate below. The take-away from this discussion is that inferring
distributions of parameter values consistent with the measurements is in many
ways more informative than just estimating a single set of model parameters,
as it can help to quantify the feasibility of different parameter values and the
overall remaining level of uncertainty.
In a Bayesian setting, inferring posterior distributions also
allows us to supplement the information contained in the measurements by prior
knowledge of feasible model parameters, facilitating the inference of parameter
values when treatment response measurements are limited.

Assessing convergence
^^^^^^^^^^^^^^^^^^^^^

Similar to most numerical inference
algorithms, MCMC algorithms have practical
limitations, making them not equally well suited to all inference problems.
Some MCMC algorithms are, for example, only well suited to estimate model
parameters when the total number of parameters is small, while other MCMC algorithms only
work when the curvature of the posterior surface is not too extreme.
One way to test the suitability of an MCMC algorithm for a particular inference
problem is the use of the :math:`\hat{R}` metric.
We will motivate the metric below and provide an intuitive explanation.
Technical details are left to interested readers to explore on their own.

Let us begin this section by revisiting the right column in the figure above. The column
shows the samples from the three MCMC algorithm runs at each
iteration. For early iterations of the algorithm,
the samples from the MCMC runs look quite distinct -- each run appears to sample
from a different area of the parameter space. In contrast, at later iterations
the MCMC runs are harder to distinguish and sample from the same area of the parameter space.

Intuitively,
it does not really make sense for the samples from the MCMC runs to look different
-- after all, we use the same MCMC algorithm to sample from the same posterior distribution.
The histogram over the samples *should* therefore be identical within the limits of
sampling variation. However, if
we plot the samples from each of the three runs separately,
we find that the histograms actually look quite different. For illustrative purposes,
we focus on the absorption rate in the code block.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 1025-1066

We select the samples of the absorption rate from the ``xarray.Dataset`` using
the name of the parameter in the mechanistic model,
``samples['dose.absorption_rate'].values``. The returned object is a numpy array
of shape ``(n_runs, n_iterations)``. We can select the samples from
run n using the index n-1, e.g. for run 1: ``samples['dose.absosption_rate'].values[0]``.
In addition, we subsample the MCMC samples in the above code block, which we can,
for now, regard as a cosmetic choice.

.. raw:: html
   :file: images/3_fitting_models_4.html

We can see that the histograms over the samples from the three runs look very
different. This seems contradictory to the orange distribution which we presented
above as the histogram over the MCMC samples across the runs. But also, how can it
makes sense for the three histograms to differ when the
posterior distribution is the same? -- It does not, and in fact, it tells us that the three histograms in the figure above do not represent the posterior
distribution. Although, we did not disclose it until now, the orange
distribution, respresenting the posterior distribution, is only based on the
second half of each MCMC run!

So why is it crucial that we only choose the second half of each MCMC run,
and is the particular choice of the *second* half important? The answer comes
back to a common limitation of all MCMC algorithm which we can see in the right
column of the figure presented earlier: MCMC algorithms generate samples
from the posterior distribution conditional on the latest generated sample.
For some MCMC algorithms, this conditioning has little influence on sequential samples
because the internal sampling strategy is advanced enough to
sufficiently decorrelate subsequent samples. But for
many MCMC algorithms the conditioned sample substantially influences the sampled value. That
means that if the last samples of two MCMC runs are far away from each other, the following
samples of the runs will also differ substantially from each other, see for example
the first 5000 iterations of the elimination rate in the second row of the figure
in the previous section.

At the start of an MCMC run, there is no "last sampled" parameter value,
and we have to manually provide a starting point to the MCMC algorithm.
The :class:`chi.SamplingController`
automatically samples these starting points from the prior distribution, to
reduce the number of manual steps during the inference in Chi.
This means that the above runs of the MCMC algorithm start from three
different positions in parameter space. These starting points have little
to do with the posterior distribution, and are potentially far away from the area of
the parameter space that is interesting for the posterior distribution.
It therefore makes sense that during the early iterations of the runs, the sampled
values of the MCMC runs do not agree. Fortunately, MCMC algorithms are constructed
in such a way that their samples are, at least in theory, guaranteed to converge
to the posterior distribution, meaning that runs of MCMC algorithms starting from
different areas in parameter space can be expected to converge to the same area.
In practice, numerical limitations may nevertheless
make it impossible, or very time-consuming for samples from MCMC algorithms not
suited to the inference problem to
converge to the posterior distribution. Starting multiple runs from different
areas in parameter space therefore provides an effective way of testing whether
a given MCMC algorithm is suited for the inference of the problem at hand. If
the runs do not converge, the MCMC algorithm either needs more iterations to converge, or
is not suited for the inference problem, in which case the MCMC samples cannot
be expected to represent the posterior distribution. If the runs do converge,
we can be confident that the inference results are correct (although there is always
a chance for edge cases where this convergence test fails to identify the
limitations of an algorithm).

So, starting multiple MCMC runs from different locations in parameter space is
a good idea to gain confidence in the inference results. We recommend running the
MCMC algorithm from 3-5 different starting points, randomly sampled from the prior distribution.
On the one hand, a large number of starting points will test the suitability
of the MCMC algorithm more thoroughly, but each run also comes at a computational
cost. 3-5 runs therefore provide a good tradeoff. In addition, starting points randomly sampled
from the prior usually make sure that the starting points are not too close together,
so that the suitability of the MCMC algorithm is indeed tested. At the same time, it also
ensures that the starting points are not extremely far away from each other, located in areas
of the parameter space that we do not even deem relevant for the inference result.
So, even if the MCMC algorithm would encounter problems in these extreme areas of parameter space,
it may not be very relevant for our inference problem. Randomly sampling from the
prior distribution therefore provides a good tradeoff between well dispersed
and not too extreme starting points.

In conclusion, the initial iterations of the MCMC runs can be used to establish the
validity of the inference results when the starting points of different runs are sufficiently
dispersed. At the same time, these initial samples have little to do
with the posterior distribution, as outlined above.
It is therefore common practice to exclude these early samples
from the inference result. To identify which samples should be discarded, and which
should be included in the result, the earlier presented
"trace plot", visualising the samples of the different runs at each iteration
can be useful. We can see, for example, that somewhere around the 8000s iteration the
three runs of the MCMC algorithm converge. We therefore choose the 10,000s iteration
as the cut-off for the "warm-up" (2000 extra iteration for good measure).
Below, we plot the histogram over the samples
from the three chains, this time using only the samples from the second half of
each run.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 1073-1114

.. raw:: html
   :file: images/3_fitting_models_5.html

We can see that the histograms over the samples are now in much better agreement!

This visual assessment of the convergence already gets us very far, but for those
that would prefer more quantitative metrics to assess the convergence of MCMC
runs, we recommend the open source library ArViz_. ArViz_ implements a number
of widely established metrics to assess the convergence of MCMC algorithms,
including the :math:`\hat{R}` metric and the effective sample size. We can estimate
both values (and many more) using the ``az.summary`` function. In the below code
block, we first estimate the values for all MCMC samples, and then estimate
the values only for the samples post warm-up.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 1121-1130

The return of ``az.summary`` is a ``pandas.DataFrame``, containing the estimated
values, as illustrated below

.. csv-table:: Summary: all samples
   :file: code/3_fitting_models_summary_1.csv
   :widths: 19, 9, 9, 9, 9, 9, 9, 9, 9, 9
   :header-rows: 1

.. csv-table:: Summary: samples post warm-up
   :file: code/3_fitting_models_summary_2.csv
   :widths: 19, 9, 9, 9, 9, 9, 9, 9, 9, 9
   :header-rows: 1

The summary output contains the estimates of the mean value, the standard deviation,
and many other values. For the assessment of the convergence, we would like to focus
on the :math:`\hat{R}` values in the right-most column of the dataframes, the ``r_hat`` columns.
We can see that the summary provides one estimate for each parameter. For the
samples post warm-up, the estimates are all ``1.00``, while the estimates for all
samples assume larger values. Loosely speaking, the :math:`\hat{R}` metric compares
the variance of samples across runs to the variance of samples within runs. If
samples from different runs are very different, the variance across runs is much larger
than the variance of samples within the same run, leading to an estimate :math:`>1`.
On the flip side, if the inter-run variance is the same as the intra-run variance,
the samples across runs look the same and the :math:`\hat{R}` metric returns a
value of :math:`1`. As a result, :math:`\hat{R}` estimates of :math:`1` indicate
convergence, while estimate :math:`>1` indicate that the runs have not yet fully converged.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 701-1018

Assessing convergence: Summary
******************************

1. Run the MCMC algorithm multiple times from different starting points. The :class:`chi.SamplingController` automatically samples starting points from the prior distribution. Recommended number of runs is 3-5.
2. Visually assess the convergence of the runs to the same area of parameter space using trace plots i.e. sample values on the y-axis and iteration at which each sample was obtained on the x-axis.
3. Discard the samples where the runs have not converged as "warm-up". If the runs do not converge, the MCMC algorithm has to run for more iterations until they converge, or the algorithm may not be suited for the inference problem.
4. Calculate the :math:`\hat{R}` value using the remaining samples. We recommend the ``az.rhat`` function implemented in ArViz_. Values :math:`<1.01` indicate good convergence. Larger values may indicate problems with the inference.
5. (Optional) For MCMC runs that require a large number of iterations for convergence, it can make sense to subsample the samples. Often, 1000-3000 samples are sufficient to represent a distribution. For ``n_iterations`` post warm-up, keeping every ``n_iterations // 1000`` th sample may therefore be sufficient.

For more details, please refer to the previous section.

Comparing model fits to data
****************************

Let us conclude this tutorial by comparing our model fit to the drug concentration
measurements. The simplest way to do this is to just focus on the means of the
posterior distributions, which we can extract from the summary dataframes presented
in Section 3.3.2.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 1138-1177

.. raw:: html
   :file: images/3_fitting_models_6.html

In the above code block, we first fix the initial amount parameters of the mechanistic
model using the :class:`chi.ReducedMechanisticModel` class. This class is a wrapper
for mechanistic models that implements the :meth:`ReducedMechanisticModel.fix_parameters`
method. This method works analogously to the :meth:`ProblemModellingController.fix_parameters`
and fixes model parameters to constant values. In the next lines, we extract the
dosing regimen from the :class:`ProblemModellingController`. The controller formats
the dosing regimens from the dataset, Dataset_1_, into a useful format for the :class:`chi.PKPDModel`, returning the
dosing regimens in form of a dictionary, using the
IDs of the individuals as keys. This makes it possible to access the dosing regimen
of the individual with ``'ID 1'`` in Dataset_1_ using the key ``'1'``.
In the remaining lines leading up to the plotting, we extract the
estimates of the parameter means from the summary dataframe.

We can see that our fit describes the measurements reasonably well. However, it is
a little bit unsatisfying that the uncertainty of the parameter estimates is not
reflected in the model fit at all. To resolve this shortcoming of just using the
parameter means to plot the model fit, Chi implements the :class:`chi.PosteriorPredictiveModel`,
which defines the posterior predictive distribution of the measurement
process

.. math::

    p(y | \mathcal{D}, r, t) = \int \mathrm{d} \psi \, p(y | \psi, r, t)\, p(\psi | \mathcal{D}).

The posterior predictive distribution averages our model of the measurement
distribution over the parameter values that we found to be consistent with the data.
Before discussing this in more detail, let us take a look at how the posterior predictive
distribution compares to the model fit where we just used the means of the parameters.
To this end, we sample measurements repeatedly from :math:`p(y | \mathcal{D}, r, t)`
and estimate the mean and some of its percentiles at each time point.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 1184-1206

.. raw:: html
   :file: images/3_fitting_models_7.html

In the first few lines of the code block, we implement the
:class:`chi.PosteriorPredictiveModel` by first retrieving the
:class:`chi.PredictiveModel` from the problem modelling controller, and then
combining it with the inferred posterior samples.
The :class:`chi.PredictiveModel` defines the measurement distribution,
:math:`p(y | \psi, r, t)`, for the purpose of simulating measurements. Note that
:math:`p(y | \psi, r, t)` is the same distribution as the one we used to construct the
likelihood in Section 3.1.
The important difference is that for the likelihood the values of :math:`(y, t, r)`
are fixed to the measurements, while for the :class:`chi.PredictiveModel`, the times
and the dosing regimen can be chosen at will to simulate any measurements of interest.

In the following 2 lines of the code block, we sample measurements from the posterior predictive distribution --
1000 samples at each simulated time point. The samples are returned as a ``pandas.DataFrame``,
which we reshape to a numpy array of shape ``(n_samples, n_times)`` in the next few lines.
This reshaping makes it very easy to calculate the mean value and the percentiles
of the samples at each time point, as demonstrated in the final lines of the code block.

The figure visualises the estimated mean and percentiles of the posterior predictive
distribution on top of our earlier fit. The code to reproduce the figure is documented below.
The area between the percentile estimates marks the region of drug concentration values
which can be expected to be occupied by 90% of all measurements,
assuming, of course, that our model correctly describes the measurement process.
The estimated percentiles are more jittery than the plot with the mean parameter values
because we use sampling to estimate the percentiles. As a result, the estimates
are subject to sampling error, which can be reduced by increasing the number of
samples, ``n_samples``. However, for the purpose of this analysis, the jittering
is not substantial enough to warrant a larger number of samples.

Plotting the posterior predictive distribution provides two important insights
that the fit with the means of the parameters cannot provide: 1. it can quantify
the uncertainty associated with the treatment response. This uncertainty is partially
due to measurement noise, but it also is due to the residual uncertainty in the
model parameters, as quantified by the posterior distribution. 2. It is
able to better reflect nonlinearities between model parameters and the model output
which are otherwise neglected when plotting the fit with the means of the model parameters.
We can see an indication of this in the figure by
noting that the fit with the parameter means is not the same as the mean of the
posterior predictive distribution. This difference is a result of the, in general,
nonlinear dependence of treatment response models on their parameters, meaning that the average
treatment response under parametric uncertainty is **not** the same as the treatment response
predicted with the mean parameters. In fact, these two predictions can differ substantially from
each other, especially when predictions are made for future times or for previously
unexplored dosing regimens.

.. literalinclude:: code/3_fitting_models_2.py
    :lines: 1208-1256

This concludes this tutorial. If you have any feedback or suggestions for
improvement, or would like to report any typos, mistakes or bugs, please do reach out to us, for example
by creating an Issue_. We are looking
forward to hearing from you!

Reference to ErrorModel, LogPDF and PredictiveModel API
*******************************************************

.. autosummary::

    chi.ErrorModel
    chi.GaussianErrorModel
    chi.LogNormalErrorModel
    chi.MultiplicativeGaussianErrorModel
    chi.ConstantAndMultiplicativeGaussianErrorModel
    chi.ReducedErrorModel
    chi.LogLikelihood
    chi.LogPosterior
    chi.ProblemModellingController
    chi.SamplingController
    chi.PredictiveModel
    chi.PosteriorPredictiveModel