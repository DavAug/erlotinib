#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import warnings

import numpy as np
import pandas as pd
import pints
from tqdm.notebook import tqdm
import xarray as xr

import erlotinib as erlo


class InferenceController(object):
    """
    A base class for inference controllers.

    Parameters
    ----------

    log_posterior
        An instance of a :class:`LogPosterior` or a list of
        :class:`LogPosterior` instances. If multiple log-posteriors are
        provided, they have to be defined on the same parameter space.
    """

    def __init__(self, log_posterior):
        super(InferenceController, self).__init__()

        # Convert log-posterior to a list of log-posteriors
        try:
            log_posteriors = list(log_posterior)
        except TypeError:
            # If log-posterior cannot be converted to a list, it likely means
            # that there is only one log-posterior
            log_posteriors = [log_posterior]

        for log_posterior in log_posteriors:
            if not isinstance(
                    log_posterior,
                    (erlo.LogPosterior, erlo.HierarchicalLogPosterior)):
                raise ValueError(
                    'The log-posterior has to be an instance of a '
                    'erlotinib.LogPosterior or a '
                    'erlotinib.HierarchicalLogPosterior')

        # Check that the log-posteriors have the same number of parameters
        n_parameters = log_posteriors[0].n_parameters()
        for log_posterior in log_posteriors:
            if log_posterior.n_parameters() != n_parameters:
                raise ValueError(
                    'All log-posteriors have to be defined on the same '
                    'parameter space.')

        self._log_posteriors = log_posteriors
        self._log_prior = self._log_posteriors[0].get_log_prior()

        # Set defaults
        self._n_runs = 5
        self._parallel_evaluation = True
        self._transform = None

        # Get parameter names and number of parameters
        self._parameters = list(
            self._log_posteriors[0].get_parameter_names())
        self._n_parameters = self._log_posteriors[0].n_parameters()

        # Sample initial parameters from log-prior
        n_posteriors = len(self._log_posteriors)
        self._initial_params = np.empty(shape=(
            n_posteriors,
            self._n_runs,
            self._n_parameters))
        self._sample_initial_parameters()

    def _sample_initial_parameters(self):
        """
        Sample initial parameter values for inference runs from prior.

        If the underlying model has a hierarchical structure, the population
        model parameters are randomly sampled from the prior, and the
        individual parameters from the resulting population models. This avoids
        numerical instabilities from starting off with very bad initial
        parameters.
        """
        for index, log_posterior in enumerate(self._log_posteriors):
            # Construct a mask for the top-level parameters
            mask = np.ones(shape=self._n_parameters, dtype=bool)
            all_parameters = log_posterior.get_parameter_names()
            try:
                top_parameters = log_posterior.get_parameter_names(
                    exclude_bottom_level=True)
            except TypeError:
                # Flag does not exist for non-hierarchical log-posteriors
                top_parameters = all_parameters

            for param_id, parameter in enumerate(all_parameters):
                if parameter not in top_parameters:
                    # Flip mask entry to False
                    mask[param_id] = False

            # Sample initial top-level parameters from prior
            initial_params = self._initial_params[index]
            initial_params[:, mask] = self._log_prior.sample(
                self._n_runs)

            # Sample initial population, if model is hierarchical
            if isinstance(log_posterior, erlo.HierarchicalLogPosterior):
                self._initial_params[index] = self._sample_population(
                        index, log_posterior, mask)

    def _sample_population(self, index, log_posterior, mask):
        """
        Samples population for initial population model parameters.

        index: The index of the log-posterior
        log-posterior: The HierarchcalLogPosterior
        mask: A boolean mask which carries True for top-level parameters
            and False for bottom-level parameters.
        """
        # Get number of likelihoods and population models
        log_likelihood = log_posterior.get_log_likelihood()
        n_ids = log_likelihood.n_log_likelihoods()
        population_models = log_likelihood.get_population_models()

        # Create container for samples
        # (with the population parameter samples)
        container = self._initial_params[index]

        # Sample individuals from population model for each run
        start_index = 0
        for pop_model in population_models:
            # Get number of individual and population parameters
            n_indiv, n_pop = pop_model.n_hierarchical_parameters(n_ids)

            # If number of bottom-level parameters is 0, skip to next iteration
            end_index = start_index + n_indiv
            if (n_indiv == 0) or np.alltrue(mask[start_index:end_index]):
                # Shift start index by total number of hierarchical parameters
                start_index += n_indiv + n_pop
                continue

            # Get population parameters
            # (always trailing parameters in a population model)
            start = start_index + n_indiv
            end = start + n_pop
            pop_parameters = self._initial_params[index, :, start:end]

            # Substitude individual parameters by population samples
            start = start_index
            end = start + n_indiv
            for run_id, pop_params in enumerate(pop_parameters):
                sample = pop_model.sample(pop_params, n_indiv)
                container[run_id, start:end] = sample

            # Shift start_index by total number of hierarchical parameters
            start_index += n_indiv + n_pop

        return container

    def set_n_runs(self, n_runs):
        """
        Sets the number of times the inference routine is run.

        Each run starts from a random sample of the log-prior.
        """
        self._n_runs = int(n_runs)

        # Sample initial parameters from log-prior
        self._initial_params = np.empty(shape=(
            len(self._log_posteriors),
            self._n_runs,
            self._n_parameters))
        self._sample_initial_parameters()

    def set_parallel_evaluation(self, run_in_parallel):
        """
        Enables or disables parallel evaluation using either a
        :class:`pints.ParallelEvaluator` or a
        :class:`pints.SequentialEvaluator`.

        If ``run_in_parallel=True``, the method will run using a number of
        worker processes equal to the detected CPU core count. The number of
        workers can be set explicitly by setting ``run_in_parallel`` to an
        integer greater than ``0``. Parallelisation can be disabled by setting
        ``run_in_parallel`` to ``0`` or ``False``.
        """
        if not isinstance(run_in_parallel, (bool, int)):
            raise ValueError(
                '`run_in_parallel` has to a boolean or an integer.')
        if run_in_parallel < 0:
            raise ValueError(
                '`run_in_parallel` cannot be negative.')

        self._parallel_evaluation = run_in_parallel

    def set_transform(self, transform):
        """
        Sets the transformation that transforms the parameter space into the
        search space.

        Transformations of the search space can significantly improve the
        performance of the inference routine.

        ``transform`` has to be an instance of `pints.Transformation` and must
        have the same dimension as the parameter space.
        """
        if not isinstance(transform, pints.Transformation):
            raise ValueError(
                'Transform has to be an instance of `pints.Transformation`.')
        if transform.n_parameters() != self._n_parameters:
            raise ValueError(
                'The dimensionality of the transform does not match the '
                'dimensionality of the log-posterior.')
        self._transform = transform


class OptimisationController(InferenceController):
    """
    Sets up an optimisation routine that attempts to find the parameter values
    that maximise a :class:`pints.LogPosterior`. If multiple log-posteriors are
    provided, the posteriors are assumed to be structurally identical and only
    differ due to different data sources.

    By default the optimisation is run 5 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.

    Parameters
    ----------

    log_posterior
        An instance of a :class:`LogPosterior` or a list of
        :class:`LogPosterior` instances. If multiple log-posteriors are
        provided, they have to be defined on the same parameter space.
    """

    def __init__(self, log_posterior):
        super(OptimisationController, self).__init__(log_posterior)

        # Set default optimiser
        self._optimiser = pints.CMAES

    def run(
            self, n_max_iterations=10000, show_id_progress_bar=False,
            show_run_progress_bar=False, log_to_screen=False):
        """
        Runs the optimisation and returns the maximum a posteriori probability
        parameter estimates in from of a :class:`pandas.DataFrame` with the
        columns 'ID', 'Parameter', 'Estimate', 'Score' and 'Run'.

        The number of maximal iterations of the optimisation routine can be
        limited by setting ``n_max_iterations`` to a finite, non-negative
        integer value.

        Parameters
        ----------

        n_max_iterations
            The maximal number of optimisation iterations to find the MAP
            estimates for each log-posterior. By default the maximal number
            of iterations is set to 10000.
        show_id_progress_bar
            A boolean flag which indicates whether a progress bar for looping
            through the individual log-posteriors is displayed.
        show_run_progress_bar
            A boolean flag which indicates whether a progress bar for looping
            through the optimisation runs is displayed.
        log_to_screen
            A boolean flag which indicates whether the optimiser logging output
            is displayed.
        """

        # Initialise result dataframe
        result = pd.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])

        # Initialise intermediate container for individual runs
        run_result = pd.DataFrame(
            columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])
        run_result['Parameter'] = self._parameters

        # Get posterior
        for posterior_id, log_posterior in enumerate(tqdm(
                self._log_posteriors, disable=not show_id_progress_bar)):
            individual_result = pd.DataFrame(
                columns=['ID', 'Parameter', 'Estimate', 'Score', 'Run'])

            # Set ID of individual (or IDs of parameters, if hierarchical)
            run_result['ID'] = log_posterior.get_id()

            # Run optimisation multiple times
            for run_id in tqdm(
                    range(self._n_runs), disable=not show_run_progress_bar):
                opt = pints.OptimisationController(
                    function=log_posterior,
                    x0=self._initial_params[posterior_id, run_id, :],
                    method=self._optimiser,
                    transform=self._transform)

                # Configure optimisation routine
                opt.set_log_to_screen(log_to_screen)
                opt.set_max_iterations(iterations=n_max_iterations)
                opt.set_parallel(self._parallel_evaluation)

                # Find optimal parameters
                try:
                    estimates, score = opt.run()
                except Exception:
                    # If inference breaks fill estimates with nan
                    estimates = [np.nan] * self._n_parameters
                    score = np.nan

                # Save estimates and score of runs
                run_result['Estimate'] = estimates
                run_result['Score'] = score
                run_result['Run'] = run_id + 1
                individual_result = individual_result.append(run_result)

            # Save runs for individual
            result = result.append(individual_result)

        return result

    def set_optimiser(self, optimiser):
        """
        Sets method that is used to find the maximum a posteiori probability
        estimates.
        """
        if not issubclass(optimiser, pints.Optimiser):
            raise ValueError(
                'Optimiser has to be a `pints.Optimiser`.')
        self._optimiser = optimiser


class SamplingController(InferenceController):
    """
    Sets up a sampling routine that attempts to find the posterior
    distribution of parameters defined by a :class:`pints.LogPosterior`. If
    multiple log-posteriors are provided, the posteriors are assumed to be
    structurally identical and only differ due to different data sources.

    By default the sampling is run 5 times from different initial
    starting points. Starting points are randomly sampled from the
    specified :class:`pints.LogPrior`. The optimisation is run by default in
    parallel using :class:`pints.ParallelEvaluator`.

    Extends :class:`InferenceController`.
    """

    def __init__(self, log_posterior):
        super(SamplingController, self).__init__(log_posterior)

        # Set default sampler
        self._sampler = pints.HaarioACMC
        self._divergent_iterations = None

    def _format_chains(self, chains, names, ids):
        """
        Formats the chains generated by pints in shape of
        (n_chains, n_iterations, n_parameters) to a xarray.Dataset, where
        each parameter name gets an entry yielding a xarray.DataArray of
        shape (n_chains, n_iterations, n_ids) or if ID is None
        (n_chains, n_iterations).

        Note that the naming of the dimensions matter for ArviZ so we call
        the dimensions (chain, draw, individual) (the last dimension
        is not set by ArviZ).

        Parameters
        ----------
        chains
            np.ndarray in shape (n_chains, n_iterations, n_parameters)
        names
            List of length n_parameters. Names may not be unique
        ids
            Str or list of str of length n_parameters. IDs are ``None`` for
            population parameters.
        """
        # Broadcast IDs to length of names, if posterior has only one ID
        if isinstance(ids, str):
            ids = np.broadcast_to(ids, shape=len(names))

        # Convert names and ids to numpy arrays
        ids = np.asarray(ids)
        names = np.asarray(names)

        # Get the coordinates for the chains and draws
        n_chains, n_draws, _ = chains.shape
        chain_coords = list(range(n_chains))
        draw_coords = list(range(n_draws))

        # Sort samples of parameters into xarrays
        container = {}
        parameter_names = np.unique(names)
        for parameter in parameter_names:
            # Get IDs and chains associated to parameter
            mask = names == parameter
            parameter_ids = ids[mask]
            parameter_chains = chains[:, :, mask]

            # If parameter is a population parameter (ID is None), save xarray
            # without individual dimension.
            is_population_param = (
                len(parameter_ids) == 1) and (parameter_ids[0] is None)
            if is_population_param is True:
                parameter_chains = xr.DataArray(
                    data=parameter_chains[:, :, 0],
                    dims=['chain', 'draw'],
                    coords={'chain': chain_coords, 'draw': draw_coords})

            # The parameter is a bottom-level parameter, so individual
            # information is important
            else:
                parameter_chains = xr.DataArray(
                    data=parameter_chains,
                    dims=['chain', 'draw', 'individual'],
                    coords={
                        'chain': chain_coords,
                        'draw': draw_coords,
                        'individual': list(parameter_ids)})

            # Add DataArray to container
            container[parameter] = parameter_chains

        return xr.Dataset(container)

    def _get_id_parameter_pairs(self, log_posterior):
        """
        Returns a zipped list of ID (pop_prefix), and parameter name pairs.

        Posteriors that are not derived from a HierarchicalLoglikelihood carry
        typically only a single ID (the ID of the individual they are
        modelling). In that case all parameters are assigned with the same ID.

        For posteriors that are derived from a HierarchicalLoglikelihood it
        often makes sense to label the parameters with different IDs. These ID
        parameter name pairs are reconstructed here.
        """
        # Get IDs and parameter names
        ids = log_posterior.get_id()
        parameters = log_posterior.get_parameter_names()

        # If IDs is only one ID, expand to list of length n_parameters
        if not isinstance(ids, list):
            n_parameters = len(parameters)
            ids = [ids] * n_parameters

        return zip(ids, parameters)

    def get_divergencent_iterations(self):
        """
        Returns the number of divergent trajectories for Hamiltonian Monte
        Carlo (HMC) samplers, or ``None``.

        HMC samplers leverage geometric properties of the log-posterior
        surface to efficiently generate less correlated samples. This
        exploitation of the geometry has the draw back that it can go wrong
        in areas of parameter space with very large curvature. When that
        happens, the posterior samples will no longer converge to the true
        posterior distribution. The number of divergent trajectories
        """
        return self._divergent_iterations

    def run(
            self, n_iterations=10000, hyperparameters=None,
            show_progress_bar=False, log_to_screen=False):
        """
        Runs the sampling routine and returns the sampled parameter values in
        form of a :class:`xarray.Dataset` with :class:`xarray.DataArray`
        instances for each parameter.

        If multiple posteriors are inferred a list of :class:`xarray.Dataset`
        instances is returned.

        The number of iterations of the sampling routine can be set by setting
        ``n_iterations`` to a finite, non-negative integer value. By default
        the routines run for 10000 iterations.

        :param n_iterations: A non-negative integer number which sets the
            number of iterations of the MCMC runs.
        :type n_iterations: int, optional
        :param hyperparameters: A list of hyperparameters for the sampling
            method. If ``None`` the default hyperparameters are set.
        :type hyperparameters: list[float], optional
        :param show_progress_bar: A boolean flag which indicates for how
            many posteriors the inference has been completed.
        :type show_progress_bar: bool, optional
        :param log_to_screen: A boolean flag which can be used to print
            the progress of the runs to the screen. The progress is printed
            every 500 iterations.
        :type log_to_screen: bool, optional
        """
        # Sample from the individual log_posteriors
        posterior_samples = []
        divergent_iters = []
        for posterior_id, log_posterior in enumerate(tqdm(
                self._log_posteriors, disable=not show_progress_bar)):
            # Set up sampler
            sampler = pints.MCMCController(
                log_pdf=log_posterior,
                chains=self._n_runs,
                x0=self._initial_params[posterior_id, ...],
                method=self._sampler,
                transform=self._transform)

            # Configure sampling routine
            sampler.set_log_to_screen(log_to_screen)
            sampler.set_log_interval(iters=20, warm_up=3)
            sampler.set_max_iterations(iterations=n_iterations)
            sampler.set_parallel(self._parallel_evaluation)

            if hyperparameters is not None:
                for s in sampler.samplers():
                    s.set_hyper_parameters(hyperparameters)

            # Run sampling routine
            chains = sampler.run()

            # Format chains
            names = self._parameters
            ids = log_posterior.get_id()
            chains = self._format_chains(chains, names, ids)

            # Append chains to container
            posterior_samples.append(chains)

            # If Hamiltonian Monte Carlo, get number of divergent
            # iterations
            if issubclass(
                    self._sampler, (pints.HamiltonianMCMC, pints.NoUTurnMCMC)):
                divergent_iters.append(
                    [s.divergent_iterations() for s in sampler.samplers()])

        # Remember divergences
        self._divergent_iterations = None
        if divergent_iters:
            self._divergent_iterations = divergent_iters

        # If only one posterior is run, remove padding list
        if len(posterior_samples) == 1:
            return posterior_samples[0]

        return posterior_samples

    def set_initial_parameters(
            self, data, id_key='ID', param_key='Parameter', est_key='Estimate',
            score_key='Score', run_key='Run'):
        """
        Sets the initial parameter values of the MCMC runs to the parameter set
        with the maximal a posteriori probability across a number of parameter
        sets.

        This method is intended to be used in conjunction with the results of
        the :class:`OptimisationController`.

        It expects a :class:`pandas.DataFrame` with the columns 'ID',
        'Parameter', 'Estimate', 'Score' and 'Run'. The maximum a posteriori
        probability values across all estimates is determined and used as
        initial point for the MCMC runs.

        If multiple parameter sets assume the maximal a posteriori probability
        value, a parameter set is drawn randomly from them.

        Parameters
        ----------
        data
            A :class:`pandas.DataFrame` with the parameter estimates in form of
            a parameter, estimate and score column.
        id_key
            Key label of the :class:`DataFrame` which specifies the individual
            ID column. Defaults to ``'ID'``.
        param_key
            Key label of the :class:`DataFrame` which specifies the parameter
            name column. Defaults to ``'Parameter'``.
        est_key
            Key label of the :class:`DataFrame` which specifies the parameter
            estimate column. Defaults to ``'Estimate'``.
        score_key
            Key label of the :class:`DataFrame` which specifies the score
            estimate column. The score refers to the maximum a posteriori
            probability associated with the estimate. Defaults to ``'Score'``.
        run_key
            Key label of the :class:`DataFrame` which specifies the
            optimisation run column. Defaults to ``'Run'``.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                'Data has to be pandas.DataFrame.')

        for key in [id_key, param_key, est_key, score_key, run_key]:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Convert dataframe IDs and parameter names to strings
        data = data.astype({id_key: str, param_key: str})

        # Get posterior IDs (one posterior may have multiple IDs, one
        # for each parameter)
        for index, log_posterior in enumerate(self._log_posteriors):

            # Get MAP for each parameter of log_posterior
            for prefix, parameter in self._get_id_parameter_pairs(
                    log_posterior):

                # Get estimates for ID (prefix)
                mask = data[id_key] == prefix
                individual_data = data[mask]

                # If ID (prefix) doesn't exist, move on to next iteration
                if individual_data.empty:
                    warnings.warn(
                        'The log-posterior ID <' + str(prefix) + '> could not'
                        ' be identified in the dataset, and was therefore '
                        'not set to a specific value.')

                    continue

                # Among estimates for this ID (prefix), get the relevant
                # parameter
                mask = individual_data[param_key] == parameter
                individual_data = individual_data[mask]

                # If parameter with this ID (prefix) doesn't exist, move on to
                # next iteration
                if individual_data.empty:
                    warnings.warn(
                        'The parameter <' + str(parameter) + '> with ID '
                        '<' + str(prefix) + '> could not be identified in the '
                        'dataset, and was therefore not set to a specific '
                        'value.')

                    continue

                # Get estimates with maximum a posteriori probability
                max_prob = individual_data[score_key].max()
                mask = individual_data[score_key] == max_prob
                individual_data = individual_data[mask]

                # Find a unique set of parameter values
                runs = individual_data[run_key].unique()
                selected_param_set = np.random.choice(runs)
                mask = individual_data[run_key] == selected_param_set
                individual_data = individual_data[mask]

                # Create mask for parameter position in log-posterior
                ids = log_posterior.get_id()
                if not isinstance(ids, list):
                    n_parameters = len(self._parameters)
                    ids = [ids] * n_parameters
                id_mask = np.array(ids) == prefix
                param_mask = np.array(self._parameters) == parameter
                mask = id_mask & param_mask

                # Set initial parameters across runs to map estimate
                map_estimate = individual_data[est_key].to_numpy()
                self._initial_params[index, :, mask] = map_estimate

    def set_sampler(self, sampler):
        """
        Sets method that is used to sample from the log-posterior.
        """
        if not issubclass(sampler, pints.MCMCSampler):
            raise ValueError(
                'Sampler has to be a `pints.MCMCSampler`.'
            )
        self._sampler = sampler
