#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#
# The InverseProblem class is based on the SingleOutputProblem and
# MultiOutputProblem classes of PINTS (https://github.com/pints-team/pints/),
# which is distributed under the BSD 3-clause license.
#

import copy

import myokit
import numpy as np
import pandas as pd
import pints

import erlotinib as erlo


class InverseProblem(object):
    """
    Represents an inference problem where a model is fit to a
    one-dimensional or multi-dimensional time series, such as measured in a
    PKPD study.

    Parameters
    ----------
    model
        An instance of a :class:`MechanisticModel`.
    times
        A sequence of points in time. Must be non-negative and increasing.
    values
        A sequence of single- or multi-valued measurements. Must have shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of points in
        ``times`` and ``n_outputs`` is the number of outputs in the model. For
        ``n_outputs = 1``, the data can also have shape ``(n_times, )``.
    """

    def __init__(self, model, times, values):

        # Check model
        if not isinstance(model, erlo.MechanisticModel):
            raise ValueError(
                'Model has to be an instance of a erlotinib.Model.'
            )
        self._model = model

        # Check times, copy so that they can no longer be changed and set them
        # to read-only
        self._times = pints.vector(times)
        if np.any(self._times < 0):
            raise ValueError('Times cannot be negative.')
        if np.any(self._times[:-1] > self._times[1:]):
            raise ValueError('Times must be increasing.')

        # Check values, copy so that they can no longer be changed
        values = np.asarray(values)
        if values.ndim == 1:
            np.expand_dims(values, axis=1)
        self._values = pints.matrix2d(values)

        # Check dimensions
        self._n_parameters = int(model.n_parameters())
        self._n_outputs = int(model.n_outputs())
        self._n_times = len(self._times)

        # Check for correct shape
        if self._values.shape != (self._n_times, self._n_outputs):
            raise ValueError(
                'Values array must have shape `(n_times, n_outputs)`.')

    def evaluate(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values as a NumPy array of shape ``(n_times, n_outputs)``.
        """
        output = self._model.simulate(parameters, self._times)

        # The erlotinib.Model.simulate method returns the model output as
        # (n_outputs, n_times). We therefore need to transponse the result.
        return output.transpose()

    def evaluateS1(self, parameters):
        """
        Runs a simulation using the given parameters, returning the simulated
        values.
        The returned data is a tuple of NumPy arrays ``(y, y')``, where ``y``
        has shape ``(n_times, n_outputs)``, while ``y'`` has shape
        ``(n_times, n_outputs, n_parameters)``.
        *This method only works for problems whose model implements the
        :class:`ForwardModelS1` interface.*
        """
        raise NotImplementedError

    def n_outputs(self):
        """
        Returns the number of outputs for this problem.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the dimension (the number of parameters) of this problem.
        """
        return self._n_parameters

    def n_times(self):
        """
        Returns the number of sampling points, i.e. the length of the vectors
        returned by :meth:`times()` and :meth:`values()`.
        """
        return self._n_times

    def times(self):
        """
        Returns this problem's times.
        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._times

    def values(self):
        """
        Returns this problem's values.
        The returned value is a read-only NumPy array of shape
        ``(n_times, n_outputs)``, where ``n_times`` is the number of time
        points and ``n_outputs`` is the number of outputs.
        """
        return self._values


class ProblemModellingController(object):
    """
    A problem modelling controller which simplifies the model building process
    of a pharmacokinetic and pharmacodynamic problem.

    The class is instantiated with an instance of a :class:`MechanisticModel`
    and one instance of an :class:`ErrorModel` for each mechanistic model
    output.

    Parameters
    ----------
    mechanistic_model
        An instance of a :class:`MechanisticModel`.
    error_models
        A list of :class:`ErrorModel` instances. One error model has to be
        provided for each mechanistic model output.
    outputs
        A list of mechanistic model output names, which can be used to map
        the error models to mechanistic model outputs. If ``None``, the
        error models are assumed to be ordered in the same order as
        :meth:`MechanisticModel.outputs`.
    """

    def __init__(self, mechanistic_model, error_models, outputs=None):
        super(ProblemModellingController, self).__init__()

        # Check inputs
        if not isinstance(mechanistic_model, erlo.MechanisticModel):
            raise TypeError(
                'The mechanistic model has to be an instance of a '
                'erlotinib.MechanisticModel.')

        if not isinstance(error_models, list):
            error_models = [error_models]

        for error_model in error_models:
            if not isinstance(error_model, erlo.ErrorModel):
                raise TypeError(
                    'Error models have to be instances of a '
                    'erlotinib.ErrorModel.')

        # Copy mechanistic model
        mechanistic_model = copy.deepcopy(mechanistic_model)

        # Set outputs
        if outputs is not None:
            mechanistic_model.set_outputs(outputs)

        # Get number of outputs
        n_outputs = mechanistic_model.n_outputs()

        if len(error_models) != n_outputs:
            raise ValueError(
                'Wrong number of error models. One error model has to be '
                'provided for each mechanistic error model.')

        # Copy error models
        error_models = [copy.copy(error_model) for error_model in error_models]

        # Remember models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Set defaults
        self._population_models = None
        self._log_prior = None
        self._data = None
        self._dosing_regimens = None

        # Set parameter names and number of parameters
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

    def _clean_data(self, dose_key, dose_duration_key):
        """
        Makes sure that the data is formated properly.

        1. ids are strings
        2. time are numerics or NaN
        3. biomarkers are strings
        4. measurements are numerics or NaN
        5. dose are numerics or NaN
        6. duration are numerics or NaN
        """
        # Create container for data
        columns = [
            self._id_key, self._time_key, self._biom_key, self._meas_key]
        if dose_key is not None:
            columns += [dose_key]
        if dose_duration_key is not None:
            columns += [dose_duration_key]
        data = pd.DataFrame(columns=columns)

        # Convert IDs to strings
        data[self._id_key] = self._data[self._id_key].astype(
            "string")

        # Convert times to numerics
        data[self._time_key] = pd.to_numeric(self._data[self._time_key])

        # Convert biomarkers to strings
        data[self._biom_key] = self._data[self._biom_key].astype(
            "string")

        # Convert measurements to numerics
        data[self._meas_key] = pd.to_numeric(self._data[self._meas_key])

        # Convert dose to numerics
        if dose_key is not None:
            data[dose_key] = pd.to_numeric(
                self._data[dose_key])

        # Convert duration to numerics
        if dose_duration_key is not None:
            data[dose_duration_key] = pd.to_numeric(
                self._data[dose_duration_key])

        self._data = data

    def _create_log_likelihoods(self):
        """
        Returns a dict of log-likelihoods, one for each individual in the
        dataset. The keys are the individual IDs and the values are the
        log-likelihoods.
        """
        # Create a likelihood for each individual
        log_likelihoods = []
        for individual in self._ids:
            # Set dosing regimen
            try:
                self._mechanistic_model.simulator.set_protocol(
                    self._dosing_regimens[individual])
            except TypeError:
                # TypeError is raised when applied regimens is still None,
                # i.e. no doses were defined by the datasets.
                pass

            log_likelihoods.append(self._create_log_likelihood(individual))

        return log_likelihoods

    def _create_log_likelihood(self, individual):
        """
        Gets the relevant data for the individual and returns the resulting
        erlotinib.LogLikelihood.
        """
        # Get individuals data
        times = []
        observations = []
        mask = self._data[self._id_key] == individual
        data = self._data[mask][
            [self._time_key, self._biom_key, self._meas_key]]
        for output in self._mechanistic_model.outputs():
            # Mask data for biomarker
            biomarker = self._output_biomarker_dict[output]
            mask = data[self._biom_key] == biomarker
            temp_df = data[mask]

            # Filter times and observations for non-NaN entries
            mask = temp_df[self._meas_key].notnull()
            temp_df = temp_df[[self._time_key, self._meas_key]][mask]
            mask = temp_df[self._time_key].notnull()
            temp_df = temp_df[mask]

            # Collect data for output
            times.append(temp_df[self._time_key].to_numpy())
            observations.append(temp_df[self._meas_key].to_numpy())

        # Create log-likelihood and set ID to individual
        log_likelihood = erlo.LogLikelihood(
            self._mechanistic_model, self._error_models, observations, times)
        log_likelihood.set_id(individual)

        return log_likelihood

    def _extract_dosing_regimens(self, dose_key, duration_key):
        """
        Converts the dosing regimens defined by the pandas.DataFrame into
        myokit.Protocols, and returns them as a dictionary with individual
        IDs as keys, and regimens as values.

        For each dose entry in the dataframe a dose event is added
        to the myokit.Protocol. If the duration of the dose is not provided
        a bolus dose of duration 0.01 time units is assumed.
        """
        # Create duration column if it doesn't exist and set it to default
        # bolus duration of 0.01
        if duration_key is None:
            duration_key = 'Duration in base time unit'
            self._data[duration_key] = 0.01

        # Extract regimen from dataset
        regimens = dict()
        for label in self._ids:
            # Filter times and dose events for non-NaN entries
            mask = self._data[self._id_key] == label
            data = self._data[
                [self._time_key, dose_key, duration_key]][mask]
            mask = data[dose_key].notnull()
            data = data[mask]
            mask = data[self._time_key].notnull()
            data = data[mask]

            # Add dose events to dosing regimen
            regimen = myokit.Protocol()
            for _, row in data.iterrows():
                # Set duration
                duration = row[duration_key]
                if np.isnan(duration):
                    # If duration is not provided, we assume a bolus dose
                    # which we approximate by 0.01 time_units.
                    duration = 0.01

                # Compute dose rate and set regimen
                dose_rate = row[dose_key] / duration
                time = row[self._time_key]
                regimen.add(myokit.ProtocolEvent(dose_rate, time, duration))

            regimens[label] = regimen

        return regimens

    def _get_number_and_parameter_names(self, exclude_pop_model=False):
        """
        Returns the number and names of the log-likelihood.

        The parameters of the HierarchicalLogLikelihood depend on the
        data, and the population model. So unless both are set, the
        parameters will reflect the parameters of the individual
        log-likelihoods.
        """
        # Get mechanistic model parameters
        parameter_names = self._mechanistic_model.parameters()

        # Get error model parameters
        n_outputs = self._mechanistic_model.n_outputs()
        outputs = self._mechanistic_model.outputs()
        for output_id, error_model in enumerate(self._error_models):
            # Get original parameter names
            names = error_model.get_parameter_names()

            # Prepend output name for multi-output problem
            if n_outputs > 1:
                output = outputs[output_id]
                names = [output + ' ' + name for name in names]

            parameter_names += names

        # Stop here if no population model or data has been set
        if (self._population_models is None) or (self._data is None) or (
                exclude_pop_model is True):
            # Get number of parameters
            n_parameters = len(parameter_names)

            return (n_parameters, parameter_names)

        # Construct population parameter names
        pop_parameter_names = []
        n_ids = len(self._ids)
        for param_id, pop_model in enumerate(self._population_models):
            # Get mechanistic/error model parameter name
            name = parameter_names[param_id]

            # Create names for individual parameters
            n_indiv, _ = pop_model.n_hierarchical_parameters(n_ids)
            if n_indiv > 0:
                # If individual parameters are relevant for the hierarchical
                # model, append them
                names = ['ID %s: %s' % (n, name) for n in self._ids]
                pop_parameter_names += names

            # Create names for population-level parameters
            if pop_model.n_parameters() > 0:
                top_names = pop_model.get_parameter_names()
                names = [
                    '%s %s' % (pop_prefix, name) for pop_prefix in top_names]
                pop_parameter_names += names

        # Get number of parameters
        n_parameters = len(pop_parameter_names)

        return (n_parameters, pop_parameter_names)

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        Fixing model parameters resets the population models and the log-prior
        to ``None``.

        Parameters
        ----------
        name_value_dict
            A dictionary with model parameters as keys, and the value to be
            fixed at as values.
        """
        # Check type of dictionanry
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        # Get submodels
        mechanistic_model = self._mechanistic_model
        error_models = self._error_models

        # Convert models to reduced models
        if not isinstance(mechanistic_model, erlo.ReducedMechanisticModel):
            mechanistic_model = erlo.ReducedMechanisticModel(mechanistic_model)
        for model_id, error_model in enumerate(error_models):
            if not isinstance(error_model, erlo.ReducedErrorModel):
                error_models[model_id] = erlo.ReducedErrorModel(error_model)

        # Fix model parameters
        mechanistic_model.fix_parameters(name_value_dict)
        for error_model in error_models:
            error_model.fix_parameters(name_value_dict)

        # If no parameters are fixed, get original model back
        if mechanistic_model.n_fixed_parameters() == 0:
            mechanistic_model = mechanistic_model.mechanistic_model()

        for model_id, error_model in enumerate(error_models):
            if error_model.n_fixed_parameters() == 0:
                error_model = error_model.get_error_model()
                error_models[model_id] = error_model

        # Safe reduced models
        self._mechanistic_model = mechanistic_model
        self._error_models = error_models

        # Update names and number of parameters
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

    def get_dosing_regimens(self):
        """
        Returns a dictionary of dosing regimens in form of myokit.Protocols.

        The dosing regimens are extracted from the dataset if a dose key is
        provided. If no dose key is provided ``None`` is returned.
        """
        return self._dosing_regimens

    def get_log_posteriors(self):
        """
        Returns a list of :class:`LogPosterior` instances, defined by
        the dataset, the mechanistic model, the error model, the log-prior,
        and optionally the population model and the fixed model parameters.

        If a population model has been set, the list will contain only a single
        log-posterior for the populational inference. If no population model
        has been set, the list contains a log-posterior for each individual
        separately.

        This method raises an error if the mechanistic model, the error
        model, or the log-prior has not been set. They can be set with
        :meth:`set_mechanistic_model`, :meth:`set_error_model` and
        :meth:`set_log_prior`.
        """
        if self._log_prior is None:
            raise ValueError(
                'The log-prior has not been set.')

        # Create log-likelihoods
        log_likelihoods = self._create_log_likelihoods()
        if self._population_models is not None:
            # Compose HierarchicalLogLikelihoods
            log_likelihoods = [erlo.HierarchicalLogLikelihood(
                log_likelihoods, self._population_models)]

        # Compose the log-posteriors
        log_posteriors = []
        for log_likelihood in log_likelihoods:
            log_posterior = erlo.LogPosterior(log_likelihood, self._log_prior)
            log_posteriors.append(log_posterior)

        return log_posteriors

    def get_n_parameters(self, exclude_pop_model=False):
        """
        Returns the number of free parameters of the structural model, i.e. the
        mechanistic model, the error model and, if set, the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the number of model parameters.

        If the mechanistic model or the error model have not been set, ``None``
        is returned. If the population model has not been set, only the number
        of parameters for one structural model is returned, as the models are
        structurally the same across individuals.

        If a population model has been set and not all parameters are pooled,
        the mechanistic and error model parameters are counted multiple times
        (once for each individual). To get the number of mechanistic and error
        model parameters prior to setting a population model the
        ``exlude_pop_model`` flag can be set to ``True``.
        """
        if exclude_pop_model is True:
            n_parameters, _ = self._get_number_and_parameter_names(
                exclude_pop_model=True)
            return n_parameters

        return self._n_parameters

    def get_parameter_names(self, exclude_pop_model=False):
        """
        Returns the names of the free structural model parameters, i.e. the
        free parameters of the mechanistic model, the error model and
        optionally the population model.

        Any parameters that have been fixed to a constant value will not be
        included in the list of model parameters.

        If the mechanistic model or the error model have not been set, ``None``
        is returned. If the population model has not been set, only the names
        of parameters for one structural model is returned, as the models are
        structurally the same across individuals.

        If a population model has been set and not all parameters are pooled,
        the mechanistic and error model parameters appear multiple times (once
        for each individual). To get the mechanistic and error model parameters
        prior to setting a population model the ``exlude_pop_model`` flag can
        be set to ``True``.
        """
        if exclude_pop_model is True:
            _, parameter_names = self._get_number_and_parameter_names(
                exclude_pop_model=True)
            return copy.copy(parameter_names)

        return copy.copy(self._parameter_names)

    def get_predictive_model(self, exclude_pop_model=False):
        """
        #TODO:
        Returns the predictive model.
        """
        # Create predictive model
        predictive_model = erlo.PredictiveModel(
            self._mechanistic_model, self._error_models)

        # Return if no population model has been set, or is excluded
        if (self._population_models is None) or (exclude_pop_model is True):
            return predictive_model

        # Create predictive population model
        predictive_model = erlo.PredictivePopulationModel(
            predictive_model, self._population_models)

        return predictive_model

    def set_data(
            self, data, output_biomarker_dict=None, id_key='ID',
            time_key='Time', biom_key='Biomarker', meas_key='Measurement',
            dose_key='Dose', dose_duration_key='Duration'):
        """
        #TODO: add data
        If no dose or duration information exists, they can be set to None.
        """
        # Check input format
        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                'Data has to be pandas.DataFrame.')

        # If model does not support dose administration, set dose keys to None
        if isinstance(self._mechanistic_model, erlo.PharmacodynamicModel):
            dose_key = None
            dose_duration_key = None

        keys = [id_key, time_key, biom_key, meas_key]
        if dose_key is not None:
            keys += [dose_key]
        if dose_duration_key is not None:
            keys += [dose_duration_key]

        for key in keys:
            if key not in data.keys():
                raise ValueError(
                    'Data does not have the key <' + str(key) + '>.')

        # Get default output-biomarker map
        # (only possible if single output-problem, and only one biomarker in
        # dataframe)
        outputs = self._mechanistic_model.outputs()
        biomarkers = data[biom_key].dropna().unique()
        if output_biomarker_dict is None:
            if (len(outputs) > 1) or (len(biomarkers) > 1):
                raise ValueError(
                    'If more than one model output is set, or more than one '
                    'biomarker in the dataframe exists, a output-biomarker '
                    'map has to be provided.')

            # Create trivial map
            output_biomarker_dict = {outputs[0]: biomarkers[0]}

        # Check that output-biomarker map is valid
        for output in outputs:
            if output not in list(output_biomarker_dict.keys()):
                raise ValueError(
                    'The output <' + str(output) + '> could not be identified '
                    'in the output-biomarker map.')

            biomarker = output_biomarker_dict[output]
            if biomarker not in biomarkers:
                raise ValueError(
                    'The biomarker <' + str(biomarker) + '> could not be '
                    'identified in the dataframe.')

        self._id_key, self._time_key, self._biom_key, self._meas_key = [
            id_key, time_key, biom_key, meas_key]
        self._data = data[keys]
        self._output_biomarker_dict = output_biomarker_dict

        # Make sure data is formatted correctly
        self._clean_data(dose_key, dose_duration_key)
        self._ids = self._data[self._id_key].unique()

        # Extract dosing regimens
        self._dosing_regimens = None
        if dose_key is not None:
            self._dosing_regimens = self._extract_dosing_regimens(
                dose_key, dose_duration_key)

        # Update number and names of parameters
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

    def set_log_prior(self, log_priors, param_names=None):
        """
        Sets the log-prior probability distribution of the model parameters.

        The log-priors input is a list of :class:`pints.LogPrior` instances of
        the same length as the number of parameters, :meth:`n_parameters`.

        Correlations between model parameters is currently not supported. Each
        model parameter is assigned with an independent prior distribution,
        i.e. the joint log-prior for the model parameters is assumed to be a
        product of the marginal log-priors.

        If a population model has not been set, the provided log-prior is used
        for the parameters across all individuals.

        By default the log-priors are assumed to be ordered according to
        :meth:`get_parameter_names`. Alternatively, the mapping of the
        log-priors can be specified explicitly with ``param_names``.

        Parameters
        ----------
        log_priors
            A list of :class:`pints.LogPrior` of the length
            :meth:`n_parameters`.
        param_names
            A list of model parameter names, which is used to map the
            log-priors to the model parameters.
        """
        # Check inputs
        for log_prior in log_priors:
            if not isinstance(log_prior, pints.LogPrior):
                raise ValueError(
                    'All marginal log-priors have to be instances of a '
                    'pints.LogPrior.')

        if len(log_priors) != self.get_n_parameters():
            raise ValueError(
                'One marginal log-prior has to be provided for each parameter.'
            )

        n_parameters = 0
        for log_prior in log_priors:
            n_parameters += log_prior.n_parameters()

        if n_parameters != self.get_n_parameters():
            raise ValueError(
                'The joint log-prior does not match the dimensionality of the '
                'problem. At least one of the marginal log-priors has to be '
                'multi-dimensional.')

        if param_names is not None:
            if sorted(list(param_names)) != sorted(self._parameter_names):
                raise ValueError(
                    'The specified parameter names do not match the model '
                    'parameter names.')

            # Sort log-priors according to parameter names
            ordered = []
            for name in self._parameter_names:
                index = param_names.index(name)
                ordered.append(log_priors[index])

            log_priors = ordered

        self._log_prior = pints.ComposedLogPrior(*log_priors)

    def set_population_model(self, pop_models, params=None):
        """
        Sets the population model for each model parameter.

        A population model is a :class:`PopulationModel` class. A
        population model specifies how a model parameter varies across
        individuals.

        The population models ``pop_models`` are mapped to the model
        parameters. By default the first population model is mapped to the
        first mechanistic-error model parameter, the second
        population model to the second parameter, and so on. One
        population model has to be provided for each model parameter. The
        names of the mechanistic-error model parameters can be retrieved with
        :meth:`get_parameter_names` with ``exclude_pop_model=True``.

        Setting a population model resets the log-prior to ``None``.

        Parameters
        ----------
        pop_models
            A list of :class:`PopulationModel` classes that specifies the
            variation of model parameters between individuals. By default
            the list has to be of the same length as the number of mechanistic
            and error model parameters. If ``params`` is not ``None``, the list
            of population models has to be of the same length as ``params``.
        params
            A list of model parameter names, which map the population models
            to the parameter names.
        """
        # Check inputs
        for pop_model in pop_models:
            if not isinstance(pop_model, erlo.PopulationModel):
                raise ValueError(
                    'The population models have to be an instance of a '
                    'erlotinib.PopulationModel.')

        # Get individual parameter names
        n_parameters, parameter_names = self._get_number_and_parameter_names(
            exclude_pop_model=True)

        # Make sure that each parameter is assigned to a population model
        if len(pop_models) != n_parameters:
            raise ValueError(
                'If no parameter names are specified, exactly one population '
                'model has to be provided for each free parameter.')

        # Sort inputs according to `params`
        if params is not None:
            # Create default population model container
            ordered_pop_models = []

            # Map population models according to parameter names
            for name in parameter_names:
                try:
                    index = params.index(name)
                except ValueError:
                    raise ValueError(
                        'The parameter <' + str(name) + '> could not be '
                        'identified in the model')
                ordered_pop_models.append(pop_models[index])

            pop_models = ordered_pop_models

        # Save individual parameter names and population models
        self._population_models = copy.copy(pop_models)

        # Update parameter names and number of parameters
        self._n_parameters, self._parameter_names = \
            self._get_number_and_parameter_names()

        # Set prior to default
        self._log_prior = None
