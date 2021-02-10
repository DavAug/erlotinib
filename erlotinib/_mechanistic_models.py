#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import copy

import myokit
import myokit.formats.sbml as sbml
import numpy as np


class MechanisticModel(object):
    """
    A base class for models that are specified by sbml files.

    Parameters
    ----------
    sbml
        A path to the SBML model file, or a byte string
        in SBML format that specifies the model.

    """

    def __init__(self, sbml_input):
        super(MechanisticModel, self).__init__()

        if isinstance(sbml_input, bytes):
            model = sbml.SBMLParser().parse_string(sbml_input).myokit_model()
        elif isinstance(sbml_input, str):
            model = sbml.SBMLImporter().model(sbml_input)


        # Set default number and names of states, parameters and outputs.
        self._set_number_and_names(model)

        # Get time unit
        self._time_unit = self._get_time_unit(model)

        # Create simulator (intentionally public property)
        self.simulator = myokit.Simulation(model)

    def _get_time_unit(self, model):
        """
        Gets the model's time unit.
        """
        # Get bound variables
        bound_variables = [var for var in model.variables(bound=True)]

        # Get the variable that is bound to time
        # (only one can exist in myokit.Model)
        for var in bound_variables:
            if var._binding == 'time':
                return var.unit()

    def _set_const(self, parameters):
        """
        Sets values of constant model parameters.
        """
        for id_var, var in enumerate(self._const_names):
            self.simulator.set_constant(var, float(parameters[id_var]))

    def _set_state(self, parameters):
        """
        Sets initial values of states.
        """
        parameters = np.array(parameters)
        parameters = parameters[self._original_order]
        self.simulator.set_state(parameters)

    def _set_number_and_names(self, model):
        """
        Sets the number of states, parameters and outputs, as well as their
        names.
        """
        # Get the number of states and parameters
        self._n_states = model.count_states()
        n_const = model.count_variables(const=True)
        self._n_parameters = self._n_states + n_const

        # Get constant variable names and state names
        names = [var.qname() for var in model.states()]
        self._state_names = sorted(names)
        self._const_names = sorted(
            [var.qname() for var in model.variables(const=True)])

        # Remember original order of state names for simulation
        order_after_sort = np.argsort(names)
        self._original_order = np.argsort(order_after_sort)

        # Set default parameter names
        self._parameter_names = self._state_names + self._const_names

        # Set default outputs
        self._output_names = self._state_names
        self._n_outputs = self._n_states

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        return self._n_parameters

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return copy.copy(self._output_names)

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        return copy.copy(self._parameter_names)

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.

        Parameters
        ----------
        outputs
            A list of quantifiable variable names of the :class:`myokit.Model`,
            e.g. `compartment.variable`.
        """
        # Check that outputs are valid
        for output in outputs:
            try:
                self.simulator._model.get(output)
            except KeyError:
                raise KeyError(
                    'The variable <' + str(output) + '> does not exist in the '
                    'model.')

        self._output_names = list(outputs)
        self._n_outputs = len(outputs)

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current parameter names'
                'as keys and the new parameter names as values.')

        parameter_names = self._parameter_names
        for index, parameter in enumerate(self._parameter_names):
            try:
                parameter_names[index] = str(names[parameter])
            except KeyError:
                # KeyError indicates that a current parameter is not being
                # replaced.
                pass

        self._parameter_names = parameter_names

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs for specified
        parameters and times.

        The result is returned as a 2 dimensional NumPy array of shape
        (n_outputs, n_times).

        Parameters
        ----------
        parameters
            An array-like object with values for the model parameters.
        times
            An array-like object with time points at which the output
            values are returned.
        """
        # Reset simulation
        self.simulator.reset()

        # Set initial conditions
        self._set_state(parameters[:self._n_states])

        # Set constant model parameters
        self._set_const(parameters[self._n_states:])

        # Simulate
        output = self.simulator.run(
            times[-1] + 1, log=self._output_names, log_times=times)
        result = [output[name] for name in self._output_names]

        return np.array(result)

    def time_unit(self):
        """
        Returns the model's unit of time.
        """
        return self._time_unit


class PharmacodynamicModel(MechanisticModel):
    """
    Converts a pharmacodynamic model specified by an SBML file into a forward
    model that can be solved numerically.

    Extends :class:`MechanisticModel`.

    Parameters
    ----------
    sbml_file
        A path to the SBML model file that specifies the pharmacodynamic model.

    """

    def __init__(self, sbml_file):
        super(PharmacodynamicModel, self).__init__(sbml_file)

        # Set default pharmacokinetic input variable
        # (Typically drug concentration)
        self._pk_input = None
        model = self.simulator._model
        if model.has_variable('myokit.drug_concentration'):
            self._pk_input = 'myokit.drug_concentration'

    def pk_input(self):
        """
        Returns the pharmacokinetic input variable. In most models this will be
        the concentration of the drug.

        Defaults to ``None`` or ``myokit.drug_concentration`` if the latter is
        among the model parameters.
        """
        return self._pk_input

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current parameter names'
                'as keys and the new parameter names as values.')

        parameter_names = self._parameter_names
        for index, parameter in enumerate(self._parameter_names):
            try:
                parameter_names[index] = str(names[parameter])
            except KeyError:
                # KeyError indicates that a current parameter is not being
                # replaced.
                pass

        self._parameter_names = parameter_names

        # Rename pk input
        try:
            self._pk_input = str(names[self._pk_input])
        except KeyError:
            # KeyError indicates that the current name is not being
            # replaced.
            pass

    def set_pk_input(self, name):
        """
        Sets the pharmacokinetic input variable. In most models this will be
        the concentration of the drug.

        The name has to match a parameter of the model.
        """
        if name not in self._parameter_names:
            raise ValueError(
                'The name does not match a model parameter.')

        self._pk_input = name


class PharmacokineticModel(MechanisticModel):
    """
    Converts a pharmacokinetic model specified by an SBML file into a forward
    model that can be solved numerically.

    Extends :class:`MechanisticModel`.

    Parameters
    ----------
    sbml_file
        A path to the SBML model file that specifies the pharmacokinetic model.

    """

    def __init__(self, sbml_file):
        super(PharmacokineticModel, self).__init__(sbml_file)

        # Remember vanilla model (important for administration reset)
        self._default_model = self.simulator._model.clone()

        # Set default dose administration
        self._administration = None

        # Set default output variable that interacts with the pharmacodynamic
        # model
        # (Typically drug concentration in central compartment)
        self._pd_output = None
        if self._default_model.has_variable('central.drug_concentration'):
            self._pd_output = 'central.drug_concentration'

        # Set default output to pd output if not None
        if self._pd_output is not None:
            self._output_names = [self._pd_output]
            self._n_outputs = 1

    def _add_dose_compartment(self, model, drug_amount):
        """
        Adds a dose compartment to the model with a linear absorption rate to
        the connected compartment.
        """
        # Add a dose compartment to the model
        dose_comp = model.add_component_allow_renaming('dose')

        # Create a state variable for the drug amount in the dose compartment
        dose_drug_amount = dose_comp.add_variable('drug_amount')
        dose_drug_amount.set_rhs(0)
        dose_drug_amount.set_unit(drug_amount.unit())
        dose_drug_amount.promote()

        # Create an absorption rate variable
        absorption_rate = dose_comp.add_variable('absorption_rate')
        absorption_rate.set_rhs(1)
        absorption_rate.set_unit(1 / self.time_unit())

        # Add outflow expression to dose compartment
        dose_drug_amount.set_rhs(
            myokit.Multiply(
                myokit.PrefixMinus(myokit.Name(absorption_rate)),
                myokit.Name(dose_drug_amount)
                )
            )

        # Add inflow expression to connected compartment
        rhs = drug_amount.rhs()
        drug_amount.set_rhs(
            myokit.Plus(
                rhs,
                myokit.Multiply(
                    myokit.Name(absorption_rate),
                    myokit.Name(dose_drug_amount)
                )
            )
        )

        # Update number of parameters and states, as well as their names
        self._set_number_and_names(model)

        # Set default output to pd_output if it is not None
        if self._pd_output is not None:
            self._output_names = [self._pd_output]
            self._n_outputs = 1

        return dose_drug_amount

    def _add_dose_rate(self, drug_amount):
        """
        Adds a dose rate variable to the state variable, which is bound to the
        dosing regimen.
        """
        # Register a dose rate variable to the compartment and bind it to
        # pace, i.e. tell myokit that its value is set by the dosing regimen/
        # myokit.Protocol
        compartment = drug_amount.parent()
        dose_rate = compartment.add_variable_allow_renaming(
            str('dose_rate'))
        dose_rate.set_binding('pace')

        # Set initial value to 0 and unit to unit of drug amount over unit of
        # time
        dose_rate.set_rhs(0)
        dose_rate.set_unit(drug_amount.unit() / self.time_unit())

        # Add the dose rate to the rhs of the drug amount variable
        rhs = drug_amount.rhs()
        drug_amount.set_rhs(
            myokit.Plus(
                rhs,
                myokit.Name(dose_rate)
            )
        )

    def administration(self):
        """
        Returns the mode of administration in form of a dictionary.

        The dictionary has the keys 'compartment' and 'direct'. The former
        provides information about which compartment is dosed, and the latter
        whether the dose is administered directly ot indirectly to the
        compartment.
        """
        return self._administration

    def dosing_regimen(self):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`myokit.Protocol`. If the protocol has not been set, ``None`` is
        returned.
        """
        return self.simulator._protocol

    def set_administration(
            self, compartment, amount_var='drug_amount', direct=True):
        r"""
        Sets the route of administration of the compound.

        The compound is administered to the selected compartment either
        directly or indirectly. If it is administered directly, a dose rate
        variable is added to the drug amount's rate of change expression

        .. math ::

            \frac{\text{d}A}{\text{d}t} = \text{RHS} + r_d,

        where :math:`A` is the drug amount in the selected compartment, RHS is
        the rate of change of :math:`A` prior to adding the dose rate, and
        :math:`r_d` is the dose rate.

        The dose rate can be set by :meth:`set_dosing_regimen`.

        If the route of administration is indirect, a dosing compartment
        is added to the model, which is connected to the selected compartment.
        The dose rate variable is then added to the rate of change expression
        of the dose amount variable in the dosing compartment. The drug amount
        in the dosing compartment flows at a linear absorption rate into the
        selected compartment

        .. math ::

            \frac{\text{d}A_d}{\text{d}t} = -k_aA_d + r_d \\
            \frac{\text{d}A}{\text{d}t} = \text{RHS} + k_aA_d,

        where :math:`A_d` is the amount of drug in the dose compartment and
        :math:`k_a` is the absorption rate.

        Setting an indirect administration route changes the number of
        parameters of the model, and resets the parameter names to their
        defaults.

        Parameters
        ----------
        compartment
            Compartment to which doses are either directly or indirectly
            administered.
        amount_var
            Drug amount variable in the compartment. By default the drug amount
            variable is assumed to be 'drug_amount'.
        direct
            A boolean flag that indicates whether the dose is administered
            directly or indirectly to the compartment.
        """
        # Check inputs
        model = self._default_model.clone()
        if not model.has_component(compartment):
            raise ValueError(
                'The model does not have a compartment named <'
                + str(compartment) + '>.')
        comp = model.get(compartment, class_filter=myokit.Component)

        if not comp.has_variable(amount_var):
            raise ValueError(
                'The drug amount variable <' + str(amount_var) + '> could not '
                'be found in the compartment.')

        drug_amount = comp.get(amount_var)
        if not drug_amount.is_state():
            raise ValueError(
                'The variable <' + str(drug_amount) + '> is not a state '
                'variable, and can therefore not be dosed.')

        # If administration is indirect, add a dosing compartment and update
        # the drug amount variable to the one in the dosing compartment
        if not direct:
            drug_amount = self._add_dose_compartment(model, drug_amount)

        # Add dose rate variable to the right hand side of the drug amount
        self._add_dose_rate(drug_amount)

        # Update simulator
        # (otherwise simulator won't know about pace bound variable)
        self.simulator = myokit.Simulation(model)

        # Remember type of administration
        self._administration = dict(
            {'compartment': compartment, 'direct': direct})

    def set_dosing_regimen(
            self, dose, start, duration=0.01, period=None, num=None):
        """
        Sets the dosing regimen with which the compound is administered.

        The route of administration can be set with :meth:`set_administration`.
        However, the type of administration, e.g. bolus injection or infusion,
        may be controlled with the duration input.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model an infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the dose is administered once. To apply multiple doses
        provide a dose administration period.

        Parameters
        ----------
        dose
            The amount of the compound that is injected at each administration.
        start
            Start time of the treatment.
        duration
            Duration of dose administration. For a bolus injection, a dose
            duration of 1% of the time unit should suffice. By default the
            duration is set to 0.01 (bolus).
        period
            Periodicity at which doses are administered. If ``None`` the dose
            is administered only once.
        num
            Number of administered doses. If ``None`` and the periodicity of
            the administration is not ``None``, doses are administered
            indefinitely.
        """
        if self._administration is None:
            raise ValueError(
                'The route of administration of the dose has not been set.')

        if num is None:
            # Myokits default is zero, i.e. infinitely many doses
            num = 0

        if period is None:
            # If period is not provided, we administer a single dose
            # Myokits defaults are 0s for that.
            period = 0
            num = 0

        # Translate dose to dose rate
        dose_rate = dose / duration

        # Set dosing regimen
        dosing_regimen = myokit.pacing.blocktrain(
            period=period, duration=duration, offset=start, level=dose_rate,
            limit=num)
        self.simulator.set_protocol(dosing_regimen)

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
        """
        if not isinstance(names, dict):
            raise TypeError(
                'Names has to be a dictionary with the current parameter names'
                'as keys and the new parameter names as values.')

        parameter_names = self._parameter_names
        for index, parameter in enumerate(self._parameter_names):
            try:
                parameter_names[index] = str(names[parameter])
            except KeyError:
                # KeyError indicates that a current parameter is not being
                # replaced.
                pass

        self._parameter_names = parameter_names

        # Rename pd output
        try:
            self._pd_output = str(names[self._pd_output])
        except KeyError:
            # KeyError indicates that the current name is not being
            # replaced.
            pass

    def pd_output(self):
        """
        Returns the variable which interacts with the pharmacodynamic model.
        In most models this will be the concentration of the drug in the
        central compartment.

        This variable is mapped to the
        :meth:`erlotinib.PharmacodynamicModel.pk_input` variable when a
        :class:`PKPDModel` is instantiated.

        Defaults to ``None`` or ``central.drug_concentration`` if the latter is
        among the model parameters.
        """
        return self._pd_output

    def set_pd_output(self, name):
        """
        Sets the variable which interacts with the pharmacodynamic model.
        In most models this will be the concentration of the drug in the
        central compartment.

        The name has to match a parameter of the model.

        This variable is mapped to the
        :meth:`erlotinib.PharmacodynamicModel.pk_input` variable when a
        :class:`PKPDModel` is instantiated.
        """
        # Get intermediate variable names
        inter_names = [
            var.qname() for var in self.simulator._model.variables(inter=True)]

        names = inter_names + self._parameter_names
        if name not in names:
            raise ValueError(
                'The name does not match a model variable.')

        self._pd_output = name


class ReducedMechanisticModel(object):
    """
    A class that can be used to permanently fix model parameters of a
    :class:`MechanisticModel` instance.

    This may be useful to explore simplified versions of a model before
    defining a new SBML file.

    Parameters
    ----------
    mechanistic_model
        An instance of a :class:`MechanisticModel`.
    """

    def __init__(self, mechanistic_model):
        super(ReducedMechanisticModel, self).__init__()

        # Check input
        if not isinstance(mechanistic_model, MechanisticModel):
            raise ValueError(
                'The mechanistic model has to be an instance of a '
                'erlotinib.MechanisticModel')

        self._mechanistic_model = mechanistic_model
        self.simulator = mechanistic_model.simulator

        # Set defaults
        self._fixed_params_mask = None
        self._fixed_params_values = None
        self._n_parameters = mechanistic_model.n_parameters()
        self._parameter_names = mechanistic_model.parameters()

    def dosing_regimen(self):
        """
        Returns the dosing regimen of the compound in form of a
        :class:`myokit.Protocol`. If the protocol has not been set, ``None`` is
        returned.

        If the model does not support dose administration, ``None`` is
        returned.
        """
        try:
            return self._mechanistic_model.dosing_regimen()
        except AttributeError:
            return None

    def fix_parameters(self, name_value_dict):
        """
        Fixes the value of model parameters, and effectively removes them as a
        parameter from the model. Fixing the value of a parameter at ``None``,
        sets the parameter free again.

        Parameters
        ----------
        name_value_dict
            A dictionary with model parameter names as keys, and parameter
            values as values.
        """
        # Check type
        try:
            name_value_dict = dict(name_value_dict)
        except (TypeError, ValueError):
            raise ValueError(
                'The name-value dictionary has to be convertable to a python '
                'dictionary.')

        # If no model parameters have been fixed before, instantiate a mask
        # and values
        if self._fixed_params_mask is None:
            self._fixed_params_mask = np.zeros(
                shape=self._n_parameters, dtype=bool)

        if self._fixed_params_values is None:
            self._fixed_params_values = np.empty(shape=self._n_parameters)

        # Update the mask and values
        for index, name in enumerate(self._parameter_names):
            try:
                value = name_value_dict[name]
            except KeyError:
                # KeyError indicates that parameter name is not being fixed
                continue

            # Fix parameter if value is not None, else unfix it
            self._fixed_params_mask[index] = value is not None
            self._fixed_params_values[index] = value

        # If all parameters are free, set mask and values to None again
        if np.alltrue(~self._fixed_params_mask):
            self._fixed_params_mask = None
            self._fixed_params_values = None

    def mechanistic_model(self):
        """
        Returns the original mechanistic model.
        """
        return self._mechanistic_model

    def n_fixed_parameters(self):
        """
        Returns the number of fixed model parameters.
        """
        if self._fixed_params_mask is None:
            return 0

        n_fixed = int(np.sum(self._fixed_params_mask))

        return n_fixed

    def n_outputs(self):
        """
        Returns the number of output dimensions.

        By default this is the number of states.
        """
        return self._mechanistic_model.n_outputs()

    def n_parameters(self):
        """
        Returns the number of parameters in the model.

        Parameters of the model are initial state values and structural
        parameter values.
        """
        # Get number of fixed parameters
        n_fixed = 0
        if self._fixed_params_mask is not None:
            n_fixed = int(np.sum(self._fixed_params_mask))

        # Subtract fixed parameters from total number
        n_parameters = self._n_parameters - n_fixed

        return n_parameters

    def outputs(self):
        """
        Returns the output names of the model.
        """
        return self._mechanistic_model.outputs()

    def parameters(self):
        """
        Returns the parameter names of the model.
        """
        # Remove fixed model parameters
        names = self._parameter_names
        if self._fixed_params_mask is not None:
            names = np.array(names)
            names = names[~self._fixed_params_mask]
            names = list(names)

        return copy.copy(names)

    def pd_output(self):
        """
        Returns the variable which interacts with the pharmacodynamic model.
        In most models this will be the concentration of the drug in the
        central compartment.

        This variable is mapped to the
        :meth:`erlotinib.PharmacodynamicModel.pk_input` variable when a
        :class:`PKPDModel` is instantiated.

        Defaults to ``None`` or ``central.drug_concentration`` if the latter is
        among the model parameters.

        If the model does not support a pd output, ``None`` is returned.
        """
        try:
            return self._mechanistic_model.pd_output()
        except AttributeError:
            return None

    def pk_input(self):
        """
        Returns the pharmacokinetic input variable. In most models this will be
        the concentration of the drug.

        Defaults to ``None`` or ``myokit.drug_concentration`` if the latter is
        among the model parameters.

        If the model does not support a pk input, ``None`` is returned.
        """
        try:
            return self._mechanistic_model.pk_input()
        except AttributeError:
            return None

    def set_dosing_regimen(
            self, dose, start, duration=0.01, period=None, num=None):
        """
        Sets the dosing regimen with which the compound is administered.

        The route of administration can be set with :meth:`set_administration`.
        However, the type of administration, e.g. bolus injection or infusion,
        may be controlled with the duration input.

        By default the dose is administered as a bolus injection (duration on
        a time scale that is 100 fold smaller than the basic time unit). To
        model an infusion of the dose over a longer time period, the
        ``duration`` can be adjusted to the appropriate time scale.

        By default the dose is administered once. To apply multiple doses
        provide a dose administration period.

        Parameters
        ----------
        dose
            The amount of the compound that is injected at each administration.
        start
            Start time of the treatment.
        duration
            Duration of dose administration. For a bolus injection, a dose
            duration of 1% of the time unit should suffice. By default the
            duration is set to 0.01 (bolus).
        period
            Periodicity at which doses are administered. If ``None`` the dose
            is administered only once.
        num
            Number of administered doses. If ``None`` and the periodicity of
            the administration is not ``None``, doses are administered
            indefinitely.
        """
        try:
            self._mechanistic_model.set_dosing_regimen(
                dose, start, duration, period, num)
        except AttributeError:
            raise AttributeError(
                'The mechanistic model does not support dosing regimens.')

    def set_outputs(self, outputs):
        """
        Sets outputs of the model.

        Parameters
        ----------
        outputs
            A list of quantifiable variable names of the :class:`myokit.Model`,
            e.g. `compartment.variable`.
        """
        self._mechanistic_model.set_outputs(outputs)

    def set_parameter_names(self, names):
        """
        Assigns names to the parameters. By default the :class:`myokit.Model`
        names are assigned to the parameters.

        Parameters
        ----------
        names
            A dictionary that maps the current parameter names to new names.
        """
        # Set parameter names
        self._mechanistic_model.set_parameter_names(names)
        self._parameter_names = self._mechanistic_model.parameters()

    def simulate(self, parameters, times):
        """
        Returns the numerical solution of the model outputs for specified
        parameters and times.

        The result is returned as a 2 dimensional NumPy array of shape
        (n_outputs, n_times).

        Parameters
        ----------
        parameters
            An array-like object with values for the model parameters.
        times
            An array-like object with time points at which the output
            values are returned.
        """
        # Insert fixed parameter values
        if self._fixed_params_mask is not None:
            self._fixed_params_values[
                ~self._fixed_params_mask] = parameters
            parameters = self._fixed_params_values

        return self._mechanistic_model.simulate(parameters, times)

    def time_unit(self):
        """
        Returns the model's unit of time.
        """
        return self._mechanistic_model.time_unit()
