#
# This file is part of the erlotinib repository
# (https://github.com/DavAug/erlotinib/) which is released under the
# BSD 3-clause license. See accompanying LICENSE.md for copyright notice and
# full license details.
#

import unittest

import numpy as np
import pints
import pints.toy

import erlotinib as erlo


class TestHierarchicalLogLikelihood(unittest.TestCase):
    """
    Tests the erlotinib.HierarchicalLogLikelihood class.
    """

    @classmethod
    def setUpClass(cls):
        # Create data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        obs_2 = [2, 2.1, 2.2]
        times_2 = [2, 5, 6]
        cls.observations = [obs_1, obs_2]
        cls.times = [times_1, times_2]

        # Set up mechanistic and error models
        path = erlo.ModelLibrary().one_compartment_pk_model()
        cls.model = erlo.PharmacokineticModel(path)
        cls.model.set_administration('central', direct=False)
        cls.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        cls.error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihoods
        cls.log_likelihoods = [
            erlo.LogLikelihood(
                cls.model, cls.error_models, cls.observations, cls.times),
            erlo.LogLikelihood(
                cls.model, cls.error_models, cls.observations, cls.times)]

        # Create population models
        cls.population_models = [
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel(),
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel()]

        cls.hierarchical_model = erlo.HierarchicalLogLikelihood(
            cls.log_likelihoods, cls.population_models)

    def test_bad_instantiation(self):
        # Log-likelihoods are not pints.LogPDF
        log_likelihoods = ['bad', 'type']
        with self.assertRaisesRegex(ValueError, 'The log-likelihoods have'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # Log-likelihoods are defined on different parameter spaces
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        log_likelihoods = [
            self.log_likelihoods[0],
            erlo.LogLikelihood(
                model, error_models, self.observations[0], self.times[0])]

        with self.assertRaisesRegex(ValueError, 'The number of parameters'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # The log-likelihood parameter names differ
        model.set_outputs(['central.drug_concentration', 'dose.drug_amount'])
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2
        log_likelihoods = [
            self.log_likelihoods[0],
            erlo.LogLikelihood(
                model, error_models, self.observations, self.times)]

        with self.assertRaisesRegex(ValueError, 'The parameter names'):
            erlo.HierarchicalLogLikelihood(
                log_likelihoods, self.population_models)

        # Population models are not erlotinib.PopulationModel
        population_models = ['bad', 'type'] + ['match dimension'] * 7
        with self.assertRaisesRegex(ValueError, 'The population models have'):
            erlo.HierarchicalLogLikelihood(
                self.log_likelihoods, population_models)

        # Not all parameters of the likelihoods are assigned to a pop model
        population_models = [
            erlo.PooledModel(),
            erlo.PooledModel()]
        with self.assertRaisesRegex(ValueError, 'Wrong number of population'):
            erlo.HierarchicalLogLikelihood(
                self.log_likelihoods, population_models)

    def test_call(self):
        # Test case I: All parameters pooled
        model = erlo.HierarchicalLogLikelihood(
            log_likelihoods=self.log_likelihoods,
            population_models=[erlo.PooledModel()] * 9)

        # Create reference model
        pooled_log_pdf = pints.PooledLogPDF(
            self.log_likelihoods, pooled=[True]*9)

        # Test case I.1
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        score = pooled_log_pdf(parameters)

        self.assertEqual(model(parameters), score)

        # Test case I.2
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        score = pooled_log_pdf(parameters)

        self.assertEqual(model(parameters), score)

        # Test case II.1: Heterogeneous model
        likelihood = erlo.HierarchicalLogLikelihood(
            log_likelihoods=self.log_likelihoods,
            population_models=[
                erlo.HeterogeneousModel()] * 9)

        # Compute score from individual likelihoods
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        n_parameters = 9
        n_ids = 2
        parameters = [1] * n_parameters * n_ids
        self.assertEqual(likelihood(parameters), score)

        # Test case II.2
        # Compute score from individual likelihoods
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        score = 0
        for ll in self.log_likelihoods:
            score += ll(parameters)

        parameters = \
            [parameters[0]] * n_ids + \
            [parameters[1]] * n_ids + \
            [parameters[2]] * n_ids + \
            [parameters[3]] * n_ids + \
            [parameters[4]] * n_ids + \
            [parameters[5]] * n_ids + \
            [parameters[6]] * n_ids + \
            [parameters[7]] * n_ids + \
            [parameters[8]] * n_ids
        self.assertEqual(likelihood(parameters), score)

        # Test case III.1: Non-trivial population model
        # Reminder of population model
        # cls.population_models = [
        #     erlo.PooledModel(),
        #     erlo.PooledModel(),
        #     erlo.LogNormalModel(),
        #     erlo.PooledModel(),
        #     erlo.HeterogeneousModel(),
        #     erlo.PooledModel(),
        #     erlo.PooledModel(),
        #     erlo.PooledModel(),
        #     erlo.PooledModel()]

        # Create reference pop model
        ref_pop_model = erlo.LogNormalModel()
        indiv_parameters_1 = [10, 1, 0.1, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        pop_params = [0.2, 1]

        parameters = [
            indiv_parameters_1[0],
            indiv_parameters_1[1],
            indiv_parameters_1[2],
            indiv_parameters_2[2],
            pop_params[0],
            pop_params[1],
            indiv_parameters_1[3],
            indiv_parameters_1[4],
            indiv_parameters_2[4],
            indiv_parameters_1[5],
            indiv_parameters_1[6],
            indiv_parameters_1[7],
            indiv_parameters_1[8]]

        score = \
            ref_pop_model.compute_log_likelihood(
                parameters=pop_params,
                observations=[0.1, 0.2]) + \
            self.log_likelihoods[0](indiv_parameters_1) + \
            self.log_likelihoods[1](indiv_parameters_2)

        self.assertNotEqual(score, -np.inf)
        self.assertAlmostEqual(self.hierarchical_model(parameters), score)

        # Test case III.2: Returns -np.inf if individuals are far away from
        # pop distribution
        indiv_parameters_1 = [10, 1, 10E20, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        pop_params = [0.2, 10E-10]

        parameters = [
            indiv_parameters_1[0],
            indiv_parameters_1[1],
            indiv_parameters_1[2],
            indiv_parameters_2[2],
            pop_params[0],
            pop_params[1],
            indiv_parameters_1[3],
            indiv_parameters_1[4],
            indiv_parameters_2[4],
            indiv_parameters_1[5],
            indiv_parameters_1[6],
            indiv_parameters_1[7],
            indiv_parameters_1[8]]

        self.assertEqual(self.hierarchical_model(parameters), -np.inf)

    def test_evaluateS1(self):
        # Test case I: All parameters pooled
        model = erlo.HierarchicalLogLikelihood(
            log_likelihoods=self.log_likelihoods,
            population_models=[erlo.PooledModel()] * 9)

        # Create reference model
        pooled_log_pdf = pints.PooledLogPDF(
            self.log_likelihoods, pooled=[True]*9)

        # Test case I.1
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ref_score, ref_sens = pooled_log_pdf.evaluateS1(parameters)
        score, sens = model.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 9)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])

        # Test case I.2
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        ref_score, ref_sens = pooled_log_pdf.evaluateS1(parameters)
        score, sens = model.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 9)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])

        # Test case II.1: Heterogeneous model
        likelihood = erlo.HierarchicalLogLikelihood(
            log_likelihoods=self.log_likelihoods,
            population_models=[
                erlo.HeterogeneousModel()] * 9)

        # Compute score from individual likelihoods
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        ref_score = 0
        ref_senss = []
        for ll in self.log_likelihoods:
            s, se = ll.evaluateS1(parameters)
            ref_score += s
            ref_senss.append(se)

        n_parameters = 9
        n_ids = 2
        parameters = [1] * n_parameters * n_ids
        score, sens = likelihood.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 18)
        ref_sens = ref_senss[0]
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[2], ref_sens[1])
        self.assertEqual(sens[4], ref_sens[2])
        self.assertEqual(sens[6], ref_sens[3])
        self.assertEqual(sens[8], ref_sens[4])
        self.assertEqual(sens[10], ref_sens[5])
        self.assertEqual(sens[12], ref_sens[6])
        self.assertEqual(sens[14], ref_sens[7])
        self.assertEqual(sens[16], ref_sens[8])
        ref_sens = ref_senss[1]
        self.assertEqual(sens[1], ref_sens[0])
        self.assertEqual(sens[3], ref_sens[1])
        self.assertEqual(sens[5], ref_sens[2])
        self.assertEqual(sens[7], ref_sens[3])
        self.assertEqual(sens[9], ref_sens[4])
        self.assertEqual(sens[11], ref_sens[5])
        self.assertEqual(sens[13], ref_sens[6])
        self.assertEqual(sens[15], ref_sens[7])
        self.assertEqual(sens[17], ref_sens[8])

        # Test case II.2
        # Compute score from individual likelihoods
        parameters = [10, 1, 0.1, 1, 3, 1, 1, 1, 1]
        ref_score = 0
        ref_senss = []
        for ll in self.log_likelihoods:
            s, se = ll.evaluateS1(parameters)
            ref_score += s
            ref_senss.append(se)

        n_ids = 2
        parameters = \
            [parameters[0]] * n_ids + \
            [parameters[1]] * n_ids + \
            [parameters[2]] * n_ids + \
            [parameters[3]] * n_ids + \
            [parameters[4]] * n_ids + \
            [parameters[5]] * n_ids + \
            [parameters[6]] * n_ids + \
            [parameters[7]] * n_ids + \
            [parameters[8]] * n_ids
        score, sens = likelihood.evaluateS1(parameters)

        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 18)
        ref_sens = ref_senss[0]
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[2], ref_sens[1])
        self.assertEqual(sens[4], ref_sens[2])
        self.assertEqual(sens[6], ref_sens[3])
        self.assertEqual(sens[8], ref_sens[4])
        self.assertEqual(sens[10], ref_sens[5])
        self.assertEqual(sens[12], ref_sens[6])
        self.assertEqual(sens[14], ref_sens[7])
        self.assertEqual(sens[16], ref_sens[8])
        ref_sens = ref_senss[1]
        self.assertEqual(sens[1], ref_sens[0])
        self.assertEqual(sens[3], ref_sens[1])
        self.assertEqual(sens[5], ref_sens[2])
        self.assertEqual(sens[7], ref_sens[3])
        self.assertEqual(sens[9], ref_sens[4])
        self.assertEqual(sens[11], ref_sens[5])
        self.assertEqual(sens[13], ref_sens[6])
        self.assertEqual(sens[15], ref_sens[7])
        self.assertEqual(sens[17], ref_sens[8])

        # Test case III.1: Non-trivial population model
        # Reminder of population model
        # cls.population_models = [
        #     erlo.PooledModel(),
        #     erlo.PooledModel(),
        #     erlo.LogNormalModel(),
        #     erlo.PooledModel(),
        #     erlo.HeterogeneousModel(),
        #     erlo.PooledModel(),
        #     erlo.PooledModel(),
        #     erlo.PooledModel(),
        #     erlo.PooledModel()]

        # Create reference pop model
        ref_pop_model = erlo.LogNormalModel()
        indiv_parameters_1 = [10, 1, 0.1, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        pop_params = [0.2, 1]

        parameters = [
            indiv_parameters_1[0],
            indiv_parameters_1[1],
            indiv_parameters_1[2],
            indiv_parameters_2[2],
            pop_params[0],
            pop_params[1],
            indiv_parameters_1[3],
            indiv_parameters_1[4],
            indiv_parameters_2[4],
            indiv_parameters_1[5],
            indiv_parameters_1[6],
            indiv_parameters_1[7],
            indiv_parameters_1[8]]

        ref_s1, ref_sens1 = ref_pop_model.compute_sensitivities(
                parameters=pop_params,
                observations=[0.1, 0.2])
        ref_s2, ref_sens2 = self.log_likelihoods[0].evaluateS1(
            indiv_parameters_1)
        ref_s3, ref_sens3 = self.log_likelihoods[1].evaluateS1(
            indiv_parameters_2)

        ref_score = ref_s1 + ref_s2 + ref_s3
        ref_sens = [
            ref_sens2[0] + ref_sens3[0],
            ref_sens2[1] + ref_sens3[1],
            ref_sens1[0] + ref_sens2[2],
            ref_sens1[1] + ref_sens3[2],
            ref_sens1[2],
            ref_sens1[3],
            ref_sens2[3] + ref_sens3[3],
            ref_sens2[4],
            ref_sens3[4],
            ref_sens2[5] + ref_sens3[5],
            ref_sens2[6] + ref_sens3[6],
            ref_sens2[7] + ref_sens3[7],
            ref_sens2[8] + ref_sens3[8]]

        # Compute score and sensitivities with hierarchical model
        score, sens = self.hierarchical_model.evaluateS1(parameters)

        self.assertNotEqual(score, -np.inf)
        self.assertFalse(np.any(np.isinf(sens)))
        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 13)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        self.assertEqual(sens[9], ref_sens[9])
        self.assertEqual(sens[10], ref_sens[10])
        self.assertEqual(sens[11], ref_sens[11])
        self.assertEqual(sens[12], ref_sens[12])

        # Test case III.2: Returns -np.inf if individuals are far away from
        # pop distribution
        indiv_parameters_1 = [10, 1, 10E20, 1, 3, 1, 1, 2, 1.2]
        indiv_parameters_2 = [10, 1, 0.2, 1, 2, 1, 1, 2, 1.2]
        pop_params = [0.2, 10E-10]

        parameters = [
            indiv_parameters_1[0],
            indiv_parameters_1[1],
            indiv_parameters_1[2],
            indiv_parameters_2[2],
            pop_params[0],
            pop_params[1],
            indiv_parameters_1[3],
            indiv_parameters_1[4],
            indiv_parameters_2[4],
            indiv_parameters_1[5],
            indiv_parameters_1[6],
            indiv_parameters_1[7],
            indiv_parameters_1[8]]

        score, sens = self.hierarchical_model.evaluateS1(parameters)

        self.assertEqual(score, -np.inf)

    def test_get_id(self):
        ids = self.hierarchical_model.get_id()

        self.assertEqual(len(ids), 13)
        self.assertIsNone(ids[0])
        self.assertIsNone(ids[1])
        self.assertEqual(ids[2], 'automatic-id-1')
        self.assertEqual(ids[3], 'automatic-id-2')
        self.assertIsNone(ids[4])
        self.assertIsNone(ids[5])
        self.assertIsNone(ids[6])
        self.assertEqual(ids[7], 'automatic-id-1')
        self.assertEqual(ids[8], 'automatic-id-2')
        self.assertIsNone(ids[9])
        self.assertIsNone(ids[10])
        self.assertIsNone(ids[11])
        self.assertIsNone(ids[12])

    def test_get_parameter_names(self):
        # Test case I: without ids
        parameter_names = self.hierarchical_model.get_parameter_names()

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'central.size')
        self.assertEqual(parameter_names[3], 'central.size')
        self.assertEqual(parameter_names[4], 'Mean central.size')
        self.assertEqual(parameter_names[5], 'Std. central.size')
        self.assertEqual(parameter_names[6], 'Pooled dose.absorption_rate')
        self.assertEqual(parameter_names[7], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[8], 'myokit.elimination_rate')
        self.assertEqual(
            parameter_names[9], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[11], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[12], 'Pooled dose.drug_amount Sigma rel.')

        # Test case II: Exclude bottom-level
        parameter_names = self.hierarchical_model.get_parameter_names(
            exclude_bottom_level=True)

        self.assertEqual(len(parameter_names), 11)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'Mean central.size')
        self.assertEqual(parameter_names[3], 'Std. central.size')
        self.assertEqual(parameter_names[4], 'Pooled dose.absorption_rate')
        self.assertEqual(parameter_names[5], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[6], 'myokit.elimination_rate')
        self.assertEqual(
            parameter_names[7], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[8], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[9], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled dose.drug_amount Sigma rel.')

        # Test case III: with ids
        parameter_names = self.hierarchical_model.get_parameter_names(
            include_ids=True)

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'automatic-id-1 central.size')
        self.assertEqual(parameter_names[3], 'automatic-id-2 central.size')
        self.assertEqual(parameter_names[4], 'Mean central.size')
        self.assertEqual(parameter_names[5], 'Std. central.size')
        self.assertEqual(parameter_names[6], 'Pooled dose.absorption_rate')
        self.assertEqual(
            parameter_names[7], 'automatic-id-1 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[8], 'automatic-id-2 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[9], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[11], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[12], 'Pooled dose.drug_amount Sigma rel.')

        # Test case IV: Exclude bottom-level with IDs
        parameter_names = self.hierarchical_model.get_parameter_names(
            exclude_bottom_level=True, include_ids=True)

        self.assertEqual(len(parameter_names), 11)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'Mean central.size')
        self.assertEqual(parameter_names[3], 'Std. central.size')
        self.assertEqual(parameter_names[4], 'Pooled dose.absorption_rate')
        self.assertEqual(
            parameter_names[5], 'automatic-id-1 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[6], 'automatic-id-2 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[7], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[8], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[9], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled dose.drug_amount Sigma rel.')

    def test_get_population_models(self):
        pop_models = self.hierarchical_model.get_population_models()
        self.assertEqual(len(pop_models), 9)
        self.assertIsInstance(pop_models[0], erlo.PopulationModel)
        self.assertIsInstance(pop_models[1], erlo.PopulationModel)
        self.assertIsInstance(pop_models[2], erlo.PopulationModel)
        self.assertIsInstance(pop_models[3], erlo.PopulationModel)
        self.assertIsInstance(pop_models[4], erlo.PopulationModel)
        self.assertIsInstance(pop_models[5], erlo.PopulationModel)
        self.assertIsInstance(pop_models[6], erlo.PopulationModel)
        self.assertIsInstance(pop_models[7], erlo.PopulationModel)
        self.assertIsInstance(pop_models[8], erlo.PopulationModel)

    def test_n_log_likelihoods(self):
        n_ids = self.hierarchical_model.n_log_likelihoods()
        self.assertEqual(n_ids, 2)

    def test_n_parameters(self):
        # Test case I: All parameters
        # 9 individual parameters, from which 1 is modelled heterogeneously,
        # 1 log-normally and the rest is pooled
        # And there are 2 individuals
        n_parameters = 2 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.assertEqual(
            self.hierarchical_model.n_parameters(), n_parameters)

        # Test case II: Exclude bottom parameters
        n_parameters = self.hierarchical_model.n_parameters(
            exclude_bottom_level=True)
        self.assertEqual(n_parameters, 11)


class TestHierarchicalLogPosterior(unittest.TestCase):
    """
    Tests the erlotinib.HierarchicalLogPosterior class.
    """

    @classmethod
    def setUpClass(cls):
        # Create data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        obs_2 = [2, 2.1, 2.2]
        times_2 = [2, 5, 6]
        observations = [obs_1, obs_2]
        times = [times_1, times_2]

        # Set up mechanistic and error models
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        model.set_administration('central', direct=False)
        model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihoods
        log_likelihoods = [
            erlo.LogLikelihood(
                model, error_models, observations, times),
            erlo.LogLikelihood(
                model, error_models, observations, times)]

        # Create population models
        population_models = [
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.LogNormalModel(),
            erlo.PooledModel(),
            erlo.HeterogeneousModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel(),
            erlo.PooledModel()]

        # Create hierarchical log-likelihood
        cls.hierarch_log_likelihood = erlo.HierarchicalLogLikelihood(
            log_likelihoods, population_models)

        # Define log-prior
        cls.log_prior = pints.ComposedLogPrior(
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1),
            pints.LogNormalLogPrior(1, 1))

        # Create log-posterior
        cls.log_posterior = erlo.HierarchicalLogPosterior(
            cls.hierarch_log_likelihood,
            cls.log_prior)

    def test_bad_instantiation(self):
        # Log-likelihood has bad type
        log_likelihood = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-likelihood has'):
            erlo.HierarchicalLogPosterior(
                log_likelihood, self.log_prior)

        # Log-prior has bad type
        log_prior = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-prior has to be'):
            erlo.HierarchicalLogPosterior(
                self.hierarch_log_likelihood, log_prior)

        # The dimension of the log-prior does not match number of top-level
        # parameters
        log_prior = pints.LogNormalLogPrior(0, 1)
        with self.assertRaisesRegex(ValueError, 'The log-prior has to have'):
            erlo.HierarchicalLogPosterior(
                self.hierarch_log_likelihood, log_prior)

    def test_call(self):
        # Test case I: Check score contributions add appropriately
        all_params = np.arange(start=1, stop=14, step=1)
        top_params = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ref_score = self.hierarch_log_likelihood(all_params) + \
            self.log_prior(top_params)
        score = self.log_posterior(all_params)

        self.assertNotEqual(score, -np.inf)
        self.assertEqual(score, ref_score)

        # Test case II: Check exception for inf prior score
        parameters = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        self.assertEqual(self.log_posterior(parameters), -np.inf)

    def test_evaluateS1(self):
        # Test case I: Check score contributions add appropriately
        all_params = np.arange(start=1, stop=14, step=1)
        top_params = [1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        ref_score1, ref_sens1 = self.hierarch_log_likelihood.evaluateS1(
            all_params)
        ref_score2, ref_sens2 = self.log_prior.evaluateS1(top_params)
        ref_score = ref_score1 + ref_score2
        ref_sens = [
            ref_sens1[0] + ref_sens2[0],
            ref_sens1[1] + ref_sens2[1],
            ref_sens1[2],
            ref_sens1[3],
            ref_sens1[4] + ref_sens2[2],
            ref_sens1[5] + ref_sens2[3],
            ref_sens1[6] + ref_sens2[4],
            ref_sens1[7] + ref_sens2[5],
            ref_sens1[8] + ref_sens2[6],
            ref_sens1[9] + ref_sens2[7],
            ref_sens1[10] + ref_sens2[8],
            ref_sens1[11] + ref_sens2[9],
            ref_sens1[12] + ref_sens2[10]]

        score, sens = self.log_posterior.evaluateS1(all_params)

        self.assertNotEqual(score, -np.inf)
        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 13)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])
        self.assertEqual(sens[7], ref_sens[7])
        self.assertEqual(sens[8], ref_sens[8])
        self.assertEqual(sens[9], ref_sens[9])
        self.assertEqual(sens[10], ref_sens[10])
        self.assertEqual(sens[11], ref_sens[11])
        self.assertEqual(sens[12], ref_sens[12])

        # Test case II: Check exception for inf prior score
        parameters = [-1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        score, _ = self.log_posterior.evaluateS1(parameters)
        self.assertEqual(score, -np.inf)

    def test_get_log_likelihood(self):
        log_likelihood = self.log_posterior.get_log_likelihood()
        self.assertIsInstance(log_likelihood, erlo.HierarchicalLogLikelihood)

    def test_get_log_prior(self):
        log_prior = self.log_posterior.get_log_prior()
        self.assertIsInstance(log_prior, pints.LogPrior)

    def test_get_id(self):
        ids = self.log_posterior.get_id()

        self.assertEqual(len(ids), 13)
        self.assertIsNone(ids[0])
        self.assertIsNone(ids[1])
        self.assertEqual(ids[2], 'automatic-id-1')
        self.assertEqual(ids[3], 'automatic-id-2')
        self.assertIsNone(ids[4])
        self.assertIsNone(ids[5])
        self.assertIsNone(ids[6])
        self.assertEqual(ids[7], 'automatic-id-1')
        self.assertEqual(ids[8], 'automatic-id-2')
        self.assertIsNone(ids[9])
        self.assertIsNone(ids[10])
        self.assertIsNone(ids[11])
        self.assertIsNone(ids[12])

    def test_get_parameter_names(self):
        # Test case I: without ids
        parameter_names = self.log_posterior.get_parameter_names()

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'central.size')
        self.assertEqual(parameter_names[3], 'central.size')
        self.assertEqual(parameter_names[4], 'Mean central.size')
        self.assertEqual(parameter_names[5], 'Std. central.size')
        self.assertEqual(parameter_names[6], 'Pooled dose.absorption_rate')
        self.assertEqual(parameter_names[7], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[8], 'myokit.elimination_rate')
        self.assertEqual(
            parameter_names[9], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[11], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[12], 'Pooled dose.drug_amount Sigma rel.')

        # Test case II: Exclude bottom-level
        parameter_names = self.log_posterior.get_parameter_names(
            exclude_bottom_level=True)

        self.assertEqual(len(parameter_names), 11)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'Mean central.size')
        self.assertEqual(parameter_names[3], 'Std. central.size')
        self.assertEqual(parameter_names[4], 'Pooled dose.absorption_rate')
        self.assertEqual(parameter_names[5], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[6], 'myokit.elimination_rate')
        self.assertEqual(
            parameter_names[7], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[8], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[9], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled dose.drug_amount Sigma rel.')

        # Test case III: with ids
        parameter_names = self.log_posterior.get_parameter_names(
            include_ids=True)

        self.assertEqual(len(parameter_names), 13)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'automatic-id-1 central.size')
        self.assertEqual(parameter_names[3], 'automatic-id-2 central.size')
        self.assertEqual(parameter_names[4], 'Mean central.size')
        self.assertEqual(parameter_names[5], 'Std. central.size')
        self.assertEqual(parameter_names[6], 'Pooled dose.absorption_rate')
        self.assertEqual(
            parameter_names[7], 'automatic-id-1 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[8], 'automatic-id-2 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[9], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[11], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[12], 'Pooled dose.drug_amount Sigma rel.')

        # Test case IV: Exclude bottom-level with IDs
        parameter_names = self.log_posterior.get_parameter_names(
            exclude_bottom_level=True, include_ids=True)

        self.assertEqual(len(parameter_names), 11)
        self.assertEqual(parameter_names[0], 'Pooled central.drug_amount')
        self.assertEqual(parameter_names[1], 'Pooled dose.drug_amount')
        self.assertEqual(parameter_names[2], 'Mean central.size')
        self.assertEqual(parameter_names[3], 'Std. central.size')
        self.assertEqual(parameter_names[4], 'Pooled dose.absorption_rate')
        self.assertEqual(
            parameter_names[5], 'automatic-id-1 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[6], 'automatic-id-2 myokit.elimination_rate')
        self.assertEqual(
            parameter_names[7], 'Pooled central.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[8], 'Pooled central.drug_amount Sigma rel.')
        self.assertEqual(
            parameter_names[9], 'Pooled dose.drug_amount Sigma base')
        self.assertEqual(
            parameter_names[10], 'Pooled dose.drug_amount Sigma rel.')

    def test_n_parameters(self):
        # Test case I: All parameters
        # 9 individual parameters, from which 1 is modelled heterogeneously,
        # 1 log-normally and the rest is pooled
        # And there are 2 individuals
        n_parameters = 2 + 4 + 1 + 1 + 1 + 1 + 1 + 1 + 1
        self.assertEqual(
            self.log_posterior.n_parameters(), n_parameters)

        # Test case II: Exclude bottom parameters
        n_parameters = self.log_posterior.n_parameters(
            exclude_bottom_level=True)
        self.assertEqual(n_parameters, 11)


class TestLogLikelihood(unittest.TestCase):
    """
    Tests the erlotinib.LogLikelihood class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        obs_1 = [1, 1.1, 1.2, 1.3]
        times_1 = [1, 2, 3, 4]
        obs_2 = [2, 2.1, 2.2]
        times_2 = [2, 5, 6]

        cls.observations = [obs_1, obs_2]
        cls.times = [times_1, times_2]

        # Set up mechanistic and error models
        path = erlo.ModelLibrary().one_compartment_pk_model()
        cls.model = erlo.PharmacokineticModel(path)
        cls.model.set_administration('central', direct=False)
        cls.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])
        cls.error_models = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()] * 2

        # Create log-likelihood
        cls.log_likelihood = erlo.LogLikelihood(
            cls.model, cls.error_models, cls.observations, cls.times)

    def test_instantiation(self):
        # Check whether changing the model changes the log-likelihood
        obs = [1, 1.1, 1.2, 1.3]
        times = [1, 2, 3, 4]
        path = erlo.ModelLibrary().one_compartment_pk_model()
        model = erlo.PharmacokineticModel(path)
        error_model = [
            erlo.ConstantAndMultiplicativeGaussianErrorModel()]
        log_likelihood = erlo.LogLikelihood(
            model, error_model, obs, times)
        m = log_likelihood.get_submodels()['Mechanistic model']
        self.assertEqual(model.n_parameters(), 3)
        self.assertEqual(m.n_parameters(), 3)

        model.set_administration('central', direct=False)
        m = log_likelihood.get_submodels()['Mechanistic model']
        self.assertEqual(model.n_parameters(), 5)
        self.assertEqual(m.n_parameters(), 3)

    def test_bad_instantiation(self):
        # Mechantic model has wrong type
        mechanistic_model = 'wrong type'
        with self.assertRaisesRegex(TypeError, 'The mechanistic model'):
            erlo.LogLikelihood(
                mechanistic_model, self.error_models, self.observations,
                self.times)

        # Wrong number of error models
        outputs = ['central.drug_amount']
        with self.assertRaisesRegex(ValueError, 'One error model has'):
            erlo.LogLikelihood(
                self.model, self.error_models, self.observations, self.times,
                outputs)

        # Wrong number of error models
        error_models = ['Wrong', 'type']
        with self.assertRaisesRegex(TypeError, 'The error models have to'):
            erlo.LogLikelihood(
                self.model, error_models, self.observations, self.times)

        # Wrong length of observations
        observations = [['There'], ['are'], ['only two outputs']]
        with self.assertRaisesRegex(ValueError, 'The observations have'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, self.times)

        # Wrong length of times
        times = [['There'], ['are'], ['only two outputs']]
        with self.assertRaisesRegex(ValueError, 'The times have the wrong'):
            erlo.LogLikelihood(
                self.model, self.error_models, self.observations, times)

        # Negative times
        observations = [[1, 2], [1, 2]]
        times = [[-1, 2], [1, 2]]
        with self.assertRaisesRegex(ValueError, 'Times cannot be negative'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, times)

        # Not strictly increasing times
        observations = [[1, 2], [1, 2]]
        times = [[2, 1], [1, 2]]
        with self.assertRaisesRegex(ValueError, 'Times must be increasing.'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, times)

        # Observations and times don't match
        observations = [[1, 2], [1, 2]]  # Times have 4 and 3
        with self.assertRaisesRegex(ValueError, 'The observations and times'):
            erlo.LogLikelihood(
                self.model, self.error_models, observations, self.times)

    def test_call_and_compute_pointwise_ll(self):
        # Test case I: Compute reference score manually
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]

        times = self.times[0]
        observations = self.observations[0]
        model_output = self.model.simulate(parameters[:5], times)
        model_output = model_output[0]
        error_model = self.error_models[0]
        ref_score_1 = error_model.compute_log_likelihood(
            parameters[5:7], model_output, observations)

        times = self.times[1]
        observations = self.observations[1]
        model_output = self.model.simulate(parameters[:5], times)
        model_output = model_output[1]
        error_model = self.error_models[1]
        ref_score_2 = error_model.compute_log_likelihood(
            parameters[7:9], model_output, observations)

        ref_score = ref_score_1 + ref_score_2
        score = self.log_likelihood(parameters)
        pw_score = self.log_likelihood.compute_pointwise_ll(
            parameters)

        self.assertAlmostEqual(score, ref_score)
        n_obs = 7
        self.assertEqual(pw_score.shape, (n_obs,))
        self.assertEqual(np.sum(pw_score), score)

        # Test case II: Compute reference score with two likelihoods
        parameters = [9, 8, 7, 6, 5, 4, 3, 2, 1]

        times = self.times[0]
        observations = self.observations[0]
        self.model.set_outputs(['central.drug_amount'])
        error_model = self.error_models[0]
        log_likelihood = erlo.LogLikelihood(
            self.model, error_model, observations, times)
        ref_score_1 = log_likelihood(parameters[:7])

        times = self.times[1]
        observations = self.observations[1]
        self.model.set_outputs(['dose.drug_amount'])
        error_model = self.error_models[1]
        log_likelihood = erlo.LogLikelihood(
            self.model, error_model, observations, times)
        ref_score_2 = log_likelihood(parameters[:5] + parameters[7:9])

        ref_score = ref_score_1 + ref_score_2
        score = self.log_likelihood(parameters)
        pw_score = self.log_likelihood.compute_pointwise_ll(
            parameters)

        self.assertAlmostEqual(score, ref_score)
        n_obs = 7
        self.assertEqual(pw_score.shape, (n_obs,))
        self.assertEqual(np.sum(pw_score), score)

        # Reset number of outputs
        self.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])

        # Make sure that call works even when sensitivities were initially
        # switched on
        m = self.log_likelihood.get_submodels()['Mechanistic model']
        m.enable_sensitivities(True)
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.log_likelihood(parameters)
        m.enable_sensitivities(True)
        self.log_likelihood.compute_pointwise_ll(parameters)

        # Leave observations for one outputs empty
        obs = [[], self.observations[1]]
        times = [[], self.times[1]]
        ll = erlo.LogLikelihood(
            self.model, self.error_models, obs, times)

        parameters = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        score = ll(parameters)
        pw_score = ll.compute_pointwise_ll(parameters)
        self.assertEqual(score, ref_score_2)
        n_obs = 3
        self.assertEqual(pw_score.shape, (n_obs,))
        self.assertEqual(np.sum(pw_score), score)

    def test_evaluateS1(self):
        # Test case I: Compute reference score manually
        parameters = [1, 1, 1, 1, 1, 1, 1, 1, 1]
        times = self.times[0]
        observations = self.observations[0]
        self.model.enable_sensitivities(True)
        model_output, model_sens = self.model.simulate(
            parameters[:5], times)
        model_output = model_output[0]
        model_sens = model_sens[:, 0, :]
        error_model = self.error_models[0]
        ref_score_1, ref_sens_1 = error_model.compute_sensitivities(
            parameters[5:7], model_output, model_sens, observations)
        times = self.times[1]
        observations = self.observations[1]
        model_output, model_sens = self.model.simulate(
            parameters[:5], times)
        model_output = model_output[1]
        model_sens = model_sens[:, 1, :]
        error_model = self.error_models[1]
        ref_score_2, ref_sens_2 = error_model.compute_sensitivities(
            parameters[7:9], model_output, model_sens, observations)

        ref_score = ref_score_1 + ref_score_2
        score, sens = self.log_likelihood.evaluateS1(parameters)

        self.assertAlmostEqual(score, ref_score)
        self.assertEqual(len(sens), 9)
        ref_dpsi = ref_sens_1[:5] + ref_sens_2[:5]
        self.assertAlmostEqual(sens[0], ref_dpsi[0])
        self.assertAlmostEqual(sens[1], ref_dpsi[1])
        self.assertAlmostEqual(sens[2], ref_dpsi[2])
        self.assertAlmostEqual(sens[3], ref_dpsi[3])
        self.assertAlmostEqual(sens[4], ref_dpsi[4])
        self.assertAlmostEqual(sens[5], ref_sens_1[5])
        self.assertAlmostEqual(sens[6], ref_sens_1[6])
        self.assertAlmostEqual(sens[7], ref_sens_2[5])
        self.assertAlmostEqual(sens[8], ref_sens_2[6])

    def test_fix_parameters(self):
        # Test case I: fix some parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': 1,
            'dose.absorption_rate': 1})

        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 7)

        parameter_names = self.log_likelihood.get_parameter_names()
        self.assertEqual(len(parameter_names), 7)
        self.assertEqual(parameter_names[0], 'dose.drug_amount')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[3], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[4], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[5], 'dose.drug_amount Sigma base')
        self.assertEqual(parameter_names[6], 'dose.drug_amount Sigma rel.')

        # Test case II: fix overlapping set of parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'dose.absorption_rate': None,
            'dose.drug_amount Sigma base': 0.5,
            'myokit.elimination_rate': 0.3})

        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 6)

        parameter_names = self.log_likelihood.get_parameter_names()
        self.assertEqual(len(parameter_names), 6)
        self.assertEqual(parameter_names[0], 'dose.drug_amount')
        self.assertEqual(parameter_names[1], 'central.size')
        self.assertEqual(parameter_names[2], 'dose.absorption_rate')
        self.assertEqual(parameter_names[3], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[4], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[5], 'dose.drug_amount Sigma rel.')

        # Test case III: unfix all parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': None,
            'dose.drug_amount Sigma base': None,
            'myokit.elimination_rate': None})

        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 9)

        parameter_names = self.log_likelihood.get_parameter_names()
        self.assertEqual(len(parameter_names), 9)
        self.assertEqual(parameter_names[0], 'central.drug_amount')
        self.assertEqual(parameter_names[1], 'dose.drug_amount')
        self.assertEqual(parameter_names[2], 'central.size')
        self.assertEqual(parameter_names[3], 'dose.absorption_rate')
        self.assertEqual(parameter_names[4], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[5], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[6], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[7], 'dose.drug_amount Sigma base')
        self.assertEqual(parameter_names[8], 'dose.drug_amount Sigma rel.')

    def test_fix_parameters_bad_input(self):
        name_value_dict = 'Bad type'
        with self.assertRaisesRegex(ValueError, 'The name-value dictionary'):
            self.log_likelihood.fix_parameters(name_value_dict)

    def test_get_parameter_names(self):
        # Test case I: Single output problem
        parameter_names = self.log_likelihood.get_parameter_names()

        self.assertEqual(len(parameter_names), 9)
        self.assertEqual(parameter_names[0], 'central.drug_amount')
        self.assertEqual(parameter_names[1], 'dose.drug_amount')
        self.assertEqual(parameter_names[2], 'central.size')
        self.assertEqual(parameter_names[3], 'dose.absorption_rate')
        self.assertEqual(parameter_names[4], 'myokit.elimination_rate')
        self.assertEqual(parameter_names[5], 'central.drug_amount Sigma base')
        self.assertEqual(parameter_names[6], 'central.drug_amount Sigma rel.')
        self.assertEqual(parameter_names[7], 'dose.drug_amount Sigma base')
        self.assertEqual(parameter_names[8], 'dose.drug_amount Sigma rel.')

    def test_get_set_id(self):
        # Test case I: Check default
        self.assertIsNone(self.log_likelihood.get_id())

        # Test case II: Set ID
        _id = '123'
        self.log_likelihood.set_id(_id)
        self.assertEqual(self.log_likelihood.get_id(), 'ID ' + _id)

    def test_get_submodels(self):
        # Test case I: no fixed parameters
        submodels = self.log_likelihood.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 2)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)
        self.assertIsInstance(error_models[1], erlo.ErrorModel)

        # Test case II: some fixed parameters
        self.log_likelihood.fix_parameters(name_value_dict={
            'central.drug_amount': 1,
            'dose.drug_amount Sigma base': 1})
        submodels = self.log_likelihood.get_submodels()

        keys = list(submodels.keys())
        self.assertEqual(len(keys), 2)
        self.assertEqual(keys[0], 'Mechanistic model')
        self.assertEqual(keys[1], 'Error models')

        mechanistic_model = submodels['Mechanistic model']
        self.assertIsInstance(mechanistic_model, erlo.MechanisticModel)

        error_models = submodels['Error models']
        self.assertEqual(len(error_models), 2)
        self.assertIsInstance(error_models[0], erlo.ErrorModel)
        self.assertIsInstance(error_models[1], erlo.ErrorModel)

        # Unfix parameter
        self.log_likelihood.fix_parameters({
            'central.drug_amount': None,
            'dose.drug_amount Sigma base': None})

    def test_n_parameters(self):
        # Test case I:
        n_parameters = self.log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 9)

        # Test case II:
        times = self.times[0]
        observations = self.observations[0]
        self.model.set_outputs(['central.drug_amount'])
        error_model = self.error_models[0]
        log_likelihood = erlo.LogLikelihood(
            self.model, error_model, observations, times)

        n_parameters = log_likelihood.n_parameters()
        self.assertEqual(n_parameters, 7)

        # Reset number of outputs
        self.model.set_outputs(['central.drug_amount', 'dose.drug_amount'])

    def test_n_observations(self):
        # Test case I:
        n_obs = self.log_likelihood.n_observations()

        self.assertEqual(len(n_obs), 2)
        self.assertEqual(n_obs[0], 4)
        self.assertEqual(n_obs[1], 3)


class TestLogPosterior(unittest.TestCase):
    """
    Tests the erlotinib.LogPosterior class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test dataset
        times = [0, 1, 2, 3]
        values = [10, 11, 12, 13]

        # Create test model
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        error_model = erlo.ConstantAndMultiplicativeGaussianErrorModel()
        cls.log_likelihood = erlo.LogLikelihood(
            model, error_model, values, times)
        cls.log_likelihood.set_id('42')
        cls.log_prior = pints.ComposedLogPrior(
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1),
            pints.UniformLogPrior(0, 1))
        cls.log_posterior = erlo.LogPosterior(
            cls.log_likelihood, cls.log_prior)

    def test_bad_instantiation(self):
        # Log-likelihood has bad type
        log_likelihood = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-likelihood has to'):
            erlo.LogPosterior(log_likelihood, self.log_prior)

        # Log-prior has bad type
        log_prior = 'bad type'
        with self.assertRaisesRegex(TypeError, 'The log-prior has to'):
            erlo.LogPosterior(self.log_likelihood, log_prior)

        # The dimensionality of likelihood and prior don't match
        log_prior = pints.UniformLogPrior(0, 1)
        with self.assertRaisesRegex(ValueError, 'The log-prior and the'):
            erlo.LogPosterior(self.log_likelihood, log_prior)

    def test_call(self):
        parameters = [1, 2, 3, 4, 5, 6, 7]
        ref_score = \
            self.log_likelihood(parameters) + self.log_prior(parameters)
        score = self.log_posterior(parameters)
        self.assertEqual(score, ref_score)

    def test_evaluateS1(self):
        parameters = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        ref_score_1, ref_sens_1 = self.log_likelihood.evaluateS1(parameters)
        ref_score_2, ref_sens_2 = self.log_prior.evaluateS1(parameters)
        ref_score = ref_score_1 + ref_score_2
        ref_sens = ref_sens_1 + ref_sens_2
        score, sens = self.log_posterior.evaluateS1(parameters)
        self.assertEqual(score, ref_score)
        self.assertEqual(len(sens), 7)
        self.assertEqual(len(ref_sens), 7)
        self.assertEqual(sens[0], ref_sens[0])
        self.assertEqual(sens[1], ref_sens[1])
        self.assertEqual(sens[2], ref_sens[2])
        self.assertEqual(sens[3], ref_sens[3])
        self.assertEqual(sens[4], ref_sens[4])
        self.assertEqual(sens[5], ref_sens[5])
        self.assertEqual(sens[6], ref_sens[6])

        # Test that it returns inf when parameters outside
        # priors
        parameters = [-1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        score, _ = self.log_posterior.evaluateS1(parameters)
        self.assertEqual(score, -np.inf)

    def test_get_id(self):
        # Test case I: Non-trivial IDs
        _id = self.log_posterior.get_id()
        self.assertEqual(_id, 'ID 42')

    def test_get_log_likelihood(self):
        log_likelihood = self.log_posterior.get_log_likelihood()
        self.assertIsInstance(log_likelihood, erlo.LogLikelihood)

    def test_get_parameter_names(self):
        # Test case I: Non-trivial parameters
        parameter_names = self.log_posterior.get_parameter_names()

        self.assertEqual(len(parameter_names), 7)
        self.assertEqual(parameter_names[0], 'myokit.tumour_volume')
        self.assertEqual(parameter_names[1], 'myokit.drug_concentration')
        self.assertEqual(parameter_names[2], 'myokit.kappa')
        self.assertEqual(parameter_names[3], 'myokit.lambda_0')
        self.assertEqual(parameter_names[4], 'myokit.lambda_1')
        self.assertEqual(parameter_names[5], 'Sigma base')
        self.assertEqual(parameter_names[6], 'Sigma rel.')


class TestReducedLogPDF(unittest.TestCase):
    """
    Tests the erlotinib.ReducedLogPDF class.
    """

    @classmethod
    def setUpClass(cls):
        # Create test data
        times = [1, 2, 3, 4, 5]
        values = [1, 2, 3, 4, 5]

        # Set up inverse problem
        path = erlo.ModelLibrary().tumour_growth_inhibition_model_koch()
        model = erlo.PharmacodynamicModel(path)
        problem = erlo.InverseProblem(model, times, values)
        cls.log_likelihood = pints.GaussianLogLikelihood(problem)
        cls.mask = [True, False, False, True, False, True]
        cls.values = [11, 12, 13]
        cls.reduced_log_pdf = erlo.ReducedLogPDF(
            cls.log_likelihood, cls.mask, cls.values)

    def test_bad_input(self):
        # Wrong log-pdf
        log_pdf = 'Bad type'

        with self.assertRaisesRegex(ValueError, 'The log-pdf has to'):
            erlo.ReducedLogPDF(log_pdf, self.mask, self.values)

        # Mask is not as long as the number of parameyers
        mask = [True, True]

        with self.assertRaisesRegex(ValueError, 'Length of mask has to'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # Mask is not boolean
        mask = ['yes', 'no', 'yes', 'yes', 'yes', 'yes']

        with self.assertRaisesRegex(ValueError, 'Mask has to be a'):
            erlo.ReducedLogPDF(self.log_likelihood, mask, self.values)

        # There are not as many input values as fixed parameters
        values = [1]

        with self.assertRaisesRegex(ValueError, 'There have to be'):
            erlo.ReducedLogPDF(self.log_likelihood, self.mask, values)

    def test_call(self):
        parameters = np.array([11, 1, 1, 12, 1, 13])
        reduced_params = parameters[~np.array(self.mask)]

        self.assertEqual(
            self.reduced_log_pdf(reduced_params),
            self.log_likelihood(parameters))

    def test_n_parameters(self):
        before = self.log_likelihood.n_parameters()
        n_fixed = np.sum(self.mask)

        self.assertEqual(
            self.reduced_log_pdf.n_parameters(), before - n_fixed)


if __name__ == '__main__':
    unittest.main()
