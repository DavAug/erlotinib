import argparse


# Set up argument parsing, so plotting and exports can be disabled for
# testing.
parser = argparse.ArgumentParser(
    description='Run example scripts for chi docs.',
)
parser.add_argument(
    '--test',
    action='store_true',
    help='Run testing version of script which ignores plotting.',)

# Parse!
args = parser.parse_args()


import os

import chi
import numpy as np
import pandas as pd
import pints
import plotly.graph_objects as go

# 3
import pints


# Define mechanistic model
directory = os.path.dirname(__file__)
filename = os.path.join(directory, 'one_compartment_pk_model.xml')
model = chi.PKPDModel(sbml_file=filename)
model.set_administration(compartment='global', direct=False)
model.set_outputs(['global.drug_concentration'])

# Define error model
error_model = chi.LogNormalErrorModel()

# Define data
directory = os.path.dirname(__file__)
data = pd.read_csv(directory + '/dataset_1.csv')

# Define prior distribution
log_prior = pints.ComposedLogPrior(
    pints.GaussianLogPrior(mean=10, sd=2),           # absorption rate
    pints.GaussianLogPrior(mean=6, sd=2),            # elimination rate
    pints.LogNormalLogPrior(log_mean=0, scale=1),    # volume of distribution
    pints.LogNormalLogPrior(log_mean=-2, scale=0.5)  # scale of meas. distrib.
)

# Define log-posterior using the ProblemModellingController
problem = chi.ProblemModellingController(
    mechanistic_model=model, error_models=[error_model])
problem.set_data(
    data=data,
    output_observable_dict={'global.drug_concentration': 'Drug concentration'}
)
problem.fix_parameters(name_value_dict={
    'dose.drug_amount': 0,
    'global.drug_amount': 0,
})
problem.set_log_prior(log_prior=log_prior)
log_posterior = problem.get_log_posterior()


# 4
# Run MCMC algorithm
n_iterations = 20000
controller = chi.SamplingController(log_posterior=log_posterior, seed=1)
controller.set_n_runs(n_runs=3)
controller.set_parallel_evaluation(False)
controller.set_sampler(pints.HaarioBardenetACMC)
samples = controller.run(n_iterations=n_iterations, log_to_screen=True)

# 5
from plotly.colors import qualitative
from plotly.subplots import make_subplots


# Plot results
fig = make_subplots(
    rows=4, cols=2, vertical_spacing=0.15, horizontal_spacing=0.1)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Scatter(
        name='Run 1',
        x=np.arange(1, n_iterations+1),
        y=samples['dose.absorption_rate'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        x=np.arange(1, n_iterations+1),
        y=samples['dose.absorption_rate'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        x=np.arange(1, n_iterations+1),
        y=samples['dose.absorption_rate'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
    ),
    row=1,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=samples['dose.absorption_rate'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=True,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[4],
    ),
    row=1,
    col=2
)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Scatter(
        name='Run 1',
        x=np.arange(1, n_iterations+1),
        y=samples['global.elimination_rate'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        x=np.arange(1, n_iterations+1),
        y=samples['global.elimination_rate'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        x=np.arange(1, n_iterations+1),
        y=samples['global.elimination_rate'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
        showlegend=False
    ),
    row=2,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=samples['global.elimination_rate'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=False,
        xbins=dict(size=0.2),
    marker_color=qualitative.Plotly[4],
    ),
    row=2,
    col=2
)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Scatter(
        name='Run 1',
        x=np.arange(1, n_iterations+1),
        y=samples['global.volume'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
        showlegend=False
    ),
    row=3,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        x=np.arange(1, n_iterations+1),
        y=samples['global.volume'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
        showlegend=False
    ),
    row=3,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        x=np.arange(1, n_iterations+1),
        y=samples['global.volume'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
        showlegend=False
    ),
    row=3,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=samples['global.volume'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=False,
        xbins=dict(size=0.5),
    marker_color=qualitative.Plotly[4],
    ),
    row=3,
    col=2
)

# Plot traces and histogram of parameter
fig.add_trace(
    go.Scatter(
        name='Run 1',
        x=np.arange(1, n_iterations+1),
        y=samples['Sigma log'].values[0],
        mode='lines',
        line_color=qualitative.Plotly[2],
        showlegend=False
    ),
    row=4,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 2',
        x=np.arange(1, n_iterations+1),
        y=samples['Sigma log'].values[1],
        mode='lines',
        line_color=qualitative.Plotly[1],
        showlegend=False
    ),
    row=4,
    col=1
)
fig.add_trace(
    go.Scatter(
        name='Run 3',
        x=np.arange(1, n_iterations+1),
        y=samples['Sigma log'].values[2],
        mode='lines',
        line_color=qualitative.Plotly[0],
        showlegend=False
    ),
    row=4,
    col=1
)
fig.add_trace(
    go.Histogram(
        name='Posterior samples',
        x=samples['Sigma log'].values[:, n_iterations//2::(n_iterations//2)//1000].flatten(),
        histnorm='probability density',
        showlegend=False,
        xbins=dict(size=0.02),
    marker_color=qualitative.Plotly[4],
    ),
    row=4,
    col=2
)

# Visualise prior distribution
parameter_values = np.linspace(4, 16, num=200)
pdf_values = np.exp([
    log_prior._priors[0]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
    ),
    row=1,
    col=2
)

parameter_values = np.linspace(0, 12, num=200)
pdf_values = np.exp([
    log_prior._priors[1]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=2,
    col=2
)

parameter_values = np.linspace(0, 12, num=200)
pdf_values = np.exp([
    log_prior._priors[2]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=3,
    col=2
)

parameter_values = np.linspace(0, 0.6, num=200)
pdf_values = np.exp([
    log_prior._priors[3]([parameter_value])
    for parameter_value in parameter_values])
fig.add_trace(
    go.Scatter(
        name='Prior distribution',
        x=parameter_values,
        y=pdf_values,
        mode='lines',
        line_color='black',
        showlegend=False
    ),
    row=4,
    col=2
)

fig.update_layout(
    yaxis_title='k_a',
    xaxis2_title='k_a',
    yaxis2_title='p',
    yaxis3_title='k_e',
    xaxis4_title='k_e',
    yaxis4_title='p',
    yaxis5_title='v',
    xaxis6_title='v',
    yaxis6_title='p',
    xaxis7_title='Number of iterations',
    yaxis7_title='sigma',
    xaxis8_title='sigma',
    yaxis8_title='p',
    template='plotly_white',
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    margin=dict(t=0, r=0, l=0)
)
fig.show()

directory = os.path.dirname(os.path.dirname(__file__))
fig.write_html(directory + '/images/3_fitting_models_3.html')