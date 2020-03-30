# http://localhost:8050/

import datetime
import os
import yaml

import numpy as np
import pandas as pd

from scipy.integrate import solve_ivp
from scipy.optimize import minimize

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dash
import dash_core_components as dcc
import dash_daq as daq
import dash_html_components as html
from dash.dependencies import Input, Output

from scipy.optimize import minimize


def get_country(self, country3):
    return (epidemie_df[epidemie_df['Country/Region'] == country3]
            .groupby(['Country/Region', 'day'])
            .agg({'Confirmed': 'sum', 'Deaths': 'sum', 'Recovered': 'sum'})
            .reset_index()
           )
pd.DataFrame.get_country = get_country

# Lecture du fichier d'environnement
ENV_FILE = '../env.yaml'
with open(ENV_FILE) as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Initialisation des chemins vers les fichiers
ROOT_DIR = os.path.dirname(os.path.abspath(ENV_FILE))
DATA_FILE = os.path.join(ROOT_DIR,
                         params['directories']['processed'],
                         params['files']['all_data'])

# Lecture du fichier de données
epidemie_df = (pd.read_csv(DATA_FILE, parse_dates=['Last Update'])
               .assign(day=lambda _df: _df['Last Update'].dt.date)
               .drop_duplicates(subset=['Country/Region', 'Province/State', 'day'])
               [lambda df: df['day'] <= datetime.date(2020, 3, 10)]
              )

countries = [{'label': c, 'value': c} for c in sorted(epidemie_df['Country/Region'].unique())]

app = dash.Dash('Corona Virus Explorer')
app.layout = html.Div([
    html.H1(['Corona Virus Explorer'], style={'textAlign': 'center'}),
    dcc.Tabs([
        dcc.Tab(label='Time', children=[
            html.Div([
                dcc.Dropdown(
                    id='country',
                    options=countries
                )
            ]),
            html.Div([
                dcc.Dropdown(
                    id='country2',
                    options=countries
                )
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable',
                    options=[
                        {'label': 'Confirmed', 'value': 'Confirmed'},
                        {'label': 'Deaths', 'value': 'Deaths'},
                        {'label': 'Recovered', 'value': 'Recovered'}
                    ],
                    value='Confirmed',
                    labelStyle={'display': 'inline-block'}
                )
            ]),
            html.Div([
                dcc.Graph(id='graph1')
            ]),   
        ]),
        dcc.Tab(label='Map', children=[
            dcc.Graph(id='map1'),
            dcc.Slider(
                id='map_day',
                min=0,
                max=(epidemie_df['day'].max() - epidemie_df['day'].min()).days,
                value=0,
                #marks={i:str(date) for i, date in enumerate(epidemie_df['day'].unique())}
                marks={i:str(i) for i, date in enumerate(epidemie_df['day'].unique())}
            )  
        ]),
        dcc.Tab(label='Modelisation', children=[
            html.Div([
                dcc.Dropdown(
                    id='country3',
                    options=countries
                )
            ]),
            html.Div([
                dcc.RadioItems(
                    id='variable2',
                    options=[
                        {'label': 'Susceptible', 'value': 'Susceptible'},
                        {'label': 'Recovered', 'value': 'Recovered'},
                        {'label': 'Infected', 'value': 'Infected'}
                    ],
                    value='Infected',
                    labelStyle={'display': 'inline-block'}
                ),
                dcc.RadioItems(
                    id='variable6',
                    options=[
                        {'label': 'Optimisation', 'value': 'Optimisation'},
                        {'label': 'No optimisation', 'value': 'No optimisation'}
                    ],
                    value='No optimisation',
                    labelStyle={'display': 'inline-block'}
                ),
                daq.NumericInput(
                    id='variable3',
                    label='Population',
                    size=250,
                    max=3000000000,
                    value=1000
                ),
                daq.NumericInput(
                    id='variable4',
                    label='Beta',
                    size=75,
                    max=1,
                    min=0.0001,
                    value=0.01
                ),
                 daq.NumericInput(
                    id='variable5',
                    label='Gamma',
                    size=75,
                    max=100,
                    min=0.001,
                    value=0.1
                )
            ]),
            html.Div([
                dcc.Graph(id='graph2')
            ]),  
        ]),
    ]),
])

@app.callback(
    Output('graph1', 'figure'),
    [
        Input('country', 'value'),
        Input('country2', 'value'),
        Input('variable', 'value'),        
    ]
)
def update_graph(country, country2, variable):
    print(country)
    if country is None:
        graph_df = epidemie_df.groupby('day').agg({variable: 'sum'}).reset_index()
    else:
        graph_df = (epidemie_df[epidemie_df['Country/Region'] == country]
                    .groupby(['Country/Region', 'day'])
                    .agg({variable: 'sum'})
                    .reset_index()
                   )
    if country2 is not None:
        graph2_df = (epidemie_df[epidemie_df['Country/Region'] == country2]
                     .groupby(['Country/Region', 'day'])
                     .agg({variable: 'sum'})
                     .reset_index()
                    )

        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    return {
        'data': [
            dict(
                x=graph_df['day'],
                y=graph_df[variable],
                type='line',
                name=country if country is not None else 'Total'
            )
        ] + ([
            dict(
                x=graph2_df['day'],
                y=graph2_df[variable],
                type='line',
                name=country2
            )            
        ] if country2 is not None else [])
    }

@app.callback(
    Output('map1', 'figure'),
    [
        Input('map_day', 'value'),
    ]
)
def update_map(map_day):
    day = epidemie_df['day'].unique()[map_day]
    map_df = (epidemie_df[epidemie_df['day'] == day]
              .groupby(['Country/Region'])
              .agg({'Confirmed': 'sum', 'Latitude': 'mean', 'Longitude': 'mean'})
              .reset_index()
             )
    print(map_day)
    print(day)
    print(map_df.head())
    return {
        'data': [
            dict(
                type='scattergeo',
                lon=map_df['Longitude'],
                lat=map_df['Latitude'],
                text=map_df.apply(lambda r: r['Country/Region'] + ' (' + str(r['Confirmed']) + ')', axis=1),
                mode='markers',
                marker=dict(
                    size=np.maximum(map_df['Confirmed'] / 1_000, 5)
                )
            )
        ],
        'layout': dict(
            title=str(day),
            geo=dict(showland=True),
        )
    }

@app.callback(
    Output('graph2', 'figure'),
    [
        Input('country3', 'value'),
        Input('variable2', 'value'), 
        Input('variable3', 'value'), 
        Input('variable4', 'value'), 
        Input('variable5', 'value'), 
        Input('variable6', 'value'), 
    ]
)


    

def update_graph2(country3, variable2, variable3, variable4, variable5, variable6):
    print(country3)
    
    country3_df = epidemie_df.get_country('country3')
    country3_df['infected'] = country3_df['Confirmed'].diff()
    
    if variable6 is 'Optimisation':
    
        def sumsq_error(parameters):
            beta, gamma = parameters
    
            def SIRopt(t, y):
                S = y[0]
                I = y[1]
                R = y[2]
                return([-beta*S*I, beta*S*I-gamma*I, gamma*I])

            solution = solve_ivp(SIRopt, [0, nb_steps-1], [total_population, 1, 0], t_eval=np.arange(0, nb_steps, 1))
    
            return(sum((solution.y[1]-infected_population)**2))

        total_population = variable3
        infected_population = country3_df.loc[2:]['infected']
        nb_steps = len(infected_population)
        x0 = np.array([0.001, 0.1])

        msol = minimize(sumsq_error, (0.001, 0.1), method='Nelder-Mead', callback=callback, options={'disp': True})
        variable4=msol.x[0]
        variable5=msol.x[1]
    

    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return([-variable4*S*I, variable4*S*I-variable5*I, variable5*I])

    
    solution_country3 = solve_ivp(SIR, [0, 40], [variable3, 1, 0], t_eval=np.arange(0, 40, 1))
    data_solv = {'day': solution_country3.t,'Susceptible': solution_country3.y[0], 'Infected': solution_country3.y[1], 'Recovered': solution_country3.y[2]}
    graph3_df = pd.DataFrame(data_solv, columns = ['day','Infected','Recovered','Susceptible'])
    

        
    #data : [dict(...graph_df...)] + ([dict(...graph2_df)] if country2 is not None else [])
        
    return {
        'data': [
            dict(
                x=graph3_df['day'],
                y=graph3_df[variable2],
                type='line',
                name=country3 if country3 is not None else 'Total'
            )
        ]
    }



if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
    