import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from IPython.display import IFrame, display, HTML
import chart_studio.plotly as py
import plotly.graph_objects as go
import json
from urllib.request import urlopen
import requests
from io import BytesIO
import sklearn
import xgboost as xgb
import joblib
import warnings
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score, median_absolute_error
from dash import dash_table
import dash_leaflet as dl
import geopandas as gpd

warnings.filterwarnings("ignore")

custom_palette=['#1F3040','#B9CDCA','#F2C6AC', '#D99982', '#735749']

# Lee los datos desde el archivo CSV
df = pd.read_csv('https://raw.githubusercontent.com/gbuvoli/HEBO/main/datosventas%20-%20datosventas.csv')
with urlopen('https://raw.githubusercontent.com/lihkir/Uninorte/main/AppliedStatisticMS/DataVisualizationRPython/Lectures/Python/PythonDataSets/Colombia.geo.json') as response:
    counties = json.load(response)

coord=pd.read_csv('https://raw.githubusercontent.com/gbuvoli/HEBO/main/Datos_lat_long_Agrupados%20.csv')

fi_lasso = pd.read_csv("https://raw.githubusercontent.com/gbuvoli/HEBO/main/FeatureImportances_lasso.csv")
fi_ridge = pd.read_csv("https://raw.githubusercontent.com/gbuvoli/HEBO/main/FeatureImportances_ridge.csv")
fi_xgboost = pd.read_csv("https://raw.githubusercontent.com/gbuvoli/HEBO/main/FeatureImportances_xgboost.csv")
fi_mlp = pd.read_csv("https://raw.githubusercontent.com/gbuvoli/HEBO/main/FeatureImportances_mlp.csv")

lasso=pd.read_csv('https://github.com/gbuvoli/HEBO/raw/main/real_vs_predicted_lasso.csv')
ridge=pd.read_csv('https://github.com/gbuvoli/HEBO/raw/main/real_vs_predicted_ridge.csv')
xgboost=pd.read_csv('https://raw.githubusercontent.com/gbuvoli/HEBO/main/real_vs_predicted_XGBoots.csv')
mlp=pd.read_csv('https://raw.githubusercontent.com/gbuvoli/HEBO/main/real_vs_predicted_MLP.csv')


pricemax=df['price'].max()
pricemin=df['price'].min()
df['p_s']=df['price']/df['area_const']

#Lectura de modelos:
def modelo_github(url):
    response = requests.get(url)
    response.raise_for_status()  # Lanza una excepción si la solicitud no es exitosa
    return joblib.load(BytesIO(response.content))

url_modelo1='https://github.com/gbuvoli/HEBO/raw/main/mejor_modelo_lasso.joblib'
url_modelo2='https://github.com/gbuvoli/HEBO/raw/main/mejor_modelo_ridge.joblib'
url_modelo3='https://github.com/gbuvoli/HEBO/raw/main/mejor_modelo_xgboots.joblib'
url_modelo4='https://github.com/gbuvoli/HEBO/raw/main/mejor_modelo_mpl.joblib'

model_lasso = modelo_github(url_modelo1)
model_ridge = modelo_github(url_modelo2)
model_xgboost = modelo_github(url_modelo3)
model_mlp= modelo_github(url_modelo4)

model_names = ['Lasso', 'Ridge', 'XGBoost', 'MLP']
metrics = []
for i, df_model in enumerate([lasso, ridge, xgboost, mlp]):
    mape = mean_absolute_percentage_error(df_model['Y Real'], df_model['Y Predicho'])
    mse = mean_squared_error(df_model['Y Real'], df_model['Y Predicho'])
    rmse= mse**0.5
    mae = mean_absolute_error(df_model['Y Real'], df_model['Y Predicho'])
    r2 = r2_score(df_model['Y Real'], df_model['Y Predicho'])
    
    # Obtener el nombre del modelo del DataFrame
    modelo_nombre = model_names[i]
    
    # Agregar métricas formateadas a la lista
    metrics.append({
        'Modelo': modelo_nombre,
        'MAPE': round(mape, 2),
        'RMSE': round(rmse, 2),
        'MAE': round(mae, 2),
        'R2': round(r2, 2),
    })

# Crear un DataFrame con las métricas
metrics_df = pd.DataFrame(metrics)


initial_lat = 10.9878
initial_lon = -74.7889




external_stylesheets = ['/assets/typografy.css',  # CSS externo
                        '/assets/header.css']

# Crea la aplicación Dash
app = dash.Dash(__name__,suppress_callback_exceptions=True,title='Inmobiliaria HEBO',external_stylesheets=external_stylesheets)

server = app.server

# Diseño de la aplicación
app.layout = html.Div([
    # ENCABEZADO
    html.Div([
        html.Img(src='https://github.com/gbuvoli/HEBO/blob/main/HEBO_dark.png?raw=true', style={'height': '130px', 'width': '130px'}),
        html.H1("HEBO INMOBILIARIA ", style={'margin-left': '20px' })
    ], style={'display':'inline-flex', 'align-items':'center', 'background-color':'#203140', 'width': '100%'}),

    # ELEMENTO PESTAÑAS
    dcc.Tabs(id='toptabs', value='About', children=[
        dcc.Tab(label='About', value='About'),
        dcc.Tab(label='Quiero Comprar', value='tab-comprar'),
        dcc.Tab(label='Quiero Vender', value='tab-vender'),
    ]),

    # OUTPUT - CONTENIDO PESTAÑAS
    html.Div(id='tabs-content'),
    dcc.Store(id='data_store'),
    dcc.Store(id='data_store2')
])

# Callback para el contenido de las pestañas
@app.callback(Output('tabs-content', 'children'),
              [Input('toptabs', 'value')])
def render_content(tab):
    if tab == 'About':
        return html.Div([
            html.Div([
                    
                    html.H3('Te damos la bienvenida a HEBO'),
                    
                    html.P([
                "Bienvenido a ", html.Strong("HEBO Inmobiliaria"), ", tu destino confiable para encontrar la casa de tus sueños en Colombia.", html.Br(), 
                "En HEBO, estamos comprometidos a brindarte una experiencia inmobiliaria única y sin complicaciones. ", html.Br(), 
                "Nuestra última iniciativa, una aplicación innovadora, busca revolucionar la forma en que buscas y vendes propiedades en todo el país. Esta aplicación no solo te permite explorar una amplia gama de viviendas en venta en diferentes ciudades colombianas, sino que también te ofrece la oportunidad de vender tu propia propiedad de manera eficiente. ", html.Br(), 
                "Con la integración de diversos modelos de ", html.Strong("machine learning"), ", nuestra aplicación proporciona", html.Strong("estimaciones precisas del precio de venta"),  " ayudándote a tomar decisiones informadas. ", html.Br(), "Desde casas amplias hasta apartamentos modernos, nuestro extenso conjunto de datos incluye una variedad de variables clave, como ", html.Strong("ubicación geográfica, características de la propiedad y precios actualizados"),". ¡Únete a nosotros en esta emocionante aventura y encuentra el lugar perfecto para llamar hogar en Colombia!"
                    ]),

                html.H3('Fuente de datos y variables'),
                    
                    html.P([

                        "Los datos utilizados en nuestra aplicación fueron obtenidos de Kaggle, específicamente del conjunto de datos titulado ",html.A( html.Strong("Data of properties for sale in Colombia from 2021"), href="https://www.kaggle.com/datasets/angeloftechml/data-of-properties-for-sale-in-colombia-from-2021"), " Este conjunto de datos, disponible en este enlace, fue recopilado mediante técnicas de scraping de las páginas web de MetroCuadrado y Finca Raíz, dos de los principales portales inmobiliarios en Colombia. Esta fuente ofrece una amplia variedad de información sobre propiedades en venta en Colombia, incluyendo detalles como ubicación geográfica, características de la propiedad y precios actualizados."
                
                    ]), 

                    html.Br(),
                    html.H3("Variables:"),
                    html.P([
                            html.Ul([
                                    html.Li([html.Strong("lat"), ": Latitud geográfica de la ubicación de la propiedad."]),
                                    html.Li([html.Strong("long"), ": Longitud geográfica de la ubicación de la propiedad."]),
                                    html.Li([html.Strong("category"), ": Categoría de la propiedad (casa, apartamento, apartaestudio)."]),
                                    html.Li([html.Strong("price"), ": Precio de venta de la propiedad en pesos colombianos."]),
                                    html.Li([html.Strong("rooms"), ": Número de habitaciones en la propiedad."]),
                                    html.Li([html.Strong("baths"), ": Número de baños en la propiedad."]),
                                    html.Li([html.Strong("park"), ": Número de espacios de estacionamiento disponibles."]),
                                    html.Li([html.Strong("ciudad"), ": Ciudad donde se encuentra la propiedad."]),
                                    html.Li([html.Strong("barrio"), ": Barrio o sector donde se encuentra la propiedad."]),
                                    html.Li([html.Strong("area_privada"), ": Área privada de la propiedad en metros cuadrados."]),
                                    html.Li([html.Strong("area_const"), ": Área construida de la propiedad en metros cuadrados."]),
                                    html.Li([html.Strong("admon"), ": Valor de la administración (cuota de mantenimiento) en pesos colombianos."]),
                                    html.Li([html.Strong("Estrato"), ": Estrato socioeconómico de la propiedad."]),
                                    html.Li([html.Strong("Estado"), ": Estado físico de la propiedad (Bueno, Excelente, Regular)."]),
                                    html.Li([html.Strong("Antiguedad"), ":Rango de a ntigüedad de la propiedad."]),
                                    html.Li([html.Strong("Tipo de Apartamento"), ": Tipo de apartamento (Loft, Duplex, Penthouse)."]),
                                    html.Li([html.Strong("Sector"), ": Sector específico donde se encuentra la propiedad"])
                                ])
  
                ])
            ], style={'width':'70%', 'text-align': 'justify' })
        ], style={'display':'flex','justify-content':'space-around', 'background-color':'#F2F1E9' }) 




    elif tab == 'tab-comprar':
        return html.Div([

            html.Div([  html.H2(['ESTAS VIENDO:']),
                        html.Div(id='filtros'),
                        html.Br(),
                        html.Div(id='cantidad'),
                        html.Br(),
                        html.Div(id='filtro2'),

                      ], style={
                                'background-color': '#F2C6AC',       # Color de fondo del contenedor
                                'display': 'flex',                   # Configuración de visualización flexbox
                                'padding': '10px',                   # Relleno del contenedor
                                'justify-content': 'space-around',   # Alineación de los elementos horizontales
                                'align-items': 'center'              # Alineación de los elementos verticales
                            }),
            
            html.Div([#Div que incluye las 3 secciones)
                html.Div([#Esta es la barra lateral
                    html.Div([
                        html.Button('Reset Filters', id='reset-btn', n_clicks=0),
                        html.Div(id='output-div'),

                        html.H4('Selecciona una ciudad'),
                        dcc.Dropdown(
                            id='ciudad-input',
                            options=[{'label': ciudad, 'value': ciudad} for ciudad in df['ciudad'].unique()],
                            value='Barranquilla',
                            multi=False
                        ),

                        html.H4('Selecciona una zona'),
                        dcc.Dropdown(
                            id='zona-input', 
                            multi=True),

                        html.H4('Barrio'),
                        dcc.Dropdown(
                            id='barrio-input',
                            multi=True),    

                        html.H4('Selecciona una categoría'),
                        dcc.Dropdown(
                            id='categoria-input',
                            value=['Casa'],
                            multi=True),

                        html.H4('Habitaciones'),
                        dcc.Dropdown(
                            id='habitaciones-input',
                            options=[{'label': habitaciones, 'value': habitaciones} for habitaciones in df['rooms'].dropna().unique()], 
                            value=[4],
                            multi=True),

                        html.H4('Baños'),
                        dcc.Dropdown(
                            id='baños-input',
                            options=[{'label': baños, 'value': baños} for baños in df['baths'].dropna().unique()], 
                            value=[3],
                            multi=True),    

                        html.H4('Parqueaderos'),
                        dcc.Dropdown(
                            id='park-input',
                            options=[{'label': park, 'value': park} for park in df['park'].dropna().unique()], 
                            value=[1],
                            multi=True), 

                        html.H4('Estrato'),
                        dcc.Dropdown(
                            id='estrato-input',
                            options=[{'label': estrato, 'value': estrato} for estrato in df['Estrato'].dropna().unique()], 
                            value=[3,4],
                            multi=True), 

                        html.H4('Antiguedad'),
                        dcc.Dropdown(
                            id='age-input',
                            options=[{'label': age, 'value': age} for age in df['Antiguedad'].dropna().unique()], 
                            value=['1 a 8 años'],
                            multi=True), 

                        html.H4('Estado'),
                        dcc.Dropdown(
                            id='estado-input',
                            options=[{'label': estado, 'value': estado} for estado in df['Estado'].dropna().unique()], 
                            value=['5'],
                            multi=True),                          
                        
                    ], style={'padding': '20px',})
                ], style={'width': '25%', 'overflow-y': 'scroll','background-color': '#C1D9D2'}), #Formato de la barra lateral
        
        #LADO CENTRAL
                html.Div([
                    html.H3('UBICACION DE VIVIENDAS EN VENTA'),
                    html.Div([
                            html.Div([
                                html.H4('Selecciona un rango de precios [COP]'),
                                    dcc.RangeSlider(
                                        id='price-slider',
                                        min=df[df['price']>=0]['price'].min(), 
                                        max=df['price'].quantile(0.95), 
                                        marks = None,
                                        value=[df['price'].quantile(0.01),df['price'].quantile(0.99)],
                                        tooltip={
                                                "always_visible": True,
                                                "template": "$ {value} "
                                            },
                            
                                    ),
                            ]),        
                        html.Div([
                            html.H4('Selecciona un rango de área [m2]'),
                                dcc.RangeSlider(
                                    id='area-slider',
                                    min=df['area_const'].quantile(0.01), 
                                    max=df['area_const'].quantile(0.999),
                                    marks=None, 
                                    value=[df['area_const'].quantile(0.01),df['area_const'].quantile(0.95)],
                                    tooltip={
                                            "always_visible": True,
                                            "template": " {value} m^2 "
                                        },
                                ),
                            ]),
                    ],style={'display':'flex'}),


                    html.Div([#AREA DEL GRAFICO)
                            #dcc.Graph(id='mapa1',style={'width': '100%', 'height': '90vh'}),
                            
                            dl.Map(
                                    id='mapa-principal',
                                    center=[initial_lat, initial_lon],
                                    zoom=10,
                                    children=[
                                        dl.TileLayer(),
                                        # Puedes agregar otros componentes como marcadores o polígonos aquí si es necesario
                                    ],
                                    style={'width': '100%', 'height': '80vh'}
                                )
                            
                            
                            ]),   
                           
                    ], style={'width': '40%', 'background-color': '#F2F1E9', 'padding':'30px','padding-top':'20px'}),

        # LADO IZQUIERDO

                html.Div([
                        html.Div([
                            html.H3('DESCRIPTIVOS'),

                            html.Div([
                                    dcc.Graph(id='side1'),
                                    html.Br(),
                                    dcc.Graph(id='side2'),
                                    html.Br(),
                                    dcc.Graph(id='side3')
                            ]),
                        ]),

                    ], style={'width': '35%', 'background-color': '#F2F1E9','padding':'20px'}),
                
            
            ], style={'display': 'flex'}),

        ])

     


     
    elif tab == 'tab-vender':
           return html.Div([

            html.Div([  html.H2(['CALCULA EL PRECIO DE TU APARTAMENTO:']),
                        html.Div(id='vfiltros'),
                        html.Br(),
                        html.Div(id='vfiltro2'),

                      ], style={
                                'background-color': '#F2C6AC',       # Color de fondo del contenedor
                                'display': 'flex',                   # Configuración de visualización flexbox
                                'padding': '10px',                   # Relleno del contenedor
                                'justify-content': 'space-around',   # Alineación de los elementos horizontales
                                'align-items': 'center'              # Alineación de los elementos verticales
                            }),
            
            html.Div([#Div que incluye las 3 secciones)
                html.Div([#Esta es la barra lateral
                    html.Div([
                        html.Button('Reset Filters', id='vreset-btn', n_clicks=0),
                        html.Div(id='output-div'),

                        html.H4('Selecciona una ciudad'),
                        dcc.Dropdown(
                            id='vciudad-input',
                            options=[{'label': ciudad, 'value': ciudad} for ciudad in df['ciudad'].unique()],
                            value='Barranquilla',
                            multi=False
                        ),

                        html.H4('Selecciona una zona'),
                        dcc.Dropdown(
                            id='vzona-input', 
                            multi=False),

                        html.H4('Barrio'),
                        dcc.Dropdown(
                            id='vbarrio-input',
                            multi=False),    

                        html.H4('Selecciona una categoría'),
                        dcc.Dropdown(
                            id='vcategoria-input',
                            options= [{'label': categoria, 'value': categoria} for categoria in df['category'].dropna().unique()],
                            value= 'Apartamento',
                            multi=False),

                        html.H4('Habitaciones'),
                        dcc.Dropdown(
                            id='vhabitaciones-input',
                            options=[{'label': habitaciones, 'value': habitaciones} for habitaciones in df['rooms'].dropna().unique()], 
                            value=1,
                            multi=False),

                        html.H4('Baños'),
                        dcc.Dropdown(
                            id='vbaños-input',
                            options=[{'label': baños, 'value': baños} for baños in df['baths'].dropna().unique()],
                            value=1, 
                            multi=False),    

                        html.H4('Parqueaderos'),
                        dcc.Dropdown(
                            id='vpark-input',
                            options=[{'label': park, 'value': park} for park in df['park'].dropna().unique()],
                            value=1, 
                            multi=False), 

                        html.H4('Estrato'),
                        dcc.Dropdown(
                            id='vestrato-input',
                            options=[{'label': estrato, 'value': estrato} for estrato in df['Estrato'].dropna().unique()],
                            value=1.0, 
                            multi=False), 

                        html.H4('Antiguedad'),
                        dcc.Dropdown(
                            id='vage-input',
                            options=[{'label': age, 'value': age} for age in df['Antiguedad'].dropna().unique()],
                            value='1 a 8 años', 
                            multi=False), 

                        html.H4('Estado'),
                        dcc.Dropdown(
                            id='vestado-input',
                            options=[{'label': estado, 'value': estado} for estado in df['Estado'].dropna().unique()],
                            value='4',
                            multi=False), 

                        html.H4('Area construida de tu inmueble [m2]'),
                            dcc.Input(id='varea', type='number', min=2, step=1, value=30.00),

                        html.H4('Valor de la Administración'),
                            dcc.Input(id='vadmon', type='number', min=0, step=50000,value=30000),

                        html.H4('Tipo de inmueble'),
                        dcc.Dropdown(
                            id='vtipo-input',
                            options=[{'label': tipo, 'value': tipo} for tipo in df['Tipo de Apartamento'].dropna().unique()],
                            value='Loft', 
                            multi=False),                         
                        
                    ], style={'padding': '20px',})
                ], style={'width': '25%', 'overflow-y': 'scroll','background-color': '#F2C6AC'}), #Formato de la barra lateral
        
        #LADO CENTRAL
                    html.Div([
                        html.H3('ESCOGE EL MODELO PARA HACER LA PREDICCIÓN'),
                        html.Div([
                            html.Div([
                                html.H4('Selecciona los modelos'),
                                    dcc.RadioItems(
                                    id='modelo',
                                    options=[
                                        {'label': 'Modelo Lasso', 'value': 'model_lasso'},
                                        {'label': 'Modelo Ridge', 'value': 'model_ridge'},
                                        {'label': 'Modelo XGBoost', 'value': 'model_xgboost'},
                                        {'label': 'Modelo MLP', 'value': 'model_mlp'}
                                    ],
                                    value='model_xgboost'
                                ),
                            ], style={'display': 'flex'}),

                            html.Br(),

                            html.Div([
                                dcc.Graph(id='features'),
                                html.Br(),
                                dcc.Graph(id='test_graph')
                            ])    
                        ]),
        
                    ],style={'width': '40%', 'background-color': '#F2F1E9', 'padding':'30px','padding-top':'20px'}),


        # LADO IZQUIERDO

                html.Div([
                        html.Div([
                            html.H1('PREDICCIONES'),
                            html.P( "El valor calculado para tu inmueble es: "),
                            html.P( id='precio_calculado'),
                            html.Br(),
                            html.P( "El valor estimado del metro cuadrado es: "),
                            html.P(id='metro-cuadrado')
                        ]),

                        html.Div([
                                    html.H1('Métricas de los Modelos'),
                                    dash_table.DataTable(
                                        id='tabla-metricas',
                                        columns=[{'name': col, 'id': col} for col in metrics_df.columns],
                                        data=metrics_df.to_dict('records'),
                                        style_table={'overflowX': 'auto'},  # Agregar scroll horizontal
                                        style_header={'backgroundColor': '#D99982', 'fontWeight': 'bold'},  # Estilo de encabezado
                                        style_cell={'minWidth': 100, 'textAlign': 'center'},  # Estilo de celdas
                                        # Estilo de celdas pares e impares alternadas
                                        style_data_conditional=[
                                            {'if': {'row_index': 'odd'}, 'backgroundColor': '#B9CDCA'},
                                            {'if': {'row_index': 'even'}, 'backgroundColor': '#F2C6AC'}
                                        ]
                                    )
                                ])





                    ], style={'width': '35%', 'background-color': '#F2F1E9','padding':'20px'}),
                
            
            ], style={'display': 'flex'}),

        ])
    

def get_center_coordinates(filtered_df):
    # Calcula la mediana de las latitudes y longitudes del DataFrame filtrado
    center_lat = filtered_df['lat'].median()
    center_lon = filtered_df['long'].median()
    
    return center_lat, center_lon

###########   PESTAÑA COMPRAR################
# Callback para actualizar las opciones del Dropdown de zonas según la ciudad seleccionada
@app.callback(
    [Output('zona-input', 'options'),
     Output('barrio-input', 'options'),
     Output('categoria-input', 'options'),
     Output('filtros', 'children'),
     Output('filtro2', 'children'),
     Output('data_store', 'data'),
     Output('cantidad', 'children')],
     
    
    [Input('ciudad-input', 'value'),
     Input('zona-input', 'value'),
     Input('categoria-input', 'value'),
     Input('habitaciones-input', 'value'),
     Input('baños-input', 'value'),
     Input('park-input', 'value'),
     Input('estrato-input', 'value'),
     Input('age-input', 'value'),
     Input('barrio-input', 'value'),
     Input('estado-input', 'value'),
     Input('price-slider', 'value'),
     Input('area-slider', 'value')  
     ],
)
def update_filters(selected_city, selected_zones, categoria, habitaciones, baños, park, estrato, age, barrio, estado, price, area):
    # Filtrar el DataFrame por ciudad seleccionada y devolver las opciones para zona y barrio
    if selected_city:
        filtered_df = df[df['ciudad'] == selected_city].dropna()
        sector_options = [{'label': sector, 'value': sector} for sector in filtered_df['Sector'].dropna().unique()]
        barrio_options = [{'label': barrio, 'value': barrio} for barrio in filtered_df['barrio'].dropna().unique()]
        categoria_options= [{'label': categoria, 'value': categoria} for categoria in filtered_df['category'].dropna().unique()]
        filtro_ciudad = f"Inmuebles en venta en {selected_city}"
        
    else:
        sector_options = []
        barrio_options=[]
        filtro_ciudad = "No has seleccionado una ciudad"
        filtered_df = df

    # Filtrar el DataFrame por zonas seleccionadas
    if selected_zones:
        filtered_df = filtered_df[filtered_df['Sector'].isin(selected_zones)]
        filtro_zonas = f'En los sectores de {selected_zones}'
    else:
        filtro_zonas = 'Seleccione un sector'

    #   Filtrar el DataFrame por Barrio
    if barrio:
        filtered_df = filtered_df[filtered_df['barrio'].isin(barrio)]
    
    if categoria:
        filtered_df = filtered_df[filtered_df['category'].isin(categoria)]

    if habitaciones:
        filtered_df = filtered_df[filtered_df['rooms'].isin(habitaciones)]

    if baños:
        filtered_df = filtered_df[filtered_df['baths'].isin(baños)]  

    if park:
        filtered_df = filtered_df[filtered_df['park'].isin(park)]

    if estrato:
        filtered_df = filtered_df[filtered_df['Estrato'].isin(estrato)]

    if age:
        filtered_df = filtered_df[filtered_df['Antiguedad'].isin(age)]
        
    if estado:
        filtered_df = filtered_df[filtered_df['Estado'].isin(estado)]

    if price:
        filtered_df= filtered_df[(filtered_df['price'] >= price[0]) & (filtered_df['price'] <= price[1])]

    print(area)
    if area:
        filtered_df= filtered_df[(filtered_df['area_const'] >= area[0]) & (filtered_df['area_const'] <= area[1])]                              
    
    print(price)

    data_store = filtered_df.to_json(date_format='iso', orient='split')
    nresults=f'Mostrando {len(filtered_df)} resultados'
    print(filtered_df.head())
    return sector_options, barrio_options, categoria_options,html.P(filtro_ciudad), html.P(filtro_zonas), data_store, html.P(nresults)



############   MAPA PRINCIPAL - COMPRAR ##################
@app.callback(
    Output('mapa-principal', 'center'),
    Output('mapa-principal', 'children'),
    [Input('data_store', 'data')]
)
def update_map(data_store):
    print(f"Data Store en update_map: {data_store}")  # <-- Imprimir data_store recibido
    
    # Verificar si data_store está vacío
    if data_store is None:
        error_message = "No hay datos disponibles para mostrar en el mapa."
        return (0, 0), [dl.Marker(position=[0, 0], children=[dl.Tooltip(error_message)])]

    filtered_df = pd.read_json(data_store, orient='split')
    print(filtered_df.head())  # <-- Imprimir algunas filas del DataFrame filtrado

    # Calcular el centro del mapa como el promedio de las latitudes y longitudes de los puntos
    center_lat = filtered_df['lat'].median()
    center_long = filtered_df['long'].median()

    markers = [dl.TileLayer()] + [
        dl.Marker(position=[row['lat'], row['long']], children=[
            dl.Tooltip(f"Barrio: {row['barrio']}, Precio: {row['price']}")
        ]) for _, row in filtered_df.iterrows()
    ]
    print(markers)  # <-- Imprimir los marcadores generados

    return (center_lat, center_long), markers

############   GRAFICOS DESCRIPTIVOS - COMPRAR ##################

@app.callback(
    [Output('side1', 'figure'),
     Output('side2', 'figure'),
     Output('side3', 'figure')],
     
    
    [Input('ciudad-input', 'value')],
)
def side_figures(selected_city):

    if selected_city:
        # Filtrar el DataFrame por la ciudad seleccionada
        filtered_df = df[df['ciudad'] == selected_city]
        
        estratos_por_categoria = filtered_df.groupby('Estrato')['category'].value_counts().unstack().dropna()
        
        # Crear el gráfico de barras
        fig1 = go.Figure()

        # Añadir barras para cada estrato dentro de cada categoría de vivienda
        for categoria in estratos_por_categoria.columns:
            fig1.add_trace(go.Bar(
                x=estratos_por_categoria.index,
                y=estratos_por_categoria[categoria],
                name=f'Categoria {categoria}'
            ))

        # Personalizar el diseño del gráfico
            fig1.update_layout(
                title=f'Distribución de Categoría de Vivienda por Estratos en {selected_city}',
                xaxis=dict(title='Categoría de Vivienda'),
                yaxis=dict(title='Cantidad'),
                barmode='stack',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=-0.45,  # Ajuste vertical para posicionar la leyenda en la parte inferior del gráfico
                    xanchor="right",
                    x=1.05,   # Ajuste horizontal para posicionar la leyenda fuera del gráfico
                    font=dict(
                        family="Arial",  
                        size=8,   
                        color="black" 
                    )))
            
        fig2 = go.Figure(data=[go.Histogram(x=filtered_df[filtered_df['price'] >= 0]['price'])])

        fig2.update_layout(
            title='Histograma de precios en {}'.format(selected_city),
            xaxis_title='Precio',
            yaxis_title='Frecuencia'
        )

        # Agrupar por sector y calcular el valor promedio del metro cuadrado
        valor_promedio_por_sector = filtered_df.groupby('Sector')['p_s'].mean().reset_index()

        # Crear la gráfica de barras
        fig3 = px.bar(valor_promedio_por_sector, x='Sector', y='p_s', 
                    title=f'Valor Promedio del Metro Cuadrado por Sector en {selected_city}',
                    labels={'sector': 'Sector', 'valor_metro_cuadrado': 'Valor Promedio del Metro Cuadrado'})
        
        fig3.update_layout(
            title='Precio por metro cuadrado en {}'.format(selected_city),
            xaxis_title='Precio por metro cuadrado',
            yaxis_title='Sector'
        )


    return fig1, fig2, fig3

###### PESTAÑA VENDER ##############

@app.callback(
    [Output('vzona-input', 'options'),
     Output('vbarrio-input', 'options'),
     Output('vfiltros', 'children'),
     Output('vfiltro2', 'children'),
     Output('data_store2', 'data')
     ],
     
    [
    Input('vciudad-input', 'value'),
    Input('vzona-input', 'value'),
    Input('vbarrio-input', 'value')   
    ],
)
def update_filters(city, zone, barrio):
    # Filtra el DataFrame según las selecciones del usuario
    filtered_df = df[df['ciudad'] == city] if city else df
    filtered_df = filtered_df[filtered_df['Sector'] == zone] if zone else filtered_df
    filtered_df = filtered_df[filtered_df['barrio'] == barrio] if barrio else filtered_df

    # Mensajes de filtro
    filtro_ciudad = f"Estás vendiendo tu inmueble en: {city}" if city else "No has seleccionado una ciudad"
    filtro_zonas = f'En el sector de {zone}' if zone else 'Seleccione un sector'

    # Convertir el DataFrame filtrado a JSON
    data_store2 = filtered_df.to_json(date_format='iso', orient='split')

    # Obtener opciones de sector y barrio
    sector_options = [{'label': zone, 'value': zone} for zone in filtered_df['Sector'].dropna().unique()]
    barrio_options = [{'label': barrio, 'value': barrio} for barrio in filtered_df['barrio'].dropna().unique()]

    return sector_options, barrio_options, html.P(filtro_ciudad), html.P(filtro_zonas), data_store2


@app.callback(
    [Output('precio_calculado', 'children'),
    Output('features', 'figure'),
    Output('test_graph', 'figure'),
    Output('metro-cuadrado', 'children'),
     
     ],

    [
    Input('data_store2', 'data'),  
    Input('vcategoria-input', 'value'),
    Input('vhabitaciones-input', 'value'),
    Input('vbaños-input', 'value'),
    Input('vpark-input', 'value'),
    Input('varea', 'value'),
    Input('vadmon', 'value'),
    Input('vestrato-input', 'value'),
    Input('vestado-input', 'value'),
    Input('vage-input', 'value'),
    Input('vtipo-input', 'value'),
    Input('modelo', 'value')   
    ]
)
def run_models(data_store2, categoria, habitaciones, baños, park, area, admon, estrato,estado,  age, tipo, modelo):

    data_store2=pd.read_json(data_store2, orient='split')

    lat_promedio = data_store2['lat'].mean()
    lon_promedio = data_store2['long'].mean()

    data_in = pd.DataFrame({
        'lat': [lat_promedio],
        'long': [lon_promedio],
        'category': [categoria],
        'rooms': [habitaciones],
        'baths': [baños],
        'park': [park],
        'area_const': [area],
        'admon': [admon],
        'Estrato': [estrato],
        'Estado': [estado],
        'Antiguedad': [age],
        'Tipo de Apartamento': [tipo]
    })


    if modelo == 'model_lasso':
        model = model_lasso
        fi=fi_lasso
        test=lasso

    elif modelo == 'model_ridge':
        model = model_ridge
        fi=fi_ridge
        test= ridge


    elif modelo == 'model_xgboost':
        model = model_xgboost
        fi=fi_xgboost
        test= xgboost
    elif modelo == 'model_mlp':
        model = model_mlp
        fi=fi_mlp
        test=mlp
    else:
        raise ValueError("Verifique los datos Ingresados")

    # Crear el gráfico de barras horizontales
    features = px.bar(fi, x='coefficient', y='feature', orientation='h',color_discrete_sequence=['#D99982'], 
        title='Coeficientes de características')
    
    features.update_layout(
    height=600,  # Altura del gráfico
    margin=dict(l=50, r=20, t=50, b=50),  # Márgenes
    yaxis=dict(
        tickfont=dict(size=8),  # Tamaño de la letra de los labels
        tickmode='array',        # Modo de los labels (array)
        tickvals=list(range(len(fi))),  # Valores de los labels
        ticktext=[text[:20] + '\n' + text[20:] if len(text) > 20 else text for text in fi['feature']]  # Texto de los labels con división
            )
        )

    
    test_graph=px.scatter(test, x='Y Real', y='Y Predicho', trendline='ols', color_discrete_sequence=['#D99982'],
                 title='Resultados del Testing: Y Real vs. Y')

    test_graph.update_traces(line=dict(color='#1F3040'))

    print(data_in)

    prediction = model.predict(data_in)[0]

    print(f"La predicción es: {prediction}")
    
    metro_cuadrado=prediction/area
    formatted_metro = "${:,.2f}".format(metro_cuadrado)
    metro_cuadrado_element = html.H3(f"{formatted_metro}")
    formatted_prediction = "${:,.2f}".format(prediction)

    prediction_element = html.H2(f"{formatted_prediction}")

    return [prediction_element, features, test_graph, metro_cuadrado_element]



# Ejecuta la aplicación
if __name__ == '__main__':
 app.run_server(debug=True, host='0.0.0.0', port=9000)
