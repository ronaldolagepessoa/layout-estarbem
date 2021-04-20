import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyomo.environ as pyo 
from pyomo.opt import SolverFactory
import itertools
from PIL import Image
from pathlib import Path
import base64
import pyutilib.subprocess.GlobalData
pyutilib.subprocess.GlobalData.DEFINE_SIGNAL_HANDLERS_DEFAULT = False


st.set_page_config(page_title='Layout', layout='wide')

def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded


def solve_secoes(df_pedidos, df_recebimentos, n_secoes, n_recebimento, volume_secao):
    model = pyo.ConcreteModel()
    model.Rotas = pyo.Set(initialize=df_pedidos['ROTA'].unique())
    model.Notas = pyo.Set(initialize=df_pedidos['NOTA'].unique())
    model.Produtos = pyo.Set(initialize=df_pedidos['PRODUTOID'].unique())
    model.Secoes = pyo.Set(initialize=[i + 1 + n_recebimento for i in range(n_secoes - n_recebimento)])
    
    lista = list(df_pedidos.groupby(['PRODUTOID', 'NOTA', 'ROTA']).NOME_PROD.nunique().index)
    final = [(value[0], value[1], value[2], s) for value in lista for s in model.Secoes]
    model.ProdutosNotasRotasSecoes = pyo.Set(initialize=final)
    
    model.demanda = pyo.Param(model.Produtos, model.Notas, 
                                        initialize=df_pedidos.set_index(['PRODUTOID', 'NOTA'])['Soma de QUANT'].to_dict())
    
    df_pedidos['VOLUME'] = df_pedidos['VOLUME_TOTAL'] / df_pedidos['Soma de QUANT']
    volume = df_pedidos.drop_duplicates(subset=['PRODUTOID']).set_index('PRODUTOID')['VOLUME'].to_dict()
    model.volume = pyo.Param(model.Produtos, initialize=volume)
    
    model.volumeSecao = volume_secao
    
    model.y = pyo.Var(model.Rotas, model.Secoes, within=pyo.Binary)
    model.x = pyo.Var(model.ProdutosNotasRotasSecoes, within=pyo.NonNegativeIntegers)
    
    model.obj = pyo.Objective(
        expr=sum(model.y[r, s] for r in model.Rotas for s in model.Secoes), 
        sense=pyo.minimize)
    
    model.c1 = pyo.ConstraintList()
    for s in model.Secoes:
        model.c1.add(
            sum(model.y[r, s] for r in model.Rotas) <= 1
            )
    model.c2 = pyo.ConstraintList()
    for r in model.Rotas:
        for s in model.Secoes:
            model.c2.add(
                sum(model.volume[p] * model.x[p, n, r, s] 
                    for p in model.Produtos for n in model.Notas 
                    if (p, n, r, s) in model.ProdutosNotasRotasSecoes) <= 
                model.volumeSecao * model.y[r, s]
            )
    model.c3 = pyo.ConstraintList()
    conjunto = list(df_pedidos.set_index(['PRODUTOID', 'NOTA']).index)
    for p in model.Produtos:
        for n in model.Notas:
            if (p, n) in conjunto:
                model.c3.add(
                    sum(model.x[p, n, r, s] for r in model.Rotas for s in model.Secoes 
                        if (p, n, r, s) in model.ProdutosNotasRotasSecoes) == model.demanda[p, n]
                )
    
    solver = SolverFactory('cbc')
    solver.solve(model, tee=False, timelimit=5)
    array = np.array([[p, n, r, s, model.x[p, n, r, s].value] 
                      for (p, n, r, s) in model.ProdutosNotasRotasSecoes
                      if model.x[p, n, r, s].value > 0])
    results = pd.DataFrame(array, columns=['IDPRODUTO', 'NOTA', 'ROTA', 'SECAO', 'QUANTIDADE'])
    results.sort_values(['ROTA', 'SECAO'], inplace=True)
    
    secoes = results['SECAO'].unique()
    replace_dict = {s: i + n_recebimento + 1 for i, s in enumerate(secoes)}
    results['SECAO'] = results['SECAO'].replace(replace_dict)
    return results

def solve_veiculos(df_pedidos, df_veiculos):
    df_rotas = pd.read_csv('dados/ROTAS.csv')
    df_rotas = df_rotas.loc[df_rotas.ROTA.isin(df_pedidos.ROTA.unique())]
    
    model = pyo.ConcreteModel()
    
    model.Rotas = pyo.Set(initialize=df_pedidos.ROTA.unique())
    model.Notas = pyo.Set(initialize=df_pedidos.NOTA.unique())
    
    model.NotasRotas = pyo.Set(
        initialize=df_pedidos.groupby(['NOTA', 'ROTA']).NOME_PROD.nunique().index
        )
    map_NotasRotas = df_pedidos.groupby('ROTA').NOTA.unique().to_dict()
    
    model.Veiculos = pyo.Set(initialize=df_veiculos.VEICULO.unique())
    
    temp = [(r, v) for v in model.Veiculos for r in model.Rotas 
     if df_veiculos.loc[df_veiculos.VEICULO == v, 'CIDADE'].values[0] == 'sim' 
     or (df_veiculos.loc[df_veiculos.VEICULO == v, 'CIDADE'].values[0] == 'não' and 
         df_rotas.loc[df_rotas.ROTA == r, 'CIDADE'].values[0] == 'não')]
    model.RotasVeiculos = pyo.Set(initialize=temp)
    map_RotasVeiculos = pd.DataFrame(temp, columns=['ROTAS', 'VEICULOS']).groupby('VEICULOS').ROTAS.unique().to_dict()
    
    temp = [(n, v) for v in df_veiculos.VEICULO for r in df_pedidos.ROTA.unique() for n in map_NotasRotas[r] 
     if df_veiculos.loc[df_veiculos.VEICULO == v, 'CIDADE'].values[0] == 'sim' 
     or (df_veiculos.loc[df_veiculos.VEICULO == v, 'CIDADE'].values[0] == 'não' and 
         df_rotas.loc[df_rotas.ROTA == r, 'CIDADE'].values[0] == 'não')]
    model.NotasVeiculos = pyo.Set(initialize=temp)
    map_NotasVeiculos = pd.DataFrame(temp, columns=['NOTAS', 'VEICULOS']).groupby('VEICULOS').NOTAS.unique().to_dict()
    map_VeiculosNotas = pd.DataFrame(temp, columns=['NOTAS', 'VEICULOS']).groupby('NOTAS').VEICULOS.unique().to_dict()
    
    temp = [(n, r, v) for v in df_veiculos.VEICULO for r in df_pedidos.ROTA.unique() for n in map_NotasRotas[r] 
    if df_veiculos.loc[df_veiculos.VEICULO == v, 'CIDADE'].values[0] == 'sim' 
    or (df_veiculos.loc[df_veiculos.VEICULO == v, 'CIDADE'].values[0] == 'não' and 
        df_rotas.loc[df_rotas.ROTA == r, 'CIDADE'].values[0] == 'não')]
    model.NotasRotasVeiculos = pyo.Set(initialize=temp)
    
    model.volumeVeiculo = pyo.Param(model.Veiculos, initialize=df_veiculos.set_index('VEICULO').VOLUME.to_dict())
    model.pesoVeiculo = pyo.Param(model.Veiculos, initialize=df_veiculos.set_index('VEICULO').PESO.to_dict())
    model.volumeNota = pyo.Param(model.Notas, initialize=df_pedidos.groupby('NOTA').VOLUME_TOTAL.sum().to_dict())
    model.pesoNota = pyo.Param(model.Notas, initialize=df_pedidos.groupby('NOTA').PESO_TOTAL.sum().to_dict())
    
    model.x = pyo.Var(model.NotasVeiculos, within=pyo.Binary)
    model.z = pyo.Var(model.RotasVeiculos, within=pyo.Binary)
    model.y = pyo.Var(model.Veiculos, within=pyo.Binary)
    
    model.obj = pyo.Objective(
        expr=sum(model.volumeVeiculo[v] * model.y[v] for v in model.Veiculos), 
        sense=pyo.minimize
        )
    
    # Cada veículo só pode estar associado a no máximo uma rota
    model.c1 = pyo.ConstraintList()
    for v in model.Veiculos:
        model.c1.add(
            sum(model.z[r, v] for r in map_RotasVeiculos[v]) <= 1
        )
    # Só é possível associar um rota 'r'  a um veículo 'v' se o útimo for utilizado
    model.c2 = pyo.ConstraintList()
    for (r, v) in model.RotasVeiculos:
        model.c2.add(
            model.z[r, v] <= model.y[v]
        )
    # Cada nota deve estar associada a exatamento um veículo
    model.c3 = pyo.ConstraintList()
    for n in model.Notas:
        model.c3.add(
            sum(model.x[n, v] for v in map_VeiculosNotas[n]) == 1
        )
    # Nota 'n' só pode ser associada ao veículo 'v' caso sua rota 'r' também o esteja
    model.c4 = pyo.ConstraintList()
    for (n, r, v) in model.NotasRotasVeiculos:
        model.c4.add(
            model.x[n, v] <= model.z[r, v]
        )
    # Volume e peso dos veículos devem ser respeitados
    model.c5 = pyo.ConstraintList()
    for v in model.Veiculos:
        model.c5.add(
            sum(model.volumeNota[n] * model.x[n, v] for v in map_VeiculosNotas[n]) <= model.volumeVeiculo[v]
        )
        model.c5.add(
            sum(model.pesoNota[n] * model.x[n, v] for v in map_VeiculosNotas[n]) <= model.pesoVeiculo[v]
        )
        
    solver = SolverFactory('cbc')
    solver.solve(model, tee=False, timelimit=5)
    
    array = np.array([[n, r, v] 
                      for (n, r, v) in model.NotasRotasVeiculos
                      if model.z[r, v].value > 0 and model.x[n, v].value > 0])
    results = pd.DataFrame(array, columns=['NOTA', 'ROTA', 'VEICULO'])
    return results
    

# @st.cache
def get_pedidos(df_pedidos, multiplicador_volume):
    df_dim = pd.read_csv('dados/PRODUTOS.csv')
    df_pedidos1 = pd.merge(df_pedidos, df_dim[['PRODUTOID', 'ALTURA', 'LARGURA', 'COMPRIMENTO', 'PESO']], 
                           left_on='PRODUTOID', right_on='PRODUTOID',how='left')
    df_pedidos1.dropna(axis=0, inplace=True)
    df_pedidos1['NOTA'] = df_pedidos1['NOTA'].map(lambda value: f'N{value}')
    df_pedidos1['VOLUME_TOTAL'] = df_pedidos1['ALTURA'] * df_pedidos1['LARGURA'] * df_pedidos1['COMPRIMENTO'] * df_pedidos1['Soma de QUANT'] * multiplicador_volume
    df_pedidos1['PESO_TOTAL'] = df_pedidos1['PESO'] * df_pedidos1['Soma de QUANT']
    return df_pedidos1

# @st.cache
def get_recebimentos(df_recebimentos, multiplicador_volume):
    df_dim = pd.read_csv('dados/PRODUTOS.csv')
    df_recebimentos1 = pd.merge(df_recebimentos, df_dim[['PRODUTOID', 'ALTURA', 'LARGURA', 'COMPRIMENTO', 'PESO']], 
                                left_on='PRODUTOID', right_on='PRODUTOID', how='left')
    df_recebimentos1.dropna(axis=0, inplace=True)
    df_recebimentos1['NUMNOTA'] = df_recebimentos1['NUMNOTA'].map(lambda value: f'N{value}')
    df_recebimentos1['VOLUME_TOTAL'] = df_recebimentos1['ALTURA'] * df_recebimentos1['LARGURA'] * df_recebimentos1['COMPRIMENTO'] * df_recebimentos1['Total'] * multiplicador_volume
    df_recebimentos1['PESO_TOTAL'] = df_recebimentos1['PESO'] * df_recebimentos1['Total']
    return df_recebimentos1

def show_layout(df_layout, n_secoes_recebimentos, n_secoes_separacao):
    myImg = np.zeros((512, 512 * 2, 3), np.uint8)
    fig, ax = plt.subplots()
    ax.imshow(myImg)
    MARGEM_X = 20
    MARGEM_Y = 20
    for index, row in df_layout.sort_values('seção').iterrows():
        if row['seção'] <= n_secoes_recebimentos:
            rect = patches.Rectangle((MARGEM_X + row['coluna']*50, MARGEM_Y + row['linha']*40), 40, 30, linewidth=1, edgecolor='r', facecolor='red')
        elif row['seção'] <= n_secoes_separacao:
            rect = patches.Rectangle((MARGEM_X + row['coluna']*50, MARGEM_Y + row['linha']*40), 40, 30, linewidth=1, edgecolor='b', facecolor='blue')
        else:
            rect = patches.Rectangle((MARGEM_X + row['coluna']*50, MARGEM_Y + row['linha']*40), 40, 30, linewidth=1, edgecolor='g', facecolor='green')
        ax.add_patch(rect)
        ax.text(MARGEM_X + row['coluna']*50 + 10, MARGEM_Y + row['linha']*40 + 25, f"{row['seção']}")

    rect = patches.Rectangle((512 * 2 - 200, 0), 200, 120, linewidth=1, edgecolor='black', 
                             facecolor='white')
    ax.add_patch(rect)
    rect = patches.Rectangle((512 * 2 - 200 + 5, 5), 40, 30, linewidth=1, edgecolor='black', 
                             facecolor='red')
    ax.add_patch(rect)
    ax.text(512 * 2 - 200 + 50, 30, 'Recebimentos', fontsize=8)
    rect = patches.Rectangle((512 * 2 - 200 + 5, 45), 40, 30, linewidth=1, edgecolor='black', facecolor='blue')
    ax.add_patch(rect)
    ax.text(512 * 2 - 200 + 50, 30 + 40, 'Expedição', fontsize=8)
    rect = patches.Rectangle((512 * 2 - 200 + 5, 85), 40, 30, linewidth=1, edgecolor='black', facecolor='green')
    ax.add_patch(rect)
    ax.text(512 * 2 - 200 + 50, 30 + 80, 'Vazio', fontsize=8)
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("layout.png", bbox_inches = 'tight',
        pad_inches = 0)

    # plt.savefig('layout.png')
    header_html = "<img src='data:image/png;base64,{}' class='img-fluid' width='100%'>".format(
            img_to_bytes("layout.png")
            )
    st.markdown(
        header_html, unsafe_allow_html=True,
    )

def submain1():

    st.subheader('Cálculo de seções e rotas')
    
    cols = st.beta_columns(2)
    
    with cols[0]:
        volume_maximo = st.number_input('Volume máximo por seção em m³', value=1.0*1.2*1.5)
    with cols[1]:
        multiplicador_volume = st.number_input('Fator de correção do volume', value=1.1)
    df_layout = pd.read_csv('dados/LAYOUT.csv')
    n_secoes = df_layout['seção'].max()

    cols = st.beta_columns(3)
    
    with cols[0]:
        pedidos = st.file_uploader('Pedidos do dia', type=['xlsx', 'xls', 'csv'])
    with cols[1]:
        recebimentos = st.file_uploader('Recebimentos do dia', type=['xlsx', 'xls', 'csv'])
    with cols[2]:
        veiculos = st.file_uploader('Veículos', type=['xlsx', 'xls', 'csv'])
    
    if st.button('Calcular'):
        try:
            df_pedidos = pd.read_csv(pedidos)
        except Exception:
            df_pedidos = pd.read_excel(pedidos)
        try:
            df_recebimentos = pd.read_csv(recebimentos)
        except Exception:
            df_recebimentos = pd.read_excel(recebimentos)
        try:
            df_veiculos = pd.read_csv(veiculos)
        except Exception:
            df_veiculos = pd.read_excel(veiculos)
        
        df_pedidos1 = get_pedidos(df_pedidos, multiplicador_volume)
        df_recebimentos1 = get_recebimentos(df_recebimentos, multiplicador_volume)
        
        with st.beta_expander('Resultados gerais'):
        
            cols = st.beta_columns(3)
            
            cubagem_pedidos = round(df_pedidos1.VOLUME_TOTAL.sum(), 2)
            cubagem_recebimentos = round(df_recebimentos1.VOLUME_TOTAL.sum(), 2)
            
            peso_pedidos = round(df_pedidos1.PESO_TOTAL.sum(), 2)
            peso_recebimentos = round(df_recebimentos1.PESO_TOTAL.sum(), 2)
            
            volumes_pedidos = df_pedidos1['Soma de QUANT'].sum()
            volumes_recebimentos = df_recebimentos1['Total'].sum()
            
            with cols[0]:
                st.write('__Cubagem total__: ', "{:,} m³".format(round(cubagem_pedidos + cubagem_recebimentos, 2)))
                st.write('- __Cubagem total dos pedidos__: ', "{:,} m³".format(cubagem_pedidos))
                st.write('- __Cubagem total dos recebimentos__: ', "{:,} m³".format(cubagem_recebimentos))
                
            with cols[1]:
                st.write('__Peso total__: ', "{:,} kg".format(round(peso_pedidos + peso_recebimentos, 2)))
                st.write('- __Peso total dos pedidos__: ', "{:,} kg".format(peso_pedidos))
                st.write('- __Peso total dos recebimentos__: ', "{:,} kg".format(peso_recebimentos))
                
            with cols[2]:
                st.write('__Volumes totais__: ', "{}".format(volumes_pedidos + volumes_recebimentos))
                st.write('- __Volumes totais dos pedidos__: ', "{}".format(volumes_pedidos))
                st.write('- __Volumes totais dos recebimentos__: ', "{}".format(volumes_recebimentos))
        
        n_secoes_recebimentos = np.ceil(cubagem_recebimentos / volume_maximo)
        
        with st.spinner('Calculando...'):
            n_recebimento = int(np.ceil(n_secoes_recebimentos))
            results_secoes = solve_secoes(df_pedidos1, df_recebimentos1, n_secoes, n_recebimento, volume_maximo)
            results_veiculos = solve_veiculos(df_pedidos1, df_veiculos)
            results_secoes['QUANTIDADE'] = pd.to_numeric(results_secoes['QUANTIDADE'])
            # st.components.v1.html('<hr>', height=10)
            with st.beta_expander('Resultado detalhado'):
                st.subheader('Alocação para recebimentos')
                st.write(f'- Reservar seções de 1 a {n_recebimento} para recebimento')
                cols = st.beta_columns(2)
                with cols[0]:
                    st.subheader('Alocação para os pedidos')
                    # print(results_secoes.columns)
                    for rota in results_secoes['ROTA'].unique():
                        st.markdown(f'__ROTA {rota}:__')
                        for secao in results_secoes.loc[results_secoes['ROTA'] == rota, 'SECAO'].unique():
                            st.markdown(f"* __Seção {secao}__")
                            temp = results_secoes.loc[(results_secoes.ROTA == rota) & (results_secoes.SECAO == secao)].groupby('NOTA')['QUANTIDADE'].sum().reset_index()
                            for index, row in temp.iterrows():
                                st.markdown(f"* Pedido {row.NOTA} | nº de volumes = {row['QUANTIDADE']}")
                with cols[1]:
                    st.subheader('Alocação nos caminhões')
                    print(results_veiculos.columns)
                    for rota in results_veiculos['ROTA'].unique():
                        st.write(f'__ROTA {rota}:__')
                        for veiculo in results_veiculos.loc[results_veiculos.ROTA == rota, 'VEICULO'].unique():
                            st.write(f"* __Veículo {veiculo}__")
                            # for index, row in results_veiculos.loc[results_veiculos['ROTA'] == rota].iterrows():
                            #     st.write(f"* Veículo {row.VEICULO} ")
            with st.beta_expander('Visualização das seções'):
                df_layout = pd.read_csv('dados/LAYOUT.csv')
                show_layout(df_layout, n_secoes_recebimentos, results_secoes['SECAO'].max())
                    
def submain2():
    st.subheader('Upload de arquivos')
    cols = st.beta_columns(3)
    with cols[0]:
        produtos = st.file_uploader('Informações dos produtos', type=['xlsx', 'xls', 'csv'])
        if produtos:
            if st.button('Salvar', key='produtos'):
                try:
                    df_produtos = pd.read_csv(produtos)
                except Exception:
                    df_produtos = pd.read_excel(produtos)
                df_produtos.to_csv('dados/PRODUTOS.csv', index=False)
                st.success('Uploald feito com sucesso!')
    with cols[1]:
        rotas = st.file_uploader('Informações das rotas', type=['xlsx', 'xls', 'csv'])
        if rotas:
            if st.button('Salvar', key='rotas'):
                try:
                    df_rotas = pd.read_csv(rotas)
                except Exception:
                    df_rotas = pd.read_excel(rotas)
                df_rotas.to_csv('dados/ROTAS.csv', index=False)
                st.success('Uploald feito com sucesso!')
    with cols[2]:
        layout = st.file_uploader('Informações do layout', type=['xlsx', 'xls', 'csv'])
        if layout:
            if st.button('Salvar', key='layout'):
                try:
                    df_layout = pd.read_csv(layout)
                except Exception:
                    df_layout = pd.read_excel(layout)
                df_layout.to_csv('dados/LAYOUT.csv', index=False)
                st.success('Uploald feito com sucesso!')
        
if __name__ == '__main__':
    st.header('Ferramenta de Planejamento de Layout')
    st.sidebar.subheader('Menu')
    option = st.sidebar.radio('', ('Cálculo de seções e rotas', 'Upload de arquivos'))
    if option == 'Cálculo de seções e rotas':
        submain1()
    elif option == 'Upload de arquivos':
        submain2()