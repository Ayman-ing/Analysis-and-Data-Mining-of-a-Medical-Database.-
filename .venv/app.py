import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import datetime as date
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram , linkage,fcluster

def progressBar():
    progress_bar = st.progress(0)
    # Simulate progress
    for i in range(11):
        progress_bar.progress(i * i)
        time.sleep(0.1)
st.title('Analyse et Fouille de données d’une base de données médicales')
st.title("Dataset Importer")
st.session_state.is_normalized = False

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
#s' il y a un fichier importer on le traite
def dataSetInformations():
    st.subheader('Raw data')

    # lire le fichier et l'afficher
    dataSet = pd.read_csv(uploaded_file)
    st.write(dataSet)
    if "dataSet" not in st.session_state:
        st.session_state.dataSet = dataSet

    st.write("### informations")
    numberOfRows, numberOfColumns = st.session_state.dataSet.shape
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Nombre d'observations :")
        st.markdown(f"<p style='color: green;font-weight: bold; font-size: 24px;'>{numberOfRows}</p>",
                    unsafe_allow_html=True)

    with col2:
        st.write("### Nombre de caractéristiques :")
        st.markdown(f"<p style='color: green; font-weight: bold; font-size: 24px;'>{numberOfColumns}</p>",
                    unsafe_allow_html=True)
        # Vérifier s'il y a des valeurs manquantes

def replaceNoneValues():
    progressBar()
    for column in st.session_state.dataSet.columns:

        if (st.session_state.dataSet[column].dtype == 'int64' or st.session_state.dataSet[column].dtype == 'float64'):
            st.session_state.dataSet[column] = st.session_state.dataSet[column].fillna(st.session_state.dataSet[column].mean())
    time.sleep(0.1)

    st.write("remplacement est terminé")
    st.write(st.session_state.dataSet)

def codingTheValues():
    progressBar()

    this_year = date.datetime.now().year
    for column in st.session_state.dataSet.columns:
        if st.session_state.dataSet[column].astype(str).str.match(r'^\d{2}-\d{2}-\d{4}$').any():
            st.session_state.dataSet[column] = this_year - st.session_state.dataSet[column].str[-4:].astype(int)
            continue

        if not (st.session_state.dataSet[column].dtype == 'int64' or st.session_state.dataSet[
            column].dtype == 'float64'):
            my_dict = {}
            index = 0
            for element in st.session_state.dataSet[column]:
                if element not in my_dict:
                    my_dict[element] = index
                    index += 1

            for key, value in my_dict.items():
                st.session_state.dataSet[column] = st.session_state.dataSet[column].replace(key, value)
    st.write("codage est terminé")
    st.write(st.session_state.dataSet)


def normalizingTheDataSet():
    progressBar()
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(st.session_state.dataSet)

    # Create a DataFrame from the normalized data
    st.session_state.normalized_df = pd.DataFrame(normalized_data, columns=st.session_state.dataSet.columns)
    means = st.session_state.normalized_df.mean().round(2)
    stds = st.session_state.normalized_df.std().round(2)
    st.session_state.is_normalized = True
    st.session_state.already_normalized = True
    st.write("### la base a été normalisée")


def manipulatingTheDataSet():
    st.subheader('Normalized Dataset')
    st.write(st.session_state.normalized_df)
    width = st.slider('Width', min_value=8, max_value=20, value=10)
    height = st.slider('Height', min_value=8, max_value=20, value=8)
    fig, ax = plt.subplots(figsize=(width, height))
    st.session_state.corrélation_df = st.session_state.normalized_df.corr()  # les données doit etre centré reduite pour faire la corrélation
    sb.heatmap(st.session_state.corrélation_df, annot=True, ax=ax)
    st.pyplot(fig)
    st.write("#### le couple de variables les plus corrélées est séléctionné par défaut")
    xvar, yvar = st.session_state.corrélation_df[st.session_state.corrélation_df < 1].stack().idxmax()
    xvar = st.selectbox('Select x-axis:', st.session_state.corrélation_df.columns[:],
                        index=st.session_state.corrélation_df.columns.get_loc(
                            xvar))  # get_loc est utulisé pour avoir l'indice
    yvar = st.selectbox('Select y-axis:', st.session_state.corrélation_df.columns[:],
                        index=st.session_state.corrélation_df.columns.get_loc(yvar))

    st.write(px.scatter(st.session_state.corrélation_df, x=xvar, y=yvar))
    st.session_state.showing_correlation = True


def applyingTheACP():

    st.session_state.acp_normé = PCA()
    Y = st.session_state.acp_normé.fit_transform(st.session_state.normalized_df)

    st.session_state.Y_df = pd.DataFrame(Y)

    st.write(st.session_state.Y_df)
    st.session_state.valeurs_propres = st.session_state.acp_normé.explained_variance_
    st.session_state.pourcentages = (st.session_state.valeurs_propres / np.sum(st.session_state.valeurs_propres)) * 100
    st.session_state.pourcentage_cumulé = np.cumsum(st.session_state.pourcentages)
    st.title("Les valeurs propres")
    st.session_state.données = []
    for i in range(len(st.session_state.valeurs_propres)):
        st.session_state.données.append({
            'Composante Principale': 'CP{}'.format(i + 1),
            'Valeur Propre': st.session_state.valeurs_propres[i],
            'Pourcentage': st.session_state.pourcentages[i],
            'Pourcentage Cumulé': st.session_state.pourcentage_cumulé[i]
        })
    st.write("## Composantes choisis")
    st.table(st.session_state.données)
    st.title("Graphique des valeurs propres")


def showingTheGraphics():
    fig, ax = plt.subplots()
    ax.bar(np.arange(0, 15), st.session_state.valeurs_propres, color="teal")
    ax.plot(np.arange(0, 15), st.session_state.valeurs_propres, color="red")
    ax.set_xlabel("Composantes principales")
    ax.set_ylabel("Valeurs propres")
    st.pyplot(fig)

    # Créer une application Streamlit
    st.title("Pourcentage d'inertie cumulé")
    # Afficher le graphique
    fig, ax = plt.subplots()

    ax.plot(np.arange(1, 16), np.cumsum(st.session_state.acp_normé.explained_variance_ratio_))
    ax.set_xlabel("Nombre de composantes principales")
    ax.set_ylabel("Pourcentage d'inertie cumulé")
    st.pyplot(fig)


def CPWithKaizerMethod():
    st.session_state.valeurs_propres_choisis = st.session_state.valeurs_propres[st.session_state.valeurs_propres > 1]
    output = []
    for i in range(len(st.session_state.valeurs_propres_choisis)):
        output.append({
            'Composante Principale': 'CP{}'.format(i + 1),
            'Valeur Propre': st.session_state.valeurs_propres[i]})
    st.table(output)

def CPWithPourcentageMethod():
    pourcentage = st.slider('Pourcentage', min_value=20, max_value=100, value=60)
    if pourcentage == 100:
        pourcentage += 1
    st.session_state.valeurs_propres_choisis = []
    for i in range(len(st.session_state.données)):
        if (st.session_state.données[i]["Pourcentage Cumulé"] < pourcentage):

            st.session_state.valeurs_propres_choisis.append(st.session_state.données[i]["Composante Principale"])
        else:
            break
    output = []
    for i in range(len(st.session_state.valeurs_propres_choisis)):
        output.append({
            'Composante Principale': 'CP{}'.format(i + 1),
            'Valeur Propre': st.session_state.valeurs_propres[i],
            'Pourcentage Cumulé': st.session_state.pourcentage_cumulé[i]})
    st.write("## Composantes choisis")
    st.table(output)


def showingTheGraphicsWithCP():
    indices = list(range(1, len(st.session_state.valeurs_propres_choisis) + 1))
    xvar = st.selectbox('Select x-axis:', indices)
    yvar = st.selectbox('Select y-axis:', indices)
    st.write(px.scatter(st.session_state.Y_df, x=xvar - 1, y=yvar-1))

def variablesLoading():
    st.session_state.variable_saturation = st.session_state.acp_normé.components_.T * np.sqrt(
        st.session_state.acp_normé.explained_variance_)
    st.session_state.saturation_varaibles_already_calculated = True
    st.write("Variable Loadings:")
    st.write(pd.DataFrame(st.session_state.variable_saturation,
                          columns=[f"PC{i + 1}" for i in range(len(st.session_state.normalized_df.columns))],
                          index=st.session_state.normalized_df.columns))

    st.session_state.variable_symbols = [f"I{i + 1}" for i in range(len(st.session_state.normalized_df.columns))]

    # Creating a key for the symbols
    symbol_key = {symbol: var for symbol, var in
                  zip(st.session_state.variable_symbols, st.session_state.normalized_df.columns)}
    # Plotting the correlation circle
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
    indices = list(range(1, len(st.session_state.valeurs_propres_choisis) + 1))
    xvar = st.selectbox('Select x-axis: CP', indices)
    yvar = st.selectbox('Select y-axis: CP', indices)

    for i, symbol in enumerate(st.session_state.variable_symbols):
        ax.plot([0, st.session_state.variable_saturation[i, xvar - 1]],
                [0, st.session_state.variable_saturation[i, yvar-1]], color='k', linewidth=0.5)
        ax.text(st.session_state.variable_saturation[i, xvar - 1], st.session_state.variable_saturation[i, yvar - 1],
                symbol, fontsize='8', ha='right',
                va='bottom')

    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel(f"CP{xvar}")
    plt.ylabel(f"CP{yvar}")
    plt.title('Correlation Cercle')
    plt.grid()

    # Displaying the plot in Streamlit
    st.pyplot(fig)
    st.write("Variable Symbols Key:")

    for symbol, var in symbol_key.items():
        st.write(f"{symbol}: {var}")

def applyingTheKMeans():
    model_kmeans = KMeans(n_clusters=2)
    st.session_state.k = model_kmeans.fit_transform(st.session_state.Y_df)
    st.session_state.k = pd.DataFrame(st.session_state.k)
    st.write(st.session_state.k)
    st.session_state.centroides = model_kmeans.cluster_centers_
    st.write(st.session_state.centroides)
    etiquet_target = pd.DataFrame(model_kmeans.labels_)
    st.write("les etiuettes de sorties sont ", etiquet_target)
    st.write("le nombre d'apparition de chaque target est \n", etiquet_target.value_counts())
    indices = list(range(1, len(st.session_state.valeurs_propres_choisis) + 1))
    xvar = st.selectbox('Select first axis: CP', indices)
    yvar = st.selectbox('Select second axis: CP', indices)
    st.session_state.kmeans_already_applied = True
    fig, ax = plt.subplots()
    ax.scatter(st.session_state.Y_df.iloc[:, xvar - 1], st.session_state.Y_df.iloc[:, yvar - 1],
               c=model_kmeans.labels_.astype(float), s=20,
               alpha=0.5)  # c is colors

    # Scatter plot for centroids
    ax.scatter(st.session_state.centroides[:, xvar - 1], st.session_state.centroides[:, yvar - 1], c='red', s=50)

    # Annotate each point with its index
    for i in range(0, st.session_state.Y_df.shape[0]):
        ax.annotate(st.session_state.Y_df.index[i],
                    (st.session_state.Y_df.iloc[i, xvar - 1], st.session_state.Y_df.iloc[i, yvar - 1]), fontsize=4)

    # Show the plot
    st.pyplot(fig)
    return xvar,yvar


def applyingTheHAC(xvar,yvar):
    d = linkage(st.session_state.Y_df, method='single')

    fig, ax = plt.subplots()
    # Create the dendrogram plot
    plt.title('CAH')
    dendrogram(d, labels=st.session_state.dataSet.index, orientation='top', color_threshold=1)
    ax.scatter(st.session_state.dataSet.iloc[:, xvar - 1], st.session_state.dataSet.iloc[:, yvar - 1])
    ax.set_title('Hierarchical Agglomerative Clustering')
    ax.set_xlabel(f'Feature {xvar}')
    ax.set_ylabel(f'Feature {yvar}')
    st.pyplot(fig)



if uploaded_file is not None:
    dataSetInformations()
    if st.session_state.dataSet.isnull().values.any():
        st.write("## Des valeurs manquantes ont été détectées dans le jeu de données.")
        if st.button("# Remplacer par le moyenne de la variable"):
            replaceNoneValues()
    else:
        st.write("## aucune valeur manquante dans le jeu de données.")
        st.write(st.session_state.dataSet)
    if st.session_state.dataSet.map(lambda x: isinstance(x, str)).values.any(): #verifier si il y a des valeurs non numérique
        st.write("## Des valeurs non-numériques ont été détectées dans le jeu de données.")
        if st.button("# Coder les valeurs"):
            codingTheValues()
    else:
        st.write("## aucune valeur non-numérique dans le jeu de données.")
        st.write(st.session_state.dataSet)

    isReady = not (st.session_state.dataSet.map(lambda x: isinstance(x, str)).values.any() or st.session_state.dataSet.isnull().values.any())

    if (isReady) and not(st.session_state.already_normalized):

        means = st.session_state.dataSet.mean().round(2)
        stds = st.session_state.dataSet.std().round(2)
        st.session_state.is_normalized = (means == 0).all() and ((stds - 1) ==0).all()
        if st.session_state.is_normalized:
            st.write("### la base est normalisée")

        else :
            st.write("## la base est non normalisée")
            if st.button("# Normaliser la base"):
                normalizingTheDataSet()
    if st.session_state.is_normalized or st.session_state.already_normalized:
        manipulatingTheDataSet()

    if st.session_state.showing_correlation:

        if st.button("## Appliquer l'ACP normé") or  st.session_state.already_ACP_Applied:

            applyingTheACP()

            showingTheGraphics()

            st.session_state.acp_applied = True
            st.session_state.already_ACP_Applied=True
    if st.session_state.acp_applied and  st.session_state.already_ACP_Applied:
        st.title("Choisir une méthode")
        # Create a radio select widget

        selected_option = st.radio("Select an option", ["","Kaizer", "Critère du pourcentage"])

        # Use the selected option
        if selected_option == "Kaizer":
            CPWithKaizerMethod()
        elif selected_option =="Critère du pourcentage" :
            CPWithPourcentageMethod()

        if not selected_option=="":
            st.session_state.method_chose = True
            showingTheGraphicsWithCP()

    if st.session_state.method_chose:
        if(st.button("calculer la saturation des varaibels")) or st.session_state.saturation_varaibles_already_calculated:
            variablesLoading()
            st.session_state.corolation_cercle_applied = True
    if st.session_state.corolation_cercle_applied :
        if (st.button("calculer k means")) or st.session_state.kmeans_already_applied:
            xvar,yvar=applyingTheKMeans()
            applyingTheHAC(xvar,yvar)






        


else:
    st.session_state.already_normalized=False
    st.session_state.acp_applied=False
    st.session_state.showing_correlation = False
    st.session_state.already_ACP_Applied= False
    st.session_state.saturation_varaibles_already_calculated=False
    st.session_state.method_chose = False
    st.session_state.corolation_cercle_applied = False
    st.session_state.kmeans_already_applied = False
#    plt.xlabel('Sample Index')
#    plt.ylabel('Distance')
 #       plt.xticks(rotation=90)
  #      plt.tight_layout()
         # Show the dendrogram plot in Streamlit
   #     st.pyplot(fig)

