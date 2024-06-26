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


#C’est une fonction utiliser pour simuler le chargement lorsqu’ on appuie sur un bouton
def progressBar():
    progress_bar = st.progress(0)
    # Simulate progress
    for i in range(11):
        progress_bar.progress(i * i)
        time.sleep(0.1)
st.title('Analyse et Fouille de données d’une base de données médicales')
st.title("Dataset Importer")
st.session_state.is_normalized = False


#importation du fichier
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

def dataSetInformations():
    #ajouter un subheader
    st.subheader('Raw data')

    dataSet = pd.read_csv(uploaded_file)
    st.write(dataSet)
    #enregistrement  du dataset au session pour que on puisse le utiliser à chaque reload du code
    if "dataSet" not in st.session_state:
        st.session_state.dataSet = dataSet

    st.write("### informations")

    #affectation du nombre d'observations et du caractéristiques
    numberOfRows, numberOfColumns = st.session_state.dataSet.shape
    #mise en page de deux colones
    col1, col2 = st.columns(2)
    with col1:
        #affichage
        st.write("### Nombre d'observations :")
        st.markdown(f"<p style='color: green;font-weight: bold; font-size: 24px;'>{numberOfRows}</p>",
                    unsafe_allow_html=True)

    with col2:
        # affichage
        st.write("### Nombre de caractéristiques :")
        st.markdown(f"<p style='color: green; font-weight: bold; font-size: 24px;'>{numberOfColumns}</p>",
                    unsafe_allow_html=True)



#remplacement des valuers NaN
def replaceNoneValues():
    #simulation d'un bar de progress
    progressBar()
    for column in st.session_state.dataSet.columns:
        #on fait le traitement uniquement au colones numériques
        if (st.session_state.dataSet[column].dtype == 'int64' or st.session_state.dataSet[column].dtype == 'float64'):
            #on les remplace
            st.session_state.dataSet[column] = st.session_state.dataSet[column].fillna(st.session_state.dataSet[column].mean())
    time.sleep(0.1)

    st.write("remplacement est terminé")
    st.write(st.session_state.dataSet)


#codage des colonnes non-numériques
def codingTheValues():
    progressBar()
    this_year = date.datetime.now().year
    for column in st.session_state.dataSet.columns:
        # si les colones sont des colones de date(n'est pas le cas dans notre exemple dataset mais il faut le traiter pour qu'il étre un code général)
        if st.session_state.dataSet[column].astype(str).str.match(r'^\d{2}-\d{2}-\d{4}$').any():
            st.session_state.dataSet[column] = this_year - st.session_state.dataSet[column].str[-4:].astype(int)
            continue
        #si il sont des chaines de caractéres
        if not (st.session_state.dataSet[column].dtype == 'int64' or st.session_state.dataSet[
            column].dtype == 'float64'):
            my_dict = {}
            index = 0
            #on applique un index à chaque chaine
            for element in st.session_state.dataSet[column]:
                if element not in my_dict:
                    my_dict[element] = index
                    index += 1
            # on les remplace par son index
            for key, value in my_dict.items():
                st.session_state.dataSet[column] = st.session_state.dataSet[column].replace(key, value)
    st.write("codage est terminé")
    st.write(st.session_state.dataSet)

#normalisation du dataSet
def normalizingTheDataSet():
    progressBar()
    #on initialise une instance StandardScaler
    scaler = StandardScaler()
    #on utilise la fonction fit_transform pour avoir la matrice centré reduite
    normalized_data = scaler.fit_transform(st.session_state.dataSet)

    #on le transforme on une dataFrame (st.session est utiliser pour enregistrer la dataset car streamlit reload the code à chaque interaction)
    st.session_state.normalized_df = pd.DataFrame(normalized_data, columns=st.session_state.dataSet.columns)
    #on utilise c'est variable pour éviter que le code déja exécuter ne s'exécute pas aprés le reload du code (j'expliquerais plus lorsque on avance )
    st.session_state.is_normalized = True
    st.session_state.already_normalized = True
    st.write("### la base a été normalisée")

#manipulation du dataSet après la normalisation
def manipulatingTheDataSet():
    st.subheader('base normalisée')
    st.write(st.session_state.normalized_df)

    fig, ax = plt.subplots(figsize=(12, 12))
    st.session_state.corrélation_df = st.session_state.normalized_df.corr()  # les données doit etre centré reduite pour faire la corrélation
    sb.heatmap(st.session_state.corrélation_df, annot=True, ax=ax)
    st.pyplot(fig)
    st.write("#### le couple de variables les plus corrélées est séléctionné par défaut")
    #cette instruction est utiliser pour avoir le couple le plus corrolés
    xvar, yvar = st.session_state.corrélation_df[st.session_state.corrélation_df < 1].stack().idxmax()
    #ajout de deux select items
    xvar = st.selectbox('Select x-axis:', st.session_state.corrélation_df.columns[:],
                        index=st.session_state.corrélation_df.columns.get_loc(
                            xvar))  # get_loc est utulisé pour avoir l'indice
    yvar = st.selectbox('Select y-axis:', st.session_state.corrélation_df.columns[:],
                        index=st.session_state.corrélation_df.columns.get_loc(yvar))
    #affichage du corrélation scatter
    st.write(px.scatter(st.session_state.corrélation_df, x=xvar, y=yvar))

    #on utilise c'est variable pour éviter que le code déja exécuter ne s'exécute pas aprés le reload du code (j'expliquerais plus lorsque on avance )
    st.session_state.showing_correlation = True

#application de l'ACP
def applyingTheACP():
    # cette méthode est claire , j'applique l'acp , affectation des valeurs porpres,pourcentage ,pourcentage cumulé  et après affichage
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

    st.table(st.session_state.données)
    st.title("Histogramme des valeurs propres")


#affichage

def showingTheGraphics():
    #affichage des valeurs propress ,pourcentage cumulé etc .....
    fig, ax = plt.subplots()
    ax.bar(np.arange(0, 15), st.session_state.valeurs_propres, color="teal")
    ax.plot(np.arange(0, 15), st.session_state.valeurs_propres, color="red")
    ax.set_xlabel("Composantes principales")
    ax.set_ylabel("Valeurs propres")
    st.pyplot(fig)


    st.title("Pourcentage d'inertie cumulé")

    fig, ax = plt.subplots()

    ax.plot(np.arange(1, 16), np.cumsum(st.session_state.acp_normé.explained_variance_ratio_))
    ax.set_xlabel("Nombre de composantes principales")
    ax.set_ylabel("Pourcentage d'inertie cumulé")
    st.pyplot(fig)



#méthode de kaizer
def CPWithKaizerMethod():
    # on prend que les valeurs supérieur à 1
    st.session_state.valeurs_propres_choisis = st.session_state.valeurs_propres[st.session_state.valeurs_propres > 1]
    output = []
    #affichage
    for i in range(len(st.session_state.valeurs_propres_choisis)):
        output.append({
            'Composante Principale': 'CP{}'.format(i + 1),
            'Valeur Propre': st.session_state.valeurs_propres[i]})
    st.table(output)



#utilisation du silder pour la pourcentage
def CPWithPourcentageMethod():
    pourcentage = st.slider('Pourcentage', min_value=20, max_value=100, value=60)
    #cas particulier car lorsque on choisie 100 la pourcentage cumulé avec python est supérieur à 100 donc un ajoute un nombre quelquonqe
    if pourcentage == 100:
        pourcentage += 1
    st.session_state.valeurs_propres_choisis = []
    for i in range(len(st.session_state.données)):
        #ajout des cp inférieur au pourcentage choisis
        if (st.session_state.données[i]["Pourcentage Cumulé"] < pourcentage):

            st.session_state.valeurs_propres_choisis.append(st.session_state.données[i]["Composante Principale"])
        else:
            break
    output = []
    #affichage
    for i in range(len(st.session_state.valeurs_propres_choisis)):
        output.append({
            'Composante Principale': 'CP{}'.format(i + 1),
            'Valeur Propre': st.session_state.valeurs_propres[i],
            'Pourcentage Cumulé': st.session_state.pourcentage_cumulé[i]})
    st.write("## Composantes choisis")
    st.table(output)


def showingTheGraphicsWithCP():
    #affichage
    indices = list(range(1, len(st.session_state.valeurs_propres_choisis) + 1))
    xvar = st.selectbox('Select x-axis:', indices)
    yvar = st.selectbox('Select y-axis:', indices)
    st.write(px.scatter(st.session_state.Y_df, x=xvar - 1, y=yvar-1))

#saturation du variables
def variablesLoading():
    #calcul du corrélation entre les individus et le cps
    st.session_state.variable_saturation = st.session_state.acp_normé.components_.T * np.sqrt(
        st.session_state.acp_normé.explained_variance_)
    # on utilise c'est variable pour éviter que le code déja exécuter ne s'exécute pas aprés le reload du code
    st.session_state.saturation_varaibles_already_calculated = True
    st.write("Saturation des variables:")
    #affichage
    st.write(pd.DataFrame(st.session_state.variable_saturation,
                          columns=[f"PC{i + 1}" for i in range(len(st.session_state.normalized_df.columns))],
                          index=st.session_state.normalized_df.columns))

    st.session_state.variable_symbols = [f"I{i + 1}" for i in range(len(st.session_state.normalized_df.columns))]

    # utilisation du clé comme I1 et I2
    symbol_key = {symbol: var for symbol, var in
                  zip(st.session_state.variable_symbols, st.session_state.normalized_df.columns)}
    # affichage du cercle de corrélation
    fig, ax = plt.subplots()
    ax.set_aspect('equal', 'box')
    ax.add_artist(plt.Circle((0, 0), 1, color='blue', fill=False))
    indices = list(range(1, len(st.session_state.valeurs_propres_choisis) + 1))
    #selection
    xvar = st.selectbox('Select x-axis: CP', indices)
    yvar = st.selectbox('Select y-axis: CP', indices)
    #remplissement du cercle dynamiquement tout dépend du xvar et yvar choisit ,c'est xvar -1 et yvar-1 car on commence par 0 en python
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
    plt.title('Cercle de corrélation')
    plt.grid()


    st.pyplot(fig)
    #affichage des clés
    st.write("Variable Symbols Key:")

    for symbol, var in symbol_key.items():
        st.write(f"{symbol}: {var}")


#application du kmeans
def applyingTheKMeans():
    #initialisation avec k=2
    model_kmeans = KMeans(n_clusters=2)
    #application du kmeans
    st.session_state.k = model_kmeans.fit_transform(st.session_state.Y_df)
    #transformation en dataframe
    st.session_state.k = pd.DataFrame(st.session_state.k)
    st.write("Résultats des clusters :")
    #affichage des clusters
    st.write(st.session_state.k)
    #affectation des centres de gravités
    st.session_state.centroides = model_kmeans.cluster_centers_
    st.write("Centroides des clusters :")
    st.write(st.session_state.centroides)
    #chaque observation appartient au cluster N°
    etiquet_target = pd.DataFrame(model_kmeans.labels_)
    st.write("chaque observation appartient au cluster N°:", etiquet_target)
    st.write("nombre d'observations dans chaque cluster \n", etiquet_target.value_counts())
    #inertie total
    inertie_totale = model_kmeans.inertia_
    st.title("K-means Inertie totale: {}".format(inertie_totale))
    indices = list(range(1, len(st.session_state.valeurs_propres_choisis) + 1))
    xvar = st.selectbox('Select first axis: CP', indices)
    yvar = st.selectbox('Select second axis: CP', indices)
    #de meme pour tous les variables
    st.session_state.kmeans_already_applied = True
    #affichage
    fig, ax = plt.subplots()
    ax.scatter(st.session_state.Y_df.iloc[:, xvar - 1], st.session_state.Y_df.iloc[:, yvar - 1],
               c=model_kmeans.labels_.astype(float), s=20,
               alpha=0.5)  # c is colors
    ax.set_title('K-means')
    # Scatter plot for centroids
    ax.scatter(st.session_state.centroides[:, xvar - 1], st.session_state.centroides[:, yvar - 1], c='red', s=50)

    # Annotation de chaque point avec son index
    for i in range(0, st.session_state.Y_df.shape[0]):
        ax.annotate(st.session_state.Y_df.index[i],
                    (st.session_state.Y_df.iloc[i, xvar - 1], st.session_state.Y_df.iloc[i, yvar - 1]), fontsize=4)

    # affichage
    st.pyplot(fig)


    return xvar,yvar

#application du HAC
def applyingTheHAC(xvar,yvar):
    d = linkage(st.session_state.Y_df, method='ward')

    fig, ax = plt.subplots()


    labels = fcluster(d, 2, criterion='maxclust')  # Choisir le nombre de clusters

    plt.title('CAH')
    #création de dendogram
    dendrogram(d, labels=st.session_state.dataSet.index, orientation='top',color_threshold=25)
    ax.scatter(st.session_state.dataSet.iloc[:, xvar - 1], st.session_state.dataSet.iloc[:, yvar - 1])
    ax.set_title('Hierarchical Agglomerative Clustering')
    ax.set_xlabel(f'Feature {xvar}')
    ax.set_ylabel(f'Feature {yvar}')
    st.pyplot(fig)
    #avoir les centres de gravités
    cluster_centers = np.array([st.session_state.Y_df[labels == i].mean(axis=0) for i in np.unique(labels)])

    inertia = 0
    #calculons l'inertie totale
    for i in np.unique(labels):
        cluster_points = st.session_state.Y_df[labels == i]
        cluster_centroid = cluster_centers[i - 1]  # Centroid for label i
        squared_distances = np.sum((cluster_points - cluster_centroid) ** 2, axis=1)
        inertia += np.sum(squared_distances)

    st.title(f'Hierarchical Agglomerative Clustering - inertie totale: {inertia}')

#s' il y a un fichier importer on le traite
if uploaded_file is not None:
    dataSetInformations()
    #si il y a des valeurs manquantes
    if st.session_state.dataSet.isnull().values.any():
        st.write("## Des valeurs manquantes ont été détectées dans le jeu de données.")
        #si on clique sur le bouton
        if st.button("# Remplacer par le moyenne de la variable"):
            #on fait cette fonction
            replaceNoneValues()
    else:
        st.write("## aucune valeur manquante dans le jeu de données.")
        st.write(st.session_state.dataSet)
    if st.session_state.dataSet.map(lambda x: isinstance(x, str)).values.any(): #verifier si il y a des valeurs non numérique
        st.write("## Des valeurs non-numériques ont été détectées dans le jeu de données.")
        if st.button("# Coder les valeurs"):
            #on fait cette méthode
            codingTheValues()
    else:
        st.write("## aucune valeur non-numérique dans le jeu de données.")
        st.write(st.session_state.dataSet)
    #pas de valeurs manquantes et du cahines de cractéres càd que le code précédent est éxécuter
    isReady = not (st.session_state.dataSet.map(lambda x: isinstance(x, str)).values.any() or st.session_state.dataSet.isnull().values.any())
    # on a appliqué le code precédent mais pas encore le code suivant
    if (isReady) and not(st.session_state.already_normalized):

        means = st.session_state.dataSet.mean().round(2)
        stds = st.session_state.dataSet.std().round(2)
        st.session_state.is_normalized = (means == 0).all() and ((stds - 1) ==0).all()
        if st.session_state.is_normalized:
            st.write("### la base est normalisée")

        else :
            st.write("## la base est non normalisée")
            if st.button("# Normaliser la base"):
                #on normalize la base
                normalizingTheDataSet()
    # si on a en train d'exécuter le code précendent ou on a déja éxecuter est c"est un reload
    if st.session_state.is_normalized or st.session_state.already_normalized:

        manipulatingTheDataSet()
    #si l'affichage du corrélation est éxecuté
    if st.session_state.showing_correlation:
        #si on clique sur le bouton d'application d'ACP ou on a déja cliqué est c'est un reload
        if st.button("## Appliquer l'ACP normé") or  st.session_state.already_ACP_Applied:

            applyingTheACP()

            showingTheGraphics()
            #affectation des variables pour indiqué que cette code est déja fait
            st.session_state.acp_applied = True
            st.session_state.already_ACP_Applied=True
    #si on a cliquer sur appliqué l'acp
    if st.session_state.acp_applied and  st.session_state.already_ACP_Applied:
        st.title("Choisir une méthode")
        #radio selection

        selected_option = st.radio("Select an option", ["","Kaizer", "Critère du pourcentage"])


        if selected_option == "Kaizer":
            CPWithKaizerMethod()
        elif selected_option =="Critère du pourcentage" :
            CPWithPourcentageMethod()
        # si il a selection soit kaizer soit Critère du pourcentage
        if not selected_option=="":
            #on initialise le variable
            st.session_state.method_chose = True
            #et on fait cette méthode
            showingTheGraphicsWithCP()
    # si le code précedent est déja éxecuter
    if st.session_state.method_chose:
        #si on clique le bouton ou il est déja cliquer est c'est un reload
        if(st.button("calculer la saturation des varaibels")) or st.session_state.saturation_varaibles_already_calculated:
            variablesLoading()
            # on initialise le variable
            st.session_state.corolation_cercle_applied = True
    # si le code précedent est déja éxecuter
    if st.session_state.corolation_cercle_applied :
        #si on clique le bouton ou il est déja cliquer est c'est un reload

        if (st.button("calculer k means")) or st.session_state.kmeans_already_applied:
            xvar,yvar=applyingTheKMeans()
            applyingTheHAC(xvar,yvar)






        

#si non on initialise les variables booléan à False
else:
    st.session_state.already_normalized=False
    st.session_state.acp_applied=False
    st.session_state.showing_correlation = False
    st.session_state.already_ACP_Applied= False
    st.session_state.saturation_varaibles_already_calculated=False
    st.session_state.method_chose = False
    st.session_state.corolation_cercle_applied = False
    st.session_state.kmeans_already_applied = False

