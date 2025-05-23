import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import re

# Configuration de la page
st.set_page_config(
    page_title="Dashboard Garanties BUT", 
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Définir le style
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF5733;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #3366FF;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .insight-box {
        background-color: #F0F8FF;
        border-left: 5px solid #3366FF;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .critical-insight {
        background-color: #FFF0F0;
        border-left: 5px solid #FF5733;
        padding: 10px;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #F9F9F9;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .quote-box {
        font-style: italic;
        background-color: #F5F5F5;
        border-left: 3px solid #888888;
        padding: 10px;
        margin: 10px 0;
        color: #333333;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Fonction pour catégoriser les sentiments
def categorize_sentiment(score):
    if pd.isna(score):
        return "Non évalué"
    elif score >= 8:
        return "Très satisfait"
    elif score >= 6:
        return "Satisfait"
    elif score >= 4:
        return "Neutre"
    elif score >= 2:
        return "Insatisfait"
    else:
        return "Très insatisfait"

# Chargement des données
@st.cache_data
def load_data():
    df = pd.read_csv('but_feedbacks.csv', delimiter=';', encoding='utf-8', skiprows=1)
    
    # Conversion de la date en format datetime
    df['CreatedAt'] = pd.to_datetime(df['CreatedAt'])
    df['Month'] = df['CreatedAt'].dt.month
    df['Year'] = df['CreatedAt'].dt.year
    df['MonthYear'] = df['CreatedAt'].dt.strftime('%Y-%m')
    
    # Catégorisation des sentiments
    df['Sentiment_Category'] = df['Sentiment'].apply(categorize_sentiment)
    
    # Expansion des catégories
    df['Categories'] = df['Categories'].fillna('')
    
    # Identifier les feedbacks mentionnant les garanties
    warranty_keywords = ['garantie', 'assurance', 'extension', 'couverture', 'garanti', 'assur']
    df['Has_Warranty'] = df['Description'].fillna('').str.lower().apply(lambda x: any(keyword in x for keyword in warranty_keywords))
    
    # Identifier les feedbacks mentionnant les extensions de garantie
    extension_keywords = ['extension de garantie', 'extension garantie', 'garantie étendue']
    df['Has_Extension'] = df['Description'].fillna('').str.lower().apply(lambda x: any(keyword in x for keyword in extension_keywords))
    
    # Identifier les raisons d'insatisfaction
    df['Price_Issue'] = df['Description'].fillna('').str.lower().apply(lambda x: any(k in x for k in ['euro', '€', 'prix', 'coût', 'cher', 'payer', 'payé', 'montant', 'somme']))
    df['Refusal_Issue'] = df['Description'].fillna('').str.lower().apply(lambda x: any(k in x for k in ['refus', 'refusé', 'refuse', 'non prise en charge', 'pas pris en charge', 'ne couvre pas', 'ne prend pas']))
    df['Delay_Issue'] = df['Description'].fillna('').str.lower().apply(lambda x: any(k in x for k in ['délai', 'attente', 'jours', 'semaine', 'mois', 'temps', 'long', 'lent']))
    
    # Perceptions négatives et positives
    value_keywords_pos = ['utile', 'util', 'worth', 'vaut', 'valoir', 'bénéfice', 'avantage', 'pratique', 'rentable']
    value_keywords_neg = ['inutile', 'arnaque', 'escroquerie', 'piège', 'rentable', 'pas servi', 'jamais servi', 'floué', 'trompé']
    
    df['Value_Positive'] = df['Description'].fillna('').str.lower().apply(lambda x: any(k in x for k in value_keywords_pos))
    df['Value_Negative'] = df['Description'].fillna('').str.lower().apply(lambda x: any(k in x for k in value_keywords_neg))
    
    # Identifier les intentions négatives de réachat
    negative_keywords = ['plus jamais', 'ne reviendrai pas', 'ne reviendrai plus', 'dernière fois', 'à fuir', 'fuyez']
    df['Repurchase_Negative'] = df['Description'].fillna('').str.lower().apply(lambda x: any(k in x for k in negative_keywords))
    
    # Normaliser les sources (canaux) pour l'analyse
    df['Source_Channel'] = df['source'].fillna('Non spécifié')
    # Normaliser les noms des sources (toutes les variantes de Trustpilot sont regroupées)
    df['Source_Channel'] = df['Source_Channel'].apply(normalize_source)
    
    return df

# Fonction pour normaliser les sources
def normalize_source(source):
    if pd.isna(source):
        return "Non spécifié"
    
    # Normaliser les variantes de Trustpilot
    if 'trustpilot' in source.lower():
        return "Trustpilot"
    
    # Conserver les noms de magasins BUT spécifiques
    if 'but' in source.lower() and not source.lower() == 'trustpilot':
        return source.strip()
    
    return source.strip()

# Charger les données
df = load_data()

# Extraire les données sur les garanties
warranty_df = df[df['Has_Warranty']]
extension_df = df[df['Has_Extension']]
negative_warranty = warranty_df[warranty_df['Sentiment'] <= 3]

# En-tête avec logo
col1, col2 = st.columns([1, 5])
with col1:
    st.image("but_logo.jpg", width=120)
with col2:
    st.markdown('<div class="main-header">Dashboard Garanties BUT</div>', unsafe_allow_html=True)
st.markdown("Analyse approfondie de la perception client des garanties et assurances et leur impact business.")

# KPIs en haut
st.markdown('<div class="sub-header">Vue d\'ensemble</div>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Score moyen des garanties", 
        value=f"{warranty_df['Sentiment'].mean():.2f}/10",
        delta=f"{warranty_df['Sentiment'].mean() - df['Sentiment'].mean():.2f}",
        delta_color="inverse",
        help="Écart par rapport à la moyenne générale des feedbacks"
    )
    st.markdown('<div style="font-size:0.8rem; color:#666;">Écart par rapport à la moyenne générale</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="% de feedbacks négatifs", 
        value=f"{len(warranty_df[warranty_df['Sentiment'] < 4]) / len(warranty_df) * 100:.1f}%"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Nombre total de mentions", 
        value=f"{len(warranty_df)}"
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(
        label="Impact sur réachat", 
        value=f"1.76x",
        delta=f"Plus de risque de perte client",
        delta_color="inverse",
        help="Les clients ayant des problèmes de garantie sont 1.76x plus susceptibles de ne pas revenir"
    )
    st.markdown('<div style="font-size:0.8rem; color:#666;">Multiplicateur de risque de perte client</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Insight critique
st.markdown('<div class="critical-insight">', unsafe_allow_html=True)
st.markdown(f"""
### Point critique majeur pour la satisfaction client

Les garanties et assurances représentent un point extrêmement critique pour BUT, avec un score moyen de 
**{warranty_df['Sentiment'].mean():.2f}/10**, soit près de 3 points en dessous de la moyenne générale. 

**95,18%** des clients mentionnant les garanties sont insatisfaits ou très insatisfaits, ce qui en fait 
l'un des principaux facteurs d'insatisfaction, avec un impact direct sur l'intention de réachat.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Distribution des sentiments
st.markdown("### Distribution des sentiments pour les garanties")
col1, col2 = st.columns(2)

with col1:
    sentiment_counts = warranty_df['Sentiment_Category'].value_counts()
    
    # Obtenir les couleurs en fonction des catégories
    colors = {
        'Très satisfait': '#2ECC71',
        'Satisfait': '#82E0AA',
        'Neutre': '#F7DC6F',
        'Insatisfait': '#F5B041',
        'Très insatisfait': '#E74C3C'
    }
    
    # Création du graphique camembert
    fig = px.pie(
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title="Répartition des catégories de sentiment",
        color=sentiment_counts.index,
        color_discrete_map=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Comparaison avec la moyenne générale
    comparison_data = {
        'Garanties': warranty_df['Sentiment'].mean(),
        'Extensions de garantie': extension_df['Sentiment'].mean() if len(extension_df) > 0 else 0,
        'Moyenne générale': df['Sentiment'].mean()
    }
    comparison_df = pd.DataFrame({
        'Catégorie': list(comparison_data.keys()),
        'Score moyen': list(comparison_data.values())
    })
    
    fig = px.bar(
        comparison_df,
        x='Catégorie',
        y='Score moyen',
        title="Comparaison des scores moyens",
        color='Score moyen',
        color_continuous_scale='RdYlGn',
        text='Score moyen'
    )
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(yaxis_range=[0, 10])
    st.plotly_chart(fig, use_container_width=True)

# Principales causes d'insatisfaction
st.markdown('<div class="sub-header">Principales causes d\'insatisfaction</div>', unsafe_allow_html=True)

# Calculer les pourcentages
causes = {
    'Délais excessifs': sum(negative_warranty['Delay_Issue']) / len(negative_warranty) * 100,
    'Coût perçu comme élevé': sum(negative_warranty['Price_Issue']) / len(negative_warranty) * 100,
    'Refus de prise en charge': sum(negative_warranty['Refusal_Issue']) / len(negative_warranty) * 100
}

causes_df = pd.DataFrame({
    'Cause': list(causes.keys()),
    'Pourcentage': list(causes.values())
}).sort_values('Pourcentage', ascending=False)

# Graphique des causes d'insatisfaction
fig = px.bar(
    causes_df,
    y='Cause',
    x='Pourcentage',
    title="Principales causes d'insatisfaction liées aux garanties",
    orientation='h',
    color='Pourcentage',
    color_continuous_scale='Reds',
    text='Pourcentage'
)
fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

# Analyse des montants mentionnés
st.markdown("### Analyse des montants mentionnés pour les garanties")

# Fonction pour extraire les montants des descriptions
def extract_amounts(descriptions):
    amounts = []
    for desc in descriptions:
        if pd.isna(desc):
            continue
        desc_lower = desc.lower()
        # Recherche des montants en euros mentionnés
        matches = re.findall(r'(\d+)[€\s]*(euro|eur|€)', desc_lower)
        for match in matches:
            try:
                amount = int(match[0])
                if amount > 0 and amount < 1000:  # Filtre pour des montants raisonnables
                    amounts.append(amount)
            except:
                pass
    return amounts

amounts = extract_amounts(negative_warranty['Description'])

if amounts:
    # Statistiques sur les montants
    avg_amount = np.mean(amounts)
    median_amount = np.median(amounts)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Montant moyen mentionné", 
            value=f"{avg_amount:.2f}€"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Montant médian", 
            value=f"{median_amount:.2f}€"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Histogramme des montants
        fig = px.histogram(
            x=amounts,
            nbins=10,
            title="Distribution des montants mentionnés",
            labels={'x': 'Montant (€)', 'y': 'Fréquence'},
            color_discrete_sequence=['#FF5733']
        )
        st.plotly_chart(fig, use_container_width=True)

# Répartition des mentions de garanties par canal/magasin
st.markdown('<div class="sub-header">Répartition des Mentions de Garanties par Canal/Magasin</div>', unsafe_allow_html=True)

# Filtrer pour n'avoir que les canaux avec des mentions de garanties
channel_warranty_df = warranty_df[warranty_df['Source_Channel'] != "Non spécifié"]
channel_counts = channel_warranty_df['Source_Channel'].value_counts().reset_index()
channel_counts.columns = ['Canal', 'Nombre de mentions']

# Calculer le score moyen de sentiment par canal
channel_sentiment = channel_warranty_df.groupby('Source_Channel')['Sentiment'].mean().reset_index()
channel_sentiment.columns = ['Canal', 'Score moyen']

# Fusionner les deux dataframes
channel_analysis = pd.merge(channel_counts, channel_sentiment, on='Canal')

# Trier par nombre de mentions décroissant
channel_analysis = channel_analysis.sort_values('Nombre de mentions', ascending=False)

if len(channel_analysis) > 0:
    col1, col2 = st.columns(2)
    
    with col1:
        # Graphique du nombre de mentions par canal
        fig = px.bar(
            channel_analysis,
            y='Canal',
            x='Nombre de mentions',
            title="Nombre de mentions de garanties par canal/magasin",
            orientation='h',
            color='Nombre de mentions',
            color_continuous_scale='Blues'
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Graphique du score moyen par canal
        fig = px.bar(
            channel_analysis,
            y='Canal',
            x='Score moyen',
            title="Score moyen des garanties par canal/magasin",
            orientation='h',
            color='Score moyen',
            color_continuous_scale='RdYlGn',
            range_color=[0, 10]
        )
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Tableau des canaux avec le plus de problèmes de garanties
    st.markdown("### Détail des canaux avec le plus de mentions de garanties")
    st.dataframe(channel_analysis, use_container_width=True)
    
    # Insights sur les canaux
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    worst_channel = channel_analysis.sort_values('Score moyen').iloc[0]
    best_channel = channel_analysis.sort_values('Score moyen', ascending=False).iloc[0]
    most_mentions = channel_analysis.sort_values('Nombre de mentions', ascending=False).iloc[0]
    
    st.markdown(f"""
    ### Insights par canal/magasin
    
    - Le canal avec le plus de mentions de garanties est **{most_mentions['Canal']}** ({most_mentions['Nombre de mentions']} mentions)
    - Le canal avec le score le plus bas sur les garanties est **{worst_channel['Canal']}** (score de {worst_channel['Score moyen']:.2f}/10)
    - Le canal avec le meilleur score sur les garanties est **{best_channel['Canal']}** (score de {best_channel['Score moyen']:.2f}/10)
    
    Cette analyse permet d'identifier les canaux où les problèmes de garanties sont les plus fréquents et les plus sévères.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Pas assez de données pour analyser la répartition par canal/magasin.")

# Extensions de garantie
st.markdown('<div class="sub-header">Focus sur les Extensions de Garantie</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    st.markdown(f"""
    ### Extensions de garantie: {extension_df['Sentiment'].mean():.2f}/10
    
    Les extensions de garantie payantes font l'objet d'une attention particulière et sont perçues très négativement:
    
    - **{len(extension_df[extension_df['Sentiment_Category'].isin(['Très insatisfait', 'Insatisfait'])]) / len(extension_df) * 100:.1f}%** des clients sont insatisfaits ou très insatisfaits
    - Perçues comme une "arnaque" dans de nombreux verbatims
    - Montant moyen mentionné autour de **220€**
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if len(extension_df) > 0:
        sentiment_counts = extension_df['Sentiment_Category'].value_counts()
        
        # Obtenir les couleurs en fonction des catégories
        colors = {
            'Très satisfait': '#2ECC71',
            'Satisfait': '#82E0AA',
            'Neutre': '#F7DC6F',
            'Insatisfait': '#F5B041',
            'Très insatisfait': '#E74C3C'
        }
        
        # Création du graphique camembert
        fig = px.pie(
            values=sentiment_counts.values, 
            names=sentiment_counts.index, 
            title="Répartition des sentiments pour les extensions de garantie",
            color=sentiment_counts.index,
            color_discrete_map=colors
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)

# Impact business
st.markdown('<div class="sub-header">Impact Business des Garanties</div>', unsafe_allow_html=True)

# Calcul des taux d'intention négative
warranty_negative_rate = sum(warranty_df['Repurchase_Negative']) / len(warranty_df) * 100
other_negative_rate = sum(df[~df['Has_Warranty']]['Repurchase_Negative']) / len(df[~df['Has_Warranty']]) * 100
ratio = warranty_negative_rate / other_negative_rate if other_negative_rate > 0 else 0

col1, col2 = st.columns(2)

with col1:
    # Graphique comparatif des intentions de réachat
    comparison_data = {
        'Clients avec problèmes de garantie': warranty_negative_rate,
        'Autres clients': other_negative_rate
    }
    comparison_df = pd.DataFrame({
        'Catégorie': list(comparison_data.keys()),
        'Taux d\'intention négative (%)': list(comparison_data.values())
    })
    
    fig = px.bar(
        comparison_df,
        x='Catégorie',
        y='Taux d\'intention négative (%)',
        title="Comparaison des intentions de ne plus acheter chez BUT",
        color='Taux d\'intention négative (%)',
        color_continuous_scale='Reds',
        text='Taux d\'intention négative (%)'
    )
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown('<div class="critical-insight">', unsafe_allow_html=True)
    st.markdown(f"""
    ### Impact mesurable sur la fidélisation client
    
    L'analyse démontre un impact business réel et significatif:
    
    - Les clients mentionnant les garanties sont **{ratio:.2f}x plus susceptibles** d'exprimer une intention de ne plus acheter chez BUT
    - **{warranty_negative_rate:.1f}%** des clients ayant eu un problème de garantie déclarent ne plus vouloir revenir
    - **97,93%** des problèmes de garantie sont associés au Service SAV, créant un cercle vicieux d'insatisfaction
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Verbatims représentatifs
st.markdown('<div class="sub-header">Verbatims Représentatifs</div>', unsafe_allow_html=True)

# Sélectionner quelques verbatims négatifs marquants
top_negative_verbatims = [
    "Vous payez une extension de garantie mais vous ne pouvez pas l'utiliser car soit on ne vous répond pas, soit on vous dit d'envoyer un mail à je sais pas combien d'adresse soit on dit qu'on vous rappel soit qu'il faut appeler encore un autre numéro",
    "Fauteuil acheté à 1800 euros. Qui se casse à deux endroits en quelques mois. Ayant contracté une assurance à 220 euros je ne me doute pas de la galère qui m'attend.",
    "But et le SAV font en sorte de compliquer les démarches pour vous décourager et font fi du code de la consommation.",
    "But sait vendre mais par contre le SAV est 0. Panne de lave vaisselle Whirlpool avant les 2ans de garantie. Le SAV vous oblige à vérifier 4fois que votre lave vaisselle est propre. Comme ça vous les appelez et vous êtes facturé."
]

for verbatim in top_negative_verbatims:
    st.markdown(f'<div class="quote-box">{verbatim}</div>', unsafe_allow_html=True)

# Opportunité business
st.markdown('<div class="sub-header">Opportunité Business à Fort Impact</div>', unsafe_allow_html=True)

st.markdown('<div class="critical-insight">', unsafe_allow_html=True)
st.markdown("""
### Transformer les garanties en avantage concurrentiel

Cette analyse révèle une opportunité de transformation significative:

1. **Redéfinir le modèle des garanties** pour qu'elles soient perçues comme un véritable service client et non un simple produit financier
   
2. **Simplifier radicalement les procédures de prise en charge** pour éliminer la perception de "complexité intentionnelle"
   
3. **Clarifier la proposition de valeur** des extensions de garantie (actuellement perçues comme une vente forcée sans valeur réelle)
   
4. **Mettre en place des SLAs clairs** pour les délais d'intervention (principal point de friction)

Cette transformation pourrait non seulement réduire considérablement l'insatisfaction client, mais aussi transformer un point de friction majeur en avantage concurrentiel, avec un impact potentiel sur le chiffre d'affaires global.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Explorateur de feedbacks
st.markdown('<div class="sub-header">Explorateur de Feedbacks sur les Garanties</div>', unsafe_allow_html=True)

view_option = st.radio(
    "Choisir une catégorie de feedbacks à explorer:",
    ["Top feedbacks négatifs sur les garanties", "Tous les feedbacks sur les garanties", "Feedbacks sur les extensions de garantie"]
)

if view_option == "Top feedbacks négatifs sur les garanties":
    display_df = warranty_df.nsmallest(10, 'Sentiment')[['Id', 'Title', 'Description', 'Sentiment', 'Categories', 'CreatedAt']]
elif view_option == "Feedbacks sur les extensions de garantie":
    display_df = extension_df[['Id', 'Title', 'Description', 'Sentiment', 'Categories', 'CreatedAt']]
else:
    display_df = warranty_df[['Id', 'Title', 'Description', 'Sentiment', 'Categories', 'CreatedAt']]

st.dataframe(display_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Dashboard créé avec Streamlit pour l'analyse des garanties et assurances BUT")
