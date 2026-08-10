# ==============================================================================
# PAGE D'ACCUEIL — VUE GÉNÉRALE
# ==============================================================================
#
# C'est la première chose que verra un décideur en ouvrant l'outil. Le but :
# lui donner, en un seul coup d'œil, l'état du jour (score de risque, volume,
# ITE, ITR), et un bandeau d'alerte clair si un scénario de risque combiné
# est en cours (ex : ITE ET ITR élevés en même temps -> règle combinée 1
# du script 09, "Risque thématique + Tension émotionnelle").
# ==============================================================================

import streamlit as st
import plotly.graph_objects as go
from utils.chargement_donnees import charger_indicateurs_quotidiens, charger_alertes_quotidiennes, charger_journal_alertes

st.set_page_config(page_title="Veille post-électorale Cameroun", layout="wide", page_icon="🇨🇲")

st.title("🇨🇲 Système d'aide à la décision : Crises post-électorales")
st.caption("Analyse en temps réel des sentiments et thématiques numériques : élection présidentielle 2025")

# ------------------------------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ------------------------------------------------------------------------------
indicateurs = charger_indicateurs_quotidiens()
alertes = charger_alertes_quotidiennes()
journal = charger_journal_alertes()

# On ne prend PAS simplement le dernier jour du calendrier : la fenêtre
# d'analyse (±90 jours autour du vote, script 08) inclut des jours creux,
# parfois avec 0 publication réelle, ce qui rendrait les indicateurs du jour
# ininterprétables (ITE à 0 ne veut rien dire s'il n'y a aucune publication
# ce jour-là). On utilise donc le dernier jour marqué "fiable" par le script
# 08 (au moins 5 publications ce jour-là, colonne jour_fiable).
jours_fiables_disponibles = indicateurs[indicateurs["jour_fiable"]]
dernier_jour = jours_fiables_disponibles.index.max()
ligne_du_jour = indicateurs.loc[dernier_jour]
ligne_alerte_du_jour = alertes.loc[dernier_jour]

dernier_jour_calendrier = indicateurs.index.max()
if dernier_jour_calendrier != dernier_jour:
    st.caption(
        f"ℹ️ Les derniers jours de la fenêtre d'analyse (jusqu'au "
        f"{dernier_jour_calendrier.strftime('%d/%m/%Y')}) contiennent trop peu de "
        f"publications (< 5/jour) pour être statistiquement fiables. "
        f"Le dernier jour fiable disponible est donc affiché : "
        f"**{dernier_jour.strftime('%d/%m/%Y')}**."
    )

# ------------------------------------------------------------------------------
# BANDEAU DE SCÉNARIO — LE CŒUR DE CETTE PAGE
# ------------------------------------------------------------------------------
# On regarde si, pour le dernier jour disponible, la règle combinée 1 s'est
# déclenchée (ITE en Orange/Rouge ET ITR en Orange/Rouge simultanément).
# C'est exactement le scénario qu'on avait décrit : une hausse conjointe de
# l'ITE et de l'ITR ne doit pas juste faire monter deux chiffres discrets
# dans un coin de l'écran, elle doit sauter aux yeux du décideur.

st.divider()

if ligne_alerte_du_jour["regle_combinee_1_risque_tension"]:
    st.error(
        "🟣 **ALERTE CRITIQUE — Risque thématique + Tension émotionnelle** \n\n"
        f"Aujourd'hui ({dernier_jour.strftime('%d/%m/%Y')}), l'ITE "
        f"({ligne_du_jour['ITE']:.0f}/100) **et** l'ITR ({ligne_du_jour['ITR']:.0f}/100) "
        "sont tous les deux en zone Orange ou Rouge en même temps. "
        "Cela signifie que les publications du jour combinent à la fois une forte "
        "charge émotionnelle négative (colère, peur, dégoût, tristesse) **et** une "
        "forte présence de thématiques à risque (tribalisme, contestation, pouvoir). "
        "\n\n**Réponse institutionnelle suggérée** : transmission immédiate à la cellule "
        "de veille, vérification des remontées de terrain dans les zones concernées, "
        "préparation d'un message de communication de crise."
    )
elif ligne_alerte_du_jour["regle_combinee_2_mobilisation_polarisation"]:
    # CORRECTION : cette règle combinée classe aussi le jour en "Critique" dans
    # le script 09 (voir determiner_alerte_globale), au même titre que la règle 1.
    # On utilise donc st.error (le même niveau visuel que ci-dessus), pas
    # st.warning, pour rester cohérent avec le journal et la page Alertes.
    st.error(
        "🟣 **ALERTE CRITIQUE : Mobilisation + Polarisation** \n\n"
        f"Aujourd'hui ({dernier_jour.strftime('%d/%m/%Y')}), des appels à l'action "
        f"collective (Mobilisation : {ligne_du_jour['Mobilisation']:.0f}/100) et un "
        f"public très divisé (Polarisation : {ligne_du_jour['Polarisation']:.0f}/100) "
        "sont détectés simultanément : terrain propice à un débordement."
    )
elif ligne_alerte_du_jour["ALERTE_GLOBALE"] == "Rouge":
    st.warning(f"🔴 Alerte Rouge active aujourd'hui — Score de Risque Global : {ligne_du_jour['Score_Risque_Global']:.0f}/100")
elif ligne_alerte_du_jour["ALERTE_GLOBALE"] == "Orange":
    st.info(f"🟠 Alerte Orange active aujourd'hui — Score de Risque Global : {ligne_du_jour['Score_Risque_Global']:.0f}/100")
else:
    st.success("🟢 Aucune alerte active aujourd'hui — climat numérique stable.")

# ------------------------------------------------------------------------------
# LES 4 INDICATEURS CLÉS DU JOUR
# ------------------------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)
col1.metric("Score de Risque Global", f"{ligne_du_jour['Score_Risque_Global']:.1f}/100")
col2.metric("Volume du jour", int(ligne_du_jour["volume_publications"]))
col3.metric("ITE — Tension émotionnelle", f"{ligne_du_jour['ITE']:.1f}/100")
col4.metric("ITR — Risque thématique", f"{ligne_du_jour['ITR']:.1f}/100")

st.divider()

# ------------------------------------------------------------------------------
# COURBE DU SCORE DE RISQUE GLOBAL
# ------------------------------------------------------------------------------

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=indicateurs.index, y=indicateurs["Score_Risque_Global"],
    mode="lines", name="Score de Risque Global", line=dict(color="black", width=2),
))
# On superpose l'ITE et l'ITR en transparence pour visualiser le scénario
# "hausse conjointe" directement sur le graphique
fig.add_trace(go.Scatter(
    x=indicateurs.index, y=indicateurs["ITE"],
    mode="lines", name="ITE", line=dict(color="#6A0572", width=1, dash="dot"),
))
fig.add_trace(go.Scatter(
    x=indicateurs.index, y=indicateurs["ITR"],
    mode="lines", name="ITR", line=dict(color="#9D0208", width=1, dash="dot"),
))
fig.update_layout(
    title="Évolution du Score de Risque Global (avec ITE et ITR en superposition)",
    height=420, hovermode="x unified", yaxis_title="Indice (0-100)",
)
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------------------
# DERNIÈRES ALERTES DU JOURNAL
# ------------------------------------------------------------------------------

st.subheader("Dernières alertes enregistrées")
st.dataframe(journal.tail(10), use_container_width=True)

st.caption(
    "Navigue vers les autres pages via le menu à gauche : Sentiments, Émotions, "
    "Thématiques, Indicateurs, Alertes, Recherche, Rapports."
)
