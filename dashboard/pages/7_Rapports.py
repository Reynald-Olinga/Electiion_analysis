# ==============================================================================
# PAGE RAPPORTS — EXPORTS PDF / EXCEL / CSV
# ==============================================================================

import streamlit as st
import pandas as pd
from io import BytesIO
from datetime import datetime
from fpdf import FPDF, XPos, YPos

from utils.chargement_donnees import (
    charger_indicateurs_quotidiens, charger_alertes_completes, charger_journal_alertes,
    charger_episodes_alerte, charger_synthese_themes, charger_repartition_sentiment,
    charger_repartition_emotions, charger_repartition_themes_fr,
)

st.set_page_config(page_title="Rapports — Veille post-électorale", layout="wide", page_icon="📁")
st.title("📁 Rapports")
st.caption("Exports PDF, Excel et CSV — pour partager l'état de la veille en dehors du dashboard")

indicateurs = charger_indicateurs_quotidiens()
alertes = charger_alertes_completes()
journal = charger_journal_alertes()
episodes = charger_episodes_alerte()
synthese_themes = charger_synthese_themes()
repartition_sentiment = charger_repartition_sentiment()
repartition_emotions = charger_repartition_emotions()
repartition_themes = charger_repartition_themes_fr()

# Dernier jour fiable
jours_fiables = indicateurs[indicateurs["jour_fiable"]]
dernier_jour = jours_fiables.index.max() if len(jours_fiables) > 0 else indicateurs.index.max()
ligne_du_jour = indicateurs.loc[dernier_jour]
niveau_du_jour = alertes.loc[dernier_jour, "ALERTE_GLOBALE"]

onglet1, onglet2, onglet3 = st.tabs(["📄 Rapport PDF", "📊 Export Excel", "📁 Export CSV"])


# ==============================================================================
# UTILITAIRES PDF
# ==============================================================================
def texte_pdf(s):
    """
    fpdf2 avec polices standard = encodage Latin-1.
    On remplace silencieusement les caractères non supportés (emojis, etc.).
    """
    return str(s).encode("latin-1", errors="replace").decode("latin-1")


def generer_pdf():
    pdf = FPDF()
    pdf.add_page()

    # --- En-tête ---
    pdf.set_font("Helvetica", "B", 18)
    pdf.cell(0, 12, texte_pdf("Rapport de synthèse — Veille post-électorale"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 8, texte_pdf(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Section 1 : état du jour ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, texte_pdf(f"1. État au {dernier_jour.strftime('%d/%m/%Y')}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, texte_pdf(f"Niveau d'alerte global : {niveau_du_jour}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, texte_pdf(f"Score de Risque Global : {ligne_du_jour['Score_Risque_Global']:.1f} / 100"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, texte_pdf(f"ITE (Tension émotionnelle) : {ligne_du_jour['ITE']:.1f} / 100"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, texte_pdf(f"ITR (Risque thématique) : {ligne_du_jour['ITR']:.1f} / 100"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, texte_pdf(f"Volume de publications ce jour-là : {int(ligne_du_jour['volume_publications'])}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Section 2 : historique des alertes ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, texte_pdf("2. Historique des alertes"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, texte_pdf(f"Jours analysés : {len(indicateurs)}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, texte_pdf(f"Jours en alerte (Orange/Rouge/Critique) : {len(journal)}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 7, texte_pdf(f"Nombre d'épisodes d'alerte : {len(episodes)}"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    if len(episodes) > 0:
        episode_le_plus_long = episodes.loc[episodes["duree_jours"].idxmax()]
        pdf.cell(0, 7, texte_pdf(
            f"Épisode le plus long : {episode_le_plus_long['duree_jours']} jours "
            f"({episode_le_plus_long['date_debut'].strftime('%d/%m/%Y')} au "
            f"{episode_le_plus_long['date_fin'].strftime('%d/%m/%Y')}, niveau {episode_le_plus_long['niveau_maximal']})"
        ), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Section 3 : répartition du sentiment ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, texte_pdf("3. Répartition du sentiment (corpus complet)"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for _, ligne in repartition_sentiment.iterrows():
        pdf.cell(0, 7, texte_pdf(f"  {ligne['sentiment']} : {int(ligne['count']):,} publications".replace(",", " ")),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # --- Section 4 : thèmes dominants ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, texte_pdf("4. Thèmes dominants (français)"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 11)
    for _, ligne in repartition_themes.sort_values("count", ascending=False).iterrows():
        pdf.multi_cell(
            0, 7, texte_pdf(f"  - {ligne['theme_nom']} : {int(ligne['count']):,} publications".replace(",", " ")),
            new_x=XPos.LMARGIN, new_y=YPos.NEXT,
        )
    pdf.ln(4)

    # --- Section 5 : note méthodologique ---
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, texte_pdf("5. Note méthodologique"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(
        0, 6, texte_pdf(
            "Les indicateurs (ITE, ITR, Score de Risque Global) sont normalisés de façon "
            "relative à la fenêtre de données observée, pas selon des seuils absolus "
            "universels. Les seuils Orange/Rouge sont calibrés par percentile (75e/90e). "
            "Ce rapport est un outil d'aide à la décision et ne remplace pas le jugement "
            "des acteurs institutionnels."
        ),
        new_x=XPos.LMARGIN, new_y=YPos.NEXT,
    )

    return bytes(pdf.output())


# ==============================================================================
# ONGLET 1 — PDF
# ==============================================================================
with onglet1:
    st.subheader("Aperçu du contenu du rapport")
    st.markdown(f"""
    - **État au {dernier_jour.strftime('%d/%m/%Y')}** : niveau d'alerte, Score de Risque Global, ITE, ITR
    - **Historique des alertes** : nombre de jours en alerte, épisodes, épisode le plus long
    - **Répartition du sentiment** sur l'ensemble du corpus
    - **Thèmes dominants** (français)
    - **Note méthodologique** rappelant les limites du système
    """)

    pdf_bytes = generer_pdf()
    st.download_button(
        "⬇️ Télécharger le rapport PDF", data=pdf_bytes,
        file_name=f"rapport_veille_{datetime.now().strftime('%Y%m%d')}.pdf", mime="application/pdf",
    )


# ==============================================================================
# ONGLET 2 — EXCEL (CORRIGÉ : feuilles séparées)
# ==============================================================================
with onglet2:
    st.subheader("Classeur Excel multi-feuilles")
    st.markdown("""
    Le fichier généré contient 6 feuilles :
    - `Indicateurs_quotidiens`
    - `Journal_alertes`
    - `Episodes_alerte`
    - `Synthese_themes`
    - `Repartition_sentiment`
    - `Repartition_emotions`
    """)

    buffer_excel = BytesIO()
    with pd.ExcelWriter(buffer_excel, engine="openpyxl") as writer:
        indicateurs.to_excel(writer, sheet_name="Indicateurs_quotidiens")
        journal.to_excel(writer, sheet_name="Journal_alertes")
        episodes.to_excel(writer, sheet_name="Episodes_alerte", index=False)
        synthese_themes.to_excel(writer, sheet_name="Synthese_themes")
        # CORRECTION : deux feuilles distinctes au lieu d'écraser la même
        repartition_sentiment.to_excel(writer, sheet_name="Repartition_sentiment", index=False)
        repartition_emotions.to_excel(writer, sheet_name="Repartition_emotions", index=False)
    buffer_excel.seek(0)

    st.download_button(
        "⬇️ Télécharger le classeur Excel", data=buffer_excel,
        file_name=f"tableaux_veille_{datetime.now().strftime('%Y%m%d')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ==============================================================================
# ONGLET 3 — CSV
# ==============================================================================
with onglet3:
    st.subheader("Téléchargements CSV individuels")
    tableaux_disponibles = {
        "Indicateurs quotidiens complets": indicateurs,
        "Journal des alertes": journal,
        "Épisodes d'alerte": episodes,
        "Synthèse par thème": synthese_themes,
        "Répartition du sentiment": repartition_sentiment,
        "Répartition des émotions dominantes": repartition_emotions,
        "Répartition des thèmes (français)": repartition_themes,
    }

    for nom, df in tableaux_disponibles.items():
        col_nom, col_bouton = st.columns([3, 1])
        col_nom.write(f"**{nom}** ({len(df)} lignes)")
        csv_bytes = df.to_csv(index=(df.index.name is not None)).encode("utf-8-sig")
        col_bouton.download_button(
            "⬇️ CSV", data=csv_bytes, file_name=f"{nom.lower().replace(' ', '_')}.csv",
            mime="text/csv", key=f"dl_{nom}",
        )