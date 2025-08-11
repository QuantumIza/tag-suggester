import streamlit as st
import requests

st.set_page_config(page_title="Démo API Tags", layout="centered")
st.title("Démonstration de l'API de suggestion de tags")

# --- Saisie de la question ---
title = st.text_input("📝 Titre de la question")
body = st.text_area("📄 Détails du problème")

# --- Bouton pour interroger l'API ---
if st.button("🔍 Suggérer des tags"):
    if not title:
        st.warning("Le titre est requis pour suggérer des tags.")
    else:
        # Appel à l'API ML
        payload = {"title": title, "body": body}
        try:
            response = requests.post("http://localhost:8000/predict/", json=payload)
            if response.status_code == 200:
                suggested_tags = response.json().get("suggested_tags", [])
                st.session_state["suggested_tags"] = suggested_tags
            else:
                st.error("Erreur lors de l'appel à l'API.")
        except Exception as e:
            st.error(f"Erreur réseau : {e}")

# --- Sélection des tags suggérés ---
if "suggested_tags" in st.session_state:
    st.subheader("Tags suggérés par l'API")
    selected_tags = st.multiselect(
        "Sélectionne les tags (obligatoire)",
        options=st.session_state["suggested_tags"],
        placeholder="Choisis au moins un tag"
    )
    # --- AUTRE OPTION AFFICHAGE DES TEGS PROPOSES
    st.subheader("Sélection alternative des tags (visuelle)")

    selected_checkboxes = []
    for tag in st.session_state["suggested_tags"]:
        if st.checkbox(tag, key=f"tag_{tag}"):
            selected_checkboxes.append(tag)

    if selected_checkboxes:
        st.success(f"Tags cochés : {', '.join(selected_checkboxes)}")
        st.button("➡️ Continuer avec cette sélection")
    else:
        st.info("Coche au moins un tag pour continuer.")


    # --- Bouton Next conditionnel ---
    if selected_tags:
        st.success(f"Tags sélectionnés : {', '.join(selected_tags)}")
        st.button("✅ Next")
    else:
        st.warning("Tu dois sélectionner au moins un tag pour continuer.")
