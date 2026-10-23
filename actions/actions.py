import re
import json
from datetime import datetime
import unicodedata
from rasa_sdk.events import SlotSet 

from .llm import reformuler

from typing import Any, Text, Dict, List 

from rasa_sdk import Action, FormValidationAction, Tracker 
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from rasa_sdk.events import SlotSet
from .rag import MOTEUR

HORAIRES = {
    "lundi": "de 8h à 17h",
    "mardi": "de 8h à 17h",
    "mercredi": "de 8h à 17h",
    "jeudi": "de 8h à 17h",
    "vendredi": "de 8h à 13h",
    "samedi": None,     
    "dimanche": None,    
}

SERVICES = {
    "pilotage": "Le pilotage consiste à guider les navires dans le chenal d'accès et jusqu'à leur poste à quai, avec un pilote à bord.",
    "remorquage": "Le remorquage utilise des remorqueurs pour assister les navires dans leurs manœuvres d'accostage et d'appareillage.",
    "lamanage": "Le lamanage consiste à amarrer et désamarrer les navires à quai.",
    "manutention": "La manutention regroupe le chargement, le déchargement et le déplacement des marchandises (conteneurs, vrac, etc.).",
    "entreposage": "L'entreposage permet de stocker les marchandises dans des hangars ou sur des aires dédiées avant leur enlèvement ou leur embarquement.",
}
ALIAS_SERVICES = {"pilote": "pilotage", "remorqueur": "remorquage",
                  "stockage": "entreposage", "amarrage": "lamanage"}

NAVIRES = {
    "atlantic star": {"statut": "à quai", "poste": "Poste 12", "eta": "déjà arrivé", "cargaison": "conteneurs"},
    "gulf pearl": {"statut": "en rade", "poste": "en attente", "eta": "accostage prévu demain à 08h00", "cargaison": "hydrocarbures"},
    "sahel express": {"statut": "en route", "poste": "-", "eta": "arrivée prévue dans 2 jours", "cargaison": "riz en vrac"},
}

def _normaliser(texte: str) -> str:
    texte = unicodedata.normalize("NFD", str(texte).lower().strip())
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")

class ActionInfoService(Action):

    def name(self) -> Text:
        return "action_info_service"

    def run(
        self, 
        dispatcher, 
        tracker, 
        domain
        ):
        service = tracker.get_slot("service")
        if service:
            cle = _normaliser(service)
            cle = ALIAS_SERVICES.get(cle, cle)
            if cle in SERVICES:
                dispatcher.utter_message(text=SERVICES[cle])
            else:
                dispatcher.utter_message(
                    text=f"Je n'ai pas d'information sur « {service} ». "
                         f"Services disponibles : {', '.join(SERVICES)}.")
        else:
            liste = "\n".join(f"- {s.capitalize()}" for s in SERVICES)
            dispatcher.utter_message(
                text="Voici nos principaux services :\n" + liste +
                     "\nDemandez-moi le détail d'un service, par exemple « c'est quoi le remorquage ? »")
        return [SlotSet("service", None)]

class ValidateSuiviForm(FormValidationAction):

    def name(self) -> Text: 
        return "validate_suivi_form"

    def validate_navire(self, slot_value, dispatcher, tracker, domain):
        nom = _normaliser(slot_value)
        if nom in NAVIRES:
            return {"navire": nom}
        liste = "\n".join(f"- {n.capitalize()}" for n in NAVIRES)
        dispatcher.utter_message(
                    text="Je ne trouve pas ce navire. Pour la démo, essayez :\n" + liste
                    )
        return {"navire": None}
    
class ActionSuiviNavire(Action):

    def name(self) -> Text:
        return "action_suivi_navire"

    def run(self, dispatcher, tracker, domain):
        nom = tracker.get_slot("navire")
        info = NAVIRES[nom]
        dispatcher.utter_message(
            text=f"Navire {nom.title()} : {info['statut']}.\n"
                 f"- Poste : {info['poste']}\n"
                 f"- Prévision : {info['eta']}\n"
                 f"- Cargaison : {info['cargaison']}")
        return []

class ActionResetSuivi(Action):

    def name(self) -> Text:
        return "action_reset_suivi"

    def run(self, dispatcher, tracker, domain):
        return [SlotSet("navire", None)]
            
        

class ValidateRdvForm(FormValidationAction):
    def name(self)->Text:
        return "validate_rdv_form"

    def validate_nom(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker, 
        domain: DomainDict
    ) -> Dict[Text, Any]:
        nom = str(slot_value).strip()
        if len(nom) < 2:
            dispatcher.utter_message(text="Ce nom semble trop court.")
            return {"nom":None}
        return {"nom":nom.title()}

    def validate_email(
        self, 
        slot_value: Any, 
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict
    ) -> Dict[Text, Any]:
        email = str(slot_value).strip().lower()
        if re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            return {"email": email}
        dispatcher.utter_message(text="Cette adresse email ne semble pas valide.]")
        return {"email":None}

    def validate_motif(
        self, 
        slot_value: Any, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: DomainDict
    )-> Dict[Text, Any]:
        motif = str(slot_value).strip()
        if motif is None: 
            dispatcher.utter_message(text="Vous devez entrer un motif.")
            return {"motif": None}
        return {"motif": motif}

class ActionEnregistrerRdv(Action):

    def name(self) -> Text:
        return "action_enregistrer_rdv"

    def run(
        self, 
        dispatcher, 
        tracker, 
        domain
        ):
        demande = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "nom": tracker.get_slot("nom"),
            "email": tracker.get_slot("email"),
            "motif": tracker.get_slot("motif")
        }

        with open("demandes_rdv.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(demande, ensure_ascii=False) + "\n")
        return []

class ActionResetRdvSlots(Action):
    
    def name(self) -> Text:
        return "action_reset_rdv_slots"

    def run(self, dispatcher, tracker, domain):
        return [SlotSet("nom", None), SlotSet('email', None), SlotSet('motif', None)]

class ActionDoonerHoraires(Action):

    def name(self)->Text: 
        return "action_donner_horaires"

    def run(
        self, 
        dispatcher: CollectingDispatcher, 
        tracker: Tracker, 
        domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        jour = tracker.get_slot("jour")

        if jour is None: 
            dispatcher.utter_message(
                text="Nous sommes ouverts du lundi au vendredi, de 8h à 17h. "
                     "Nous sommes fermés le week-end."
            )

            return []
        
        jour = jour.lower().strip()

        if jour not in HORAIRES:
            dispatcher.utter_message(
                text=f"Je ne connais pas le jour « {jour} ». "
                     "Pouvez-vous préciser (lundi, mardi, etc.) ?"
            )
            return []

        horaire = HORAIRES[jour]
        if horaire is None:
            dispatcher.utter_message(text=f"Le {jour}, nous sommes fermés.")
        else:
            dispatcher.utter_message(
                text=f"Le {jour}, nous sommes ouverts {horaire}."
            )
        return []

class ActionRagDocs(Action):

    def name(self) -> Text:
        return "action_rag_docs"

    def run(self, dispatcher, tracker, domain):
        question = tracker.latest_message.get("text", "")
        resultats = MOTEUR.chercher(question, k=3)

        if not resultats:
            dispatcher.utter_message(
                text="Je n'ai pas trouvé de réponse dans la documentation. "
                     "Vous pouvez reformuler ou demander un rendez-vous.")
            return []

        passages = [texte for (_, texte), _ in resultats]
        reponse = reformuler(question, passages)

        if reponse:
            sources = ", ".join(sorted({source for (source, _), _ in resultats}))
            dispatcher.utter_message(text=f"{reponse}\n\nSource : {sources}")
        else:
            morceaux = "\n\n".join(f"{texte}\n(Source : {fichier})"
                                   for (fichier, texte), _ in resultats)
            dispatcher.utter_message(text="D'après la documentation :\n\n" + morceaux)
        return []
        