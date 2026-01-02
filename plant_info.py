# plant_info.py
# ============================================================
# External Knowledge Enrichment for Medicinal Plants
# ============================================================

import wikipedia

# ------------------------------------------------------------
# Get plant information
# ------------------------------------------------------------
def get_plant_info(plant_name: str) -> str:
    """
    Fetch short description of medicinal plant.
    Priority:
    1. Wikipedia (fallback-safe)
    """

    try:
        wikipedia.set_lang("en")
        summary = wikipedia.summary(plant_name, sentences=3)
        return summary

    except wikipedia.DisambiguationError as e:
        # Pick first suggested option safely
        try:
            summary = wikipedia.summary(e.options[0], sentences=3)
            return summary
        except Exception:
            return "Information is available but could not be resolved automatically."

    except wikipedia.PageError:
        return "No detailed information found for this plant."

    except Exception as e:
        return "Plant information currently unavailable."
