# src/ui/help_button.py
"""Reusable help button widget and help texts for all starBoard tabs."""
from __future__ import annotations

from PySide6.QtWidgets import QPushButton, QMessageBox


class HelpButton(QPushButton):
    """A small '?' button that shows a help popup when clicked."""

    def __init__(self, help_text: str, parent=None):
        super().__init__("?", parent)
        self._help_text = help_text
        self.setFixedSize(22, 22)
        self.setStyleSheet(
            "QPushButton { border-radius: 11px; font-weight: bold; "
            "font-size: 12px; background: #555; color: white; } "
            "QPushButton:hover { background: #777; }"
        )
        self.setToolTip("Click for help")
        self.clicked.connect(self._show_help)

    def _show_help(self):
        QMessageBox.information(self.window(), "Help", self._help_text)


# ─── Help texts for all tabs ───────────────────────────────────────────────

HELP_TEXTS = {
    # ── Sync tab ──
    'connection': 'This is where you set up your link to the central server. Enter the server address and your lab ID, then click Save Config to store these settings. Click Test Connection to verify your machine can reach the server before syncing.',
    'sync_status': 'This panel shows when you last sent or received data, and how much is stored on the central server. Use it to check whether your local data is up to date or if you need to push or pull.',
    'push': 'Click Push All Local Data to send everything on this machine — photos, metadata, and any match decisions you have made — up to the central server. This ensures your fieldwork is backed up and available to other researchers. You can push as often as you like; only new or changed data is sent.',
    'pull': 'Use this section to download data from the central server onto your machine. You can filter by specific individuals, sightings, locations, or date ranges and click Pull Selected Data, or click Pull Everything to grab the entire archive.',
    'catalog': 'Click Refresh Catalog from Server to fetch an up-to-date list of everything stored in the central archive. The table below shows all known individuals and unidentified sightings available on the server, so you can see what is available before pulling.',
    'gallery_filter': 'This is a searchable checklist of known, identified individuals in the central archive. Type to search by ID, then check the boxes for the specific individuals whose photos and data you want to download.',
    'query_filter': 'This is a searchable checklist of unidentified sightings that still need to be matched. Type to search by ID, then check the boxes for the specific sightings you want to download and work on.',
    'location_filter': 'This is a searchable checklist of field locations where observations were recorded. Check one or more locations to download only data collected at those sites.',
    'date_filter': 'Use the After and Before date pickers to download only encounters from a specific time window. For example, set After to the start of your field season and Before to the end to pull just that season\'s data.',

    # ── Data Entry tab ──
    'setup_single_upload': 'Upload photos of a single sea star here. Choose whether it\'s a known individual (gallery) or an unidentified sighting (query), set the encounter date, and add your images. Use this for adding one animal at a time.',
    'setup_outing_entry': 'Record a field outing where you didn\'t find any sea stars. This "negative data" is important for tracking population absence at a site. Enter the date, location, and any notes about conditions.',
    'setup_batch_upload': 'Upload many animals at once by pointing to a folder of subfolders. Each subfolder becomes a new gallery or query ID automatically. Use this after a big dive day when you have photos sorted into folders by individual.',
    'setup_id_management': 'Rename or delete existing gallery and query IDs. Use this to fix typos in names, merge duplicates, or remove entries added by mistake. Renaming updates all linked photos and encounter records.',
    'setup_log_visit': 'Record that you visited a specific location on a given date. This helps track where and when surveys happened, even if no animals were photographed. Essential for presence/absence analyses.',
    'setup_metadata_edit': 'View and edit the physical trait annotations for a selected individual — things like arm count, body colors, and stripe patterns. Select an ID first, then update any of the trait fields. Accurate annotations improve matching results.',
    'setup_merge_archive': 'Import data from another starBoard archive, such as one on a USB drive or shared folder. This copies over gallery entries, queries, and encounter records without overwriting your existing data. Use this to combine archives from different team members.',
    'setup_batch_location': 'Set or update the location for many records at once. Select the IDs you want to edit, choose a location, and apply. Useful when you forgot to set the dive site during upload or need to correct a batch.',

    # ── First-order tab ──
    'first_order_query': 'Pick an unidentified sighting (query) from the list to search for possible matches. The selected query\'s photos and traits will be compared against all known individuals in the gallery. Start here when you want to identify a new sighting.',
    'first_order_fields': 'Choose which physical traits to include in the search and how strict each comparison should be. Wider tolerances return more candidates but may include weaker matches. Adjust these if you\'re getting too many or too few results.',
    'first_order_results': 'This is the ranked list of gallery individuals most similar to your selected query. Top-ranked candidates are the closest matches based on the traits you selected. Pin promising candidates, then switch to the Second-order tab for side-by-side comparison.',
    'first_order_visual': 'These controls blend AI visual similarity with trait-based matching. The Fusion slider sets how much weight goes to the AI model vs. your annotations (0% = traits only, 100% = AI only). Enable Verification for an additional AI check that scores how likely two stars are the same individual.',
    'first_order_gallery': 'The gallery panel shows ranked candidates. Each card displays the individual\'s photos and similarity score. Right-click to pin, exclude, or open the folder. Pinned candidates appear in the Second-order tab for detailed comparison.',

    # ── Second-order tab ──
    'second_order_compare': 'Compare your unidentified sighting and a gallery candidate side by side. Pan and zoom are synchronized so you can inspect the same body region on both animals at once. Decide whether they are the same individual: Yes, No, or Maybe.',
    'second_order_ids': 'Select a Query (unidentified sighting) and a Gallery candidate (known individual) to compare. The Pinned/Maybe dropdown shows candidates you flagged from the First-order tab. Use the arrow buttons to step through queries.',
    'second_order_decision': 'Record your match decision: Yes (same individual — will be merged), Maybe (uncertain — save for later review), or No (different animals). Add notes to explain your reasoning. The verification score shows the AI\'s confidence.',

    # ── Gallery Review tab ──
    'gallery_review_browse': 'Browse all known individuals in the gallery. View photos across encounters, see encounter dates, and manage entries. Filter by location or search by name. You can edit metadata, rename IDs, delete photos, and mark the best photo for each individual.',

    # ── Analytics & History tab ──
    'analytics_visualizations': 'Charts and plots summarizing your match decision history. Use this section to spot patterns in how you\'ve been identifying individuals over time.',
    'analytics_overview': 'A quick summary of all match decisions you\'ve made — how many Yes, No, and Maybe calls across your entire dataset. Check here to see your overall progress.',
    'analytics_match_dynamics': 'Shows how your match decisions have changed over time. Use this to track whether your identification rate is improving as the gallery grows.',
    'analytics_query_gallery': 'Breaks down match attempts by individual query photo or gallery individual. Use this to see which animals have been matched most often or which queries were hardest to resolve.',
    'analytics_workflow': 'Tracks how fast you\'re working — decisions per session, time per match, and overall throughput. Useful for planning how long future annotation sessions will take.',
    'analytics_morphometric': 'Displays trends and distributions of body size measurements across individuals and time. Use this to explore growth patterns or compare sizes across your population.',
    'analytics_merge': 'After confirming a match (YES), use this to officially combine the query sighting into the matched individual\'s record. This updates the gallery so that individual\'s history includes the new encounter.',
    'analytics_revert': 'Undo a previous merge if you realize a match decision was wrong. This separates the query sighting back out from the individual\'s record, restoring both to their original state.',
    'analytics_export': 'Download all your match decisions as a CSV file. Use this to analyze your data in spreadsheets or statistical software outside of starBoard.',

    # ── Deep Learning tab ──
    'dl_status': 'Shows which AI model is currently loaded and ready to use. Check here to confirm the model is active before running searches or precomputation.',
    'dl_model': 'Register new AI model files, switch between available models, or remove old ones. You need at least one model loaded before the AI can help find matching individuals.',
    'dl_precompute': 'Processes all your photos through the AI model ahead of time so that similarity searches run instantly. Run this after adding new images or switching models.',
    'dl_visualization': 'Interactive plots showing how the AI groups your individuals based on visual similarity. Animals the AI sees as similar appear close together — useful for checking if the model makes sense for your population.',
    'dl_evaluation': 'Measures how well the current AI model can tell your individuals apart. Higher scores mean the model is better at distinguishing different animals in your dataset.',
    'dl_verification_eval': 'Tests the pairwise verification model that confirms or rejects proposed matches. Use this to see how accurately the AI can judge whether two photos show the same individual.',
    'dl_finetune': 'Retrain the AI model using your own labeled data to improve accuracy for your specific population. This is advanced — only needed if the default model struggles with your animals.',

    # ── Morphometric tab ──
    'morph_camera': 'Connect to a webcam for live specimen measurement. Select your camera device and adjust the video feed before capturing images.',
    'morph_archive': 'Choose which individual and encounter to save your measurements to. Select the animal from your archive so measurements are linked to the correct record.',
    'morph_saved': 'View previously recorded measurements for the selected individual. Use this to review past size data or compare measurements across encounters.',
    'morph_metadata': 'Add morphological notes and annotations to the current measurement session — such as body condition, color, or abnormalities observed during measurement.',
    'morph_analysis': 'Fine-tune the arm detection parameters for your specimen — adjust smoothing, sensitivity, distance, and arm rotation to get an accurate measurement.',
    'morph_actions': 'Run the detection and measurement analysis on the current camera frame, then save the results to the archive. Use "Run All" to detect and measure in one step, then "Save to starBoard" to store the data.',
}
