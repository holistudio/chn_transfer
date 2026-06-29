# Makefile for the Traditional/Simplified Chinese character pipeline.
#
# Run from the PROJECT ROOT (the directory that contains data/).
# The pipeline scripts live in data/, so override SCRIPTS if yours differ.
#
#   make            # build everything (only re-runs stale steps)
#   make scrape     # force a fresh re-scrape of the Wikisource table
#   make clean      # delete generated CSVs (raw inputs are left alone)

PYTHON  := python3
SCRIPTS := data
RAW     := data/raw
PROC    := data/processed
CLEAN   := data/clean

.PHONY: all scrape clean
.DEFAULT_GOAL := all

# Final deliverable
all: $(CLEAN)/trad_simpl_clean.csv

# --- Stage 1: independent sources -------------------------------------------

# Traditional frequency + stroke data
$(PROC)/trad_usenet_freq.csv: $(SCRIPTS)/trad_usenet.py $(RAW)/trad_usenet_freq.txt | $(PROC)
	$(PYTHON) $(SCRIPTS)/trad_usenet.py

# Simplified frequency data
$(PROC)/simpl_freq.csv: $(SCRIPTS)/simpl_freq.py $(RAW)/sim_modern_freq.tsv | $(PROC)
	$(PYTHON) $(SCRIPTS)/simpl_freq.py

# Simplified/Traditional equivalents, scraped from Wikisource.
# No local prerequisite, so this only runs when the output is missing.
# Use `make scrape` to force a refresh.
$(PROC)/simpl_equivalent.csv: $(SCRIPTS)/simpl_equivalent.py | $(PROC)
	$(PYTHON) $(SCRIPTS)/simpl_equivalent.py

# --- Stage 2: traditional ids -----------------------------------------------

$(PROC)/trad_id_freq.csv: $(SCRIPTS)/trad_id_freq.py \
                          $(PROC)/trad_usenet_freq.csv \
                          $(RAW)/trad_ministry_id.tsv | $(PROC)
	$(PYTHON) $(SCRIPTS)/trad_id_freq.py

# --- Stage 3: aggregate -----------------------------------------------------

$(PROC)/trad_simpl_agg.csv: $(SCRIPTS)/agg_table.py \
                            $(PROC)/trad_id_freq.csv \
                            $(PROC)/simpl_equivalent.csv \
                            $(PROC)/simpl_freq.csv | $(PROC)
	$(PYTHON) $(SCRIPTS)/agg_table.py

# --- Stage 4: clean / final -------------------------------------------------

$(CLEAN)/trad_simpl_clean.csv: $(SCRIPTS)/drop_final.py \
                               $(PROC)/trad_simpl_agg.csv
	$(PYTHON) $(SCRIPTS)/drop_final.py

# --- Helpers ----------------------------------------------------------------

# Order-only target: ensure the processed dir exists before any script writes.
# (drop_final.py already creates data/clean itself.)
$(PROC):
	mkdir -p $(PROC)

# Force a fresh re-scrape regardless of whether the CSV already exists.
scrape:
	$(PYTHON) $(SCRIPTS)/simpl_equivalent.py

# Remove generated files only.
clean:
	rm -f $(PROC)/trad_usenet_freq.csv \
	      $(PROC)/simpl_freq.csv \
	      $(PROC)/simpl_equivalent.csv \
	      $(PROC)/trad_id_freq.csv \
	      $(PROC)/trad_simpl_agg.csv \
	      $(CLEAN)/trad_simpl_clean.csv
