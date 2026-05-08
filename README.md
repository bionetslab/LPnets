## Usage

### 1. Build graphs

- **Build graphs with all possible params configurations:**

python -m lpnets.pipeline.run_pipeline --mode build --build_all chemo_grid

- **Build graphs with specified params configuration:**

python -m lpnets.pipeline.run_pipeline --mode build --cohort mimic_cohort_aplasia_45_days --fold 0 --bin day --time_strategy TS1 --edge_method SSN --agg_method napyPCC



---

### 2. Train models

- **Build graphs with all possible params configurations and train model with all possible training configurations:**

python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid --train_all train_grid

- **Build graphs with all possible params configurations and train model with default configurations:**

python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid

- **Build graphs with all possible params configurations and train model with speficified configurations:**

python -m lpnets.pipeline.run_pipeline --mode train --build_all chemo_grid --significant --zscores --zmode graphwise --threshold p_val

- **Build graphs with speficied configurations and train model with all possible configurations:**

python -m lpnets.pipeline.run_pipeline --mode train --cohort mimic_cohort_aplasia_45_days --fold 0 --bin day --time_strategy TS1 --edge_method SSN --agg_method PCC --train_all train_grid

- **Build graphs with speficied configurations and train model with speficified configurations:**

python -m lpnets.pipeline.run_pipeline --mode train --cohort mimic_cohort_aplasia_45_days --fold 0 --bin day --time_strategy TS1 --edge_method SSN --agg_method PCC --zscores --zmode edgewise


python -m lpnets.pipeline.run_pipeline --mode train --cohort mimic_iii_icu --fold 0 --bin 6hour --time_strategy TS1 --edge_method SSN --agg_method napyPCC --stats mean --zscores --zmode edgewise --subset 1000

