# Novelty-Detection-An-NLP-Perspective
Novelty Detection: An NLP Perspective

## Setup 
### Dataset download
`./scripts/data_init.sh`

### Install requirements
```
  pip install -r requirements.txt    
  conda create -n mul python=3.6      
  conda activate mul     
  python -m spacy download en     
  python -m spacy download en_core_web_lg
```

### Train
```
  python run.py train experiment_configs/esim_snli_config.json serialization_dir/trained_esim_snli
  python run.py train experiment_configs/doc_level_mpe_esim.json serialization_dir/mpe_esim_dlnd
```

