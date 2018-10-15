# fantasy-name-classifier

Classify elvish and dwarfish names. See [here](https://www.reddit.com/r/dailyprogrammer_ideas/comments/9o9vtj/hard_tell_elf_names_and_dwarf_names_apart/)

## Install

Tested on Python 3.7.

```
pip install -r requirements.txt
```

## Run

Train model on initial data:

```
python train.py
```

This will serialize a model into the `data` directory for later use.

We can now run this model over any input file in the correct format.

```
python test.py validate.txt
python test.py final.txt
```
